from __future__ import annotations

import sys
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
import traceback
from concurrent.futures import as_completed
from pyguppyclient.decode import ReadData, CalledReadData

import argparse

from utils.models import *
from utils.bed_processing import bed_filter_factory, extract_bed_positions
from utils.parallel_processing import get_executor
from utils.writer import BinaryWriter

from processor.basecall import basecall
from processor.custom_processor import CustomProcessor
from processor.util import Interval

BATCH_SIZE = 4_000


def get_files(path: Path, recursive: bool = False) -> List[Path]:
    if path.is_file():
        return [path]

    # Finding all input FAST5 files
    if recursive:
        files = path.glob('**/*.fast5')
    else:
        files = path.glob('*.fast5')

    return list(files)


def get_info_args(args: argparse.Namespace) -> Dict[str, Any]:
    """ Function that extracts arguments relevant for data generation.

    This function extracts relevant arguments for data generation and optionally label (if label is defined at data
    level).

    :param args: Script arguments
    :return: Key-value pair of arguments relevant for data generation and optionally defined label
    """
    info_args = {
        'norm_method': args.norm_method,
        'mapq': args.mapq,
        'motif': args.motif,
        'index': args.index,
        'window': args.window,
        'label': args.label
    }

    return info_args


def get_raw_signal(read: ReadData, event_interval: Interval, norm_method, continuous: bool=True) -> np.ndarray:
    """ Returns the raw signal for the read.

    Returns the raw signal for the given read. Optionally, it converts discrete signal to continuous signal.

    :param read: Basecalled read
    :param event_interval: Interval of signal points
    :param norm_method: Signal normalization method
    :param continuous: True if returned signal should be continuous, otherwise False
    :return: Raw signal for the given read
    """
    signal = read.signal[event_interval.start: event_interval.end]

    if continuous:
        signal = read.daq_scaling * (signal + read.daq_offset)

    norm_func = normalization_factory(norm_method)
    if norm_func:
        signal = norm_func(signal)

    return signal


def generate_data(read: ReadData,
                  reseg_data: List[ResegmentationData],
                  norm_method: str) -> Optional[List[Example]]:
    """ Function that generates data for the specified read.

    This function generates list of examples for the given read. Data is generated from raw signal for the given read,
    and information extracted from resegmentation data.

    :param read: Read for which examples will be generated
    :param reseg_data: Resegmentation data
    :param norm_method: Signal normalization method
    :return: List of generated examples for the given read and resegmentation data
    """
    all_examples = []

    for reseg_example in reseg_data:
        example_points = []

        for interval in reseg_example.event_intervals:
            points = get_raw_signal(read, interval, norm_method)
            example_points.append(points)

        example_points = np.concatenate(example_points)

        example = Example(reseg_example.position, reseg_example.bases, example_points)
        all_examples.append(example)

    if len(all_examples) == 0:
        return None

    return all_examples


def process_read(
        basecall_data: Tuple[ReadData, CalledReadData],
        reference: str,
        norm_method: Optional[str]='standardization',
        mapq: int = 10,
        motif: str='CG',
        index: int=0,
        window: int=8,
        bed_pos: Optional[Set[GenomicPos]]=None) -> Optional[FeaturesData]:
    """ This function processes the given read to generate data.

    This function extracts resegmentation information, and extracts alignment data.

    :param basecall_data: Basecalled data: (read, called)
    :param reference: Path to the reference file
    :param norm_method: Signal normalization method
    :param mapq: Mapping quality threshold
    :param motif: Motif for which positions will be extracted
    :param index: Index of the central position in the motif
    :param window: Size of left and right windows around central position
    :param bed_pos: Positions used for filtering motif positions
    :return: FeaturesData object if at least one example is present, otherwise None
    """
    processor = CustomProcessor(basecall_data, reference, mapq, motif, index, window)
    resegmentation_data = processor.process()
    if not resegmentation_data:
        return

    read, called = basecall_data

    examples = generate_data(read, resegmentation_data, norm_method)
    if not examples:
        return

    align_data = processor.align(called.seq)
    strand = Strand.strand_from_str('+' if align_data.strand == 1 else '-')

    return FeaturesData(align_data.ctg, strand, read.read_id, examples)


def error_callback(path, exception):
    print(f'Error for file: {path}.', file=sys.stderr)
    print(str(exception), file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)


def init_workers(
        info_args: Dict[str, Any],
        norm_method: Optional[str]='standardization',
        mapq: int=10,
        motif: str='CG',
        index: int=0,
        window: int=8,
        bed_data: Optional[BEDData]=None) -> None:
    # Initializing workers with constant arguments
    global _INFO_ARGS, _NORM_METHOD, _MAPQ, _MOTIF, _INDEX, _WINDOW, _BED_DATA
    _INFO_ARGS = info_args
    _NORM_METHOD = norm_method
    _MAPQ = mapq
    _MOTIF = motif
    _INDEX = index
    _WINDOW = window
    _BED_DATA = bed_data


def worker_process_reads(paths: List[Path], reference: str, out_path: Path) -> Tuple[Path, int]:
    """ Function that processes input file list and stores generated data

    :param paths: List of input files that will be processed
    :param reference: Path to the reference file
    :param out_path: Path to the generated output
    :return: Path and number of failed files
    """
    error_files = 0

    with BinaryWriter(str(out_path)) as writer:
        bed_pos = set(_BED_DATA.keys()) if _BED_DATA is not None else None

        for path in tqdm(basecall(paths)):
            try:
                result = process_read(path, reference, _NORM_METHOD, _MAPQ, _MOTIF, _INDEX, _WINDOW, bed_pos)

                if result is not None:
                    writer.write_data(result, _BED_DATA, _INFO_ARGS['label'])
            except Exception as e:
                # error_callback(path, e)
                error_files += 1

        return out_path, error_files


def tqdm_with_time(msg, last_action_time):
    # Prints message with the difference between current time and last action time
    current_time = time.time()
    tqdm.write('>> ' + msg + f' {current_time - last_action_time}s')

    return current_time


def process_data(args: argparse.Namespace) -> None:
    start_time = time.time()
    last_action_time = start_time

    tqdm.write('>> Generating file list')
    files = get_files(args.input_path, args.recursive)
    if len(files) == 0:
        sys.exit('FAST5 file(s) not found.')

    info_args = get_info_args(args)
    last_action_time = tqdm_with_time(f'Parameters: {info_args}', last_action_time)

    if args.bed_path is None:
        bed_info = None
    else:
        last_action_time = tqdm_with_time('Extracting BED info', last_action_time)

        filter_method = bed_filter_factory(args.bed_filter)
        bed_info = extract_bed_positions(args.bed_path, filter_method)

    # Create output dir if it doesn't exist
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Writing extraction info
    info_path = Path(args.output_path, 'info.txt')
    BinaryWriter.write_extraction_info(info_path, info_args)

    workers_args = (info_args, args.norm_method, args.mapq, args.motif, args.index, args.window, bed_info)
    with get_executor(args.workers, initializer=init_workers, initargs=workers_args) as executor:
        futures = []

        batches = (len(files) // BATCH_SIZE)
        batches = batches if len(files) % BATCH_SIZE == 0 else batches + 1  # e.g. 8000//4000 = 2, no need for +1

        last_action_time = tqdm_with_time('Building jobs', last_action_time)
        for i in tqdm(range(batches)):
            start = i * BATCH_SIZE
            end = min(start+BATCH_SIZE, len(files))

            out_file_path = Path(args.output_path, f'{i+1}.bin.tmp')

            future = executor.submit(worker_process_reads, files[start:end], args.reference, out_file_path)
            futures.append(future)

        tqdm_with_time('Extracting features', last_action_time)
        for future in tqdm(as_completed(futures), total=len(futures)):
            out_path, errors = future.result()
            tqdm.write(f'>> File {out_path}: Total {errors} errors.')

    # Concatenate files
    BinaryWriter.on_extraction_finish(path=args.output_path)

    tqdm.write(f'Data generation finished. Total time: {time.time() - start_time}s')


def create_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Input and output arguments
    parser.add_argument('input_path', type=Path,
                        help='Path to the input file or folder containing FAST5 files')

    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Flag to indicate if folder will be searched recursively (default: False)')

    parser.add_argument('output_path', type=Path,
                        help='Path to the desired output folder')

    # Other arguments
    parser.add_argument('--reference', type=str, required=True,
                        help='Path to the reference file')

    parser.add_argument('--norm_method', type=str, default='standardization',
                        help='Function name to use for signal normalization (default: standardization)')

    parser.add_argument('--mapq', type=int, default=10,
                        help='Mapping quality threshold (default: 10)')

    parser.add_argument('--motif', type=str, default='CG',
                        help='Motif to be searched for in the sequences. Regular expressions can be used (default: CG)')

    parser.add_argument('--index', type=int, default=0,
                        help='Index of the central position in the motif (default: 0)')

    parser.add_argument('--window', type=int, default=8,
                        help='Window size around central position. Total k-mer size is: K = 2*W + 1 (default: 8)')

    parser.add_argument('--label', type=int, default=None,
                        help='Label to store for the given examples (default: None)')

    parser.add_argument('-t', '--workers', type=int, default=0,
                        help='Number of workers used for data generation (default: 0)')

    # Bisulfite BED file arguments
    parser.add_argument('--bed_path', type=Path, default=None,
                        help='Path to BED file containing modification information (default: None)')

    parser.add_argument('--bed_filter', type=str, default=None,
                        help='BED filter method (e.g. high_confidence finds only high-confidence positions) (default: None)')

    return parser.parse_args()


if __name__ == '__main__':
    arguments = create_arguments()

    process_data(arguments)
