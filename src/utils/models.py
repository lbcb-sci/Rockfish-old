from __future__ import annotations

import h5py
import numpy as np
import io
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import re

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import namedtuple
from enum import Enum

from typing import Dict, Set, List, Tuple, Any, Callable, Optional

base_encoding = {k: v for v, k in enumerate('ACGTN')}  # Label encoding for bases

Interval = namedtuple('Interval', ['start', 'end'])  # Start - inclusive, end - exclusive

DEFAULT_RESEGMENTATION_PATH = 'Analyses/RawGenomeCorrected_000/BaseCalled_template'  # Default re-segmentation path


class Strand(Enum):
    """ Enumeration that describes strand information. """
    FORWARD = '+'
    REVERSE = '-'

    @staticmethod
    def strand_from_str(string: str) -> Strand:
        """ Static function that returns strand for the given string.

        :param string: String describing strand
        :return: Strand for the given string
        """
        if string == '+':
            return Strand.FORWARD
        if string == '-':
            return Strand.REVERSE

        raise ValueError(f'{string} is an invalid value for strand.')


@dataclass(eq=True, frozen=True)
class GenomicPos:
    """ Data class that describes genomic position.

    Genomic position is uniquely defined with chromosome, position on the chromosome and strand.
    """
    chromosome: str
    position: int
    strand: Strand


@dataclass(frozen=True)
class ModInfo:
    """ Data class that describes modification information.

    Data class contains information about coverage and modification frequency (usually for genomic position).
    """
    n_reads: int
    mod_freq: int


BEDData = Dict[GenomicPos, ModInfo]  # BED file holds modification information for genomic positions
BEDPredicate = Callable[[GenomicPos, ModInfo], bool]  # Declaration for BED filter function


@dataclass
class ResegmentationData:
    position: int
    event_intervals: List[Interval]  # Interval of signal points
    event_lens: np.ndarray
    bases: str


@dataclass
class BasecallData:
    signal_point: int
    base: str
    quality: int


NormFunc = Callable[[np.ndarray], np.ndarray]  # Declaration for normalization function


from functools import lru_cache

@lru_cache(maxsize=None)
def linspace(step, samples=20):
    return [int(i * step) for i in range(0, samples)]


def standardization(signal: np.ndarray) -> np.ndarray:
    """ Normalization function that performs standardization.

    Function normalizes given array by subtracting its mean and dividing by standard deviation.

    :param signal: Array containing read signal points
    :return: Standardized signal
    """
    return (signal - np.mean(signal)) / np.std(signal)


def mad(signal: np.ndarray) -> np.ndarray:
    """ TODO
    """
    centered = signal - np.median(signal)
    return centered / np.median(np.abs(centered))


def normalization_factory(name: Optional[str]='standardization') -> Optional[NormFunc]:
    """ Factory method that returns normalization function for the given name.

    :param name: String corresponding to normalization method
    :return: Normalization method for the given string or None if normalization is not wanted
    """
    if not name or name == 'None':
        return None
    if name == 'standardization':
        return standardization
    if name == 'mad':
        return mad

    raise ValueError('Normalization method not recognized.')


class Read:
    """ Class that describes single FAST5 read. """
    def __init__(self, fd: h5py.File, normalization: Optional[str]=None) -> None:
        """ Constructs instance of Read class.

        Constructs instance of Read class from opened h5py file and optionally given normalization method.
        Multi FAST5 files should be converted to single FAST5 before processing.

        :param fd: Opened h5py file that corresponds to single FAST5 read
        :param normalization: Optional normalization method used for signal normalization
        """
        self.fd = fd

        # TODO Maybe change, needed only for Albacore
        self.sampling_rate = self.fd['UniqueGlobalKey/channel_id'].attrs['sampling_rate']
        self.scale, self.offset = self.calculate_norm_params()

        read_group_name = list(self.fd['Raw/Reads'].keys())[0]
        self.read_group_path = f'Raw/Reads/{read_group_name}'

        self.start_time = self.fd[self.read_group_path].attrs['start_time']

        self.norm_func = normalization_factory(normalization)
        self.signal = self.get_raw_signal()

    def get_read_id(self) -> str:
        """ Returns read id

        :return: read id for the given single FAST5 file
        """
        return self.fd[self.read_group_path].attrs['read_id'].decode()

    def calculate_norm_params(self) -> Tuple[int, int]:
        """ Calculates parameters for converting discrete to continuous signal.

        :return: Scale and offset used for converting discrete to continuous signal
        """
        cid = self.fd['UniqueGlobalKey/channel_id']

        scale = cid.attrs['range'] / cid.attrs['digitisation']
        offset = cid.attrs['offset']

        return scale, offset

    def get_raw_signal(self, continuous: bool=True) -> np.ndarray:
        """ Returns the raw signal for the read.

        Returns the raw signal for the given read. Optionally, it converts discrete signal to continuous signal.

        :param continuous: True if returned signal should be continuous, otherwise False
        :return: Raw signal for the given read
        """

        signal = self.fd[f'{self.read_group_path}/Signal'][()]

        if continuous:
            signal = self.scale * (signal + self.offset)
        if self.norm_func:
            signal = self.norm_func(signal)

        return signal

    def signal_for_interval(self, interval: Interval, samples: int=20) -> Tuple[np.ndarray, np.ndarray]:
        """ Function that samples signal points from the given interval.

        This function is used for sampling signal points from the given interval. If desired number of points is lower
        than number of points in the given interval, some points are discarded. If desired number of points is higher,
        some of the points are repeated. Sampling is done by finding point indices using numpy linspace function.

        :param interval: Interval from which points will be sampled
        :param samples: Number of sampled points
        :return: Tuple of sampled points and relative indices in the given interval
        """
        # points = self.signal[interval.start:interval.end]

        if samples is None:
            return self.signal[interval.start:interval.end]

        step = (interval.end - interval.start) / samples
        idx = linspace(step, samples)

        # idx = np.linspace(0, len(points), samples, endpoint=False, dtype=int)
        absolute_indices = interval.start + idx
        return self.signal[absolute_indices], absolute_indices


class BasecallerProcessor(ABC):
    def __init__(self, read: Read, path: str='Analyses/Basecall_1D_000') -> None:
        self.read = read
        self.group_path = path

        self.fastq = self.read_fastq()

    def read_fastq(self, decode: bool=False) -> SeqRecord:
        fastq_str = self.read.fd[f'{self.group_path}/BaseCalled_template/Fastq'][()]
        fastq_str = fastq_str.decode()
        str_file = io.StringIO(fastq_str)

        record = SeqIO.read(str_file, 'fastq')
        return record

    @staticmethod
    def get_processor(read: Read, path: Optional[str]=None):
        if f'{path}/BaseCalled_template/Events' in read.fd:
            return AlbacoreProcessor(read, path)
        return GuppyProcessor(read, path)


class AlbacoreProcessor(BasecallerProcessor):
    def __init__(self, read: Read, path: Optional[str]=None) -> None:
        if path is None:
            super().__init__(read)
        else:
            super().__init__(read, path)

        self.event_table = read.fd[f'{self.group_path}/BaseCalled_template/Events'][()]
        self.relative_start = self.__calculate_raw_start()

    def __calculate_raw_start(self) -> int:
        e_start = self.event_table['start'][0]

        abs_e_start = np.round(e_start.astype(np.float64) * self.read.sampling_rate).astype(np.uint64)
        read_start = self.read.start_time
        read_start_rel_to_raw = int(abs_e_start - read_start)

        if read_start_rel_to_raw < 0:
            if read_start_rel_to_raw >= -2:
                read_start_rel_to_raw = 0
            else:
                raise ValueError('Events cannot start before read.')

        return read_start_rel_to_raw

    def map_signal_to_bases(self, points_indices: Set[int]=None) -> Dict[int, BasecallData]:
        fastq = self.fastq
        qualities = fastq.letter_annotations['phred_quality']

        points_to_data = {}

        start, base_index = self.relative_start, 2  # We take quality for central base
        for event_data in self.event_table:
            event_length = int(event_data['length'] * self.read.sampling_rate)
            end = start + event_length
            base_index += event_data['move']

            for idx in range(start, end):
                if points_indices is not None and idx not in points_indices:
                    continue

                bc_data = BasecallData(idx, event_data['model_state'].decode()[2], qualities[base_index])
                points_to_data[idx] = bc_data

            start = end

        return points_to_data


class GuppyProcessor(BasecallerProcessor):
    def __init__(self, read: Read, path: Optional[str]=None, seg_path: Optional[str]=None) -> None:
        if path is None:
            super().__init__(read)
        else:
            super().__init__(read, path)

        if seg_path is None:
            self.seg_path = 'Analyses/Segmentation_000'
        else:
            self.seg_path = seg_path

    def __get_fast5_data(self) -> Tuple[int, np.ndarray, int]:
        start = self.read.fd[f'{self.seg_path}/Summary/segmentation'].attrs['first_sample_template']
        move_table = self.read.fd[f'{self.group_path}/BaseCalled_template/Move'][()]
        block_stride = self.read.fd[f'{self.group_path}/Summary/basecall_1d_template'].attrs['block_stride']

        return int(start), move_table, int(block_stride)

    def map_signal_to_bases(self, points_indices: Optional[Set[int]]=None) -> Dict[int, BasecallData]:
        basecall_data = {}

        fastq_seq, fastq_quals = str(self.fastq.seq), self.fastq.letter_annotations['phred_quality']
        start, move_table, block_stride = self.__get_fast5_data()
        fastq_offset = np.cumsum(move_table) - 1

        block_start = start
        for fastq_idx in fastq_offset:
            block_end = block_start + block_stride

            for point_idx in range(block_start, block_end):
                point_data = BasecallData(point_idx, fastq_seq[fastq_idx], fastq_quals[fastq_idx])
                basecall_data[point_idx] = point_data

            block_start = block_end

        return basecall_data


class ResegmentationProcessor:
    """ Class used for processing resegmentation data in FAST5 read. """
    def __init__(self, read: Read, path: str=DEFAULT_RESEGMENTATION_PATH) -> None:
        """ Creates instance of resegmentation processor for the given read.

        Constructs instance of resegmentation processor for the given read. Optionally, different path to the
        resegmentation data can be given.

        :param read: Read for which resegmentation data will be extracted
        :param path: Path to the resegmentation data in FAST5 file
        """
        self.read = read

        self.group_path = path
        self.events_path = f'{path}/Events'
        self.alignment_path = f'{path}/Alignment'

    def get_event_table(self) -> np.ndarray:
        """ Returns the event table for the given read.

        :return: Event table for the given read.
        """
        return self.read.fd[self.events_path][()]

    def get_relative_to_raw(self) -> int:
        """ Returns starting signal point index for the first re-segmented base.

        :return: Starting signal point index for the first re-segmented base
        """
        return self.read.fd[self.events_path].attrs['read_start_rel_to_raw']

    def motif_positions(self, motif: str='CG', window: int=8, bed_pos: Set[GenomicPos]=None) -> Set[Interval]:
        """ Function returns intervals that correspond to the given motif.

        This function extracts intervals that contain the given motif. Start of the motif is the central
        index in the interval. If genomic positions are provided, only positions included in the given set will be
        returned.

        :param motif: Motif to be searched for in the re-segmented data
        :param window: Left and right window size around the motif
        :param bed_pos: Optional set of genomic positions. Only intervals that have central position
        in the set will be included
        :return: Intervals that correspond to the given motif, optionally filtered
        """
        event_table = self.get_event_table()
        table_len = len(event_table)
        seq = ''.join([b.decode() for b in event_table['base']])

        alignment_data = self.get_alignment_data()
        ctg = alignment_data.chromosome
        strand = alignment_data.strand

        event_indices = set()
        for match in re.finditer(motif, seq):
            center = match.start()
            if alignment_data.strand == Strand.FORWARD:
                position = alignment_data.position + center
            else:
                position = alignment_data.position + (table_len - center - 1)

            # Filtering positions that are not in BED
            if bed_pos is not None and GenomicPos(ctg, position, strand) not in bed_pos:
                continue

            start, end = center - window, center + window + 1

            if start >= 0 and end <= table_len:
                event_indices.add(Interval(start, end))

        return event_indices

    def get_alignment_data(self) -> GenomicPos:
        """ Returns alignment information for the given read.

        :return: Genomic position object that contains the name of the reference chromosome, starting reference position
         and strand
        """
        align_group = self.read.fd[self.alignment_path]

        chromosome = align_group.attrs['mapped_chrom']
        start = align_group.attrs['mapped_start']
        # end = align_group.attrs['mapped_end']
        strand = Strand.strand_from_str(align_group.attrs['mapped_strand'])

        return GenomicPos(chromosome, start, strand)

    def get_resegmentation_data(self, event_positions: Set[Interval]) -> List[ResegmentationData]:
        """ Returns the list of resegmentation data for the given intervals.

        :param event_positions: Set containing relevant intervals
        :return: List of resegmentation data for the given intervals
        """
        rel_to_raw = self.get_relative_to_raw()
        event_table = self.get_event_table()
        event_table['start'] += rel_to_raw
        event_table_len = len(event_table)
        seq = ''.join([b.decode() for b in event_table['base']])

        alignment_data = self.get_alignment_data()

        resegmentation_data = []
        for start, end in event_positions:
            intervals = [Interval(e['start'], e['start'] + e['length']) for e in event_table[start:end]]
            event_lens = event_table[start:end]['length']
            bases = seq[start:end]

            assert len(intervals) == len(bases)

            center_idx = start + len(bases) // 2
            if alignment_data.strand == Strand.FORWARD:
                position = alignment_data.position + center_idx
            else:
                position = alignment_data.position + (event_table_len - center_idx - 1)

            reseg_data = ResegmentationData(position, intervals, event_lens, bases)
            resegmentation_data.append(reseg_data)

        return resegmentation_data


@dataclass
class Example:
    """ Data class that contains information about one example.

    This data class stores information about example's position on the reference genome, reference k-mer around the
    position and sampled signal points.
    """
    position: int
    ref_kmer: str
    signal_points: np.ndarray


"""@dataclass
class ExampleData:
    pos: int
    read_bases: str
    base_qualities: List[int]  # dim = k-mer * sample_size
    ref_kmer: str  # k-mer
    signal_points: np.ndarray  # dim = k-mer * sample_size
    event_lens: np.ndarray"""


@dataclass
class FeaturesData:
    """ Data class that contains examples for the given read.

    This data class stores information about read identifier, reference chromosome, mapping strand and the list of
    examples corresponding to this read.
    """
    chromosome: str
    strand: Strand
    read_id: str
    examples: List[Example]


'''class NanoRawProcessor(ResegmentationProcessor):
    def __init__(self, read: Read, path: str=None) -> None:
        if path is None: super().__init__(read)
        else: super().__init__(read, path)'''
