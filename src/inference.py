from multiprocessing.context import Process
import torch
from torch.nn import DataParallel
from torch.utils.data import IterableDataset, DataLoader

import numpy as np

from tqdm import tqdm

import os
import sys
import math
from pathlib import Path

import multiprocessing as mp

import traceback

import argparse

from extract_features import process_read
from utils.models import base_encoding, DEFAULT_RESEGMENTATION_PATH

from train import Rockfish


TMP_PATH = '{final}.{id}.tmp'

torch.backends.cudnn.benachmark = True


def error_callback(path, exception):
    print(f'Error for file: {path}.', file=sys.stderr)
    print(str(exception), file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)


def get_files(path, recursive, extension='.fast5'):
    if path.is_file():
        return [path]

    if recursive: files = path.glob(f'**/*{extension}')
    else: files = path.glob(f'*{extension}')

    return list(files)


def output_worker(temp_path, queue):
    with open(temp_path, 'w') as f:
        while True:
            batch_result = queue.get()
            if batch_result is None:
                break

            (ctgs, names, strands, positions), pred = batch_result
            for ctg, name, strand, pos, prob in zip(ctgs, names, strands, positions, pred):
                mod = 1 if prob > 0 else 0
                print(ctg, name, pos, strand, prob, mod, file=f, sep='\t')


class Fast5Data(IterableDataset):
    def __init__(self, path,
                recursive=False, reseg_path=DEFAULT_RESEGMENTATION_PATH,
                norm_method='standardization', motif='CG', sample_size=20,
                window=8):
        super().__init__()

        self.files = get_files(path, recursive)

        self.reseg_path = reseg_path
        self.norm_method = norm_method
        self.motif = motif
        self.sample_size = sample_size
        self.window = window

    def __iter__(self):
        for file in self.files:
            try:
                data = process_read(file, self.reseg_path, self.norm_method, self.motif,
                                    self.sample_size, self.window, None)
                if data is None:
                    continue
            except Exception as e:
                # error_callback(file, e)
                continue

            chrom = data.chromosome
            strand = data.strand.value
            read_id = data.read_id

            for example in data.examples:
                pos = str(example.position)
                points = torch.from_numpy(example.signal_points.astype(np.float16))

                kmer = np.repeat([base_encoding[c] for c in example.ref_kmer], self.sample_size)
                kmer = torch.from_numpy(kmer)

                yield (chrom, read_id, strand, pos), points, kmer


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    files = dataset.files  # the dataset copy in this worker process

    per_worker = int(math.ceil(len(files) / float(worker_info.num_workers)))

    start = worker_info.id * per_worker
    end = min(start + per_worker, len(files))
    dataset.files = files[start:end]


def test(args):
    model = Rockfish.load_from_checkpoint(checkpoint_path=args.checkpoint)
    model.freeze()

    test_ds = Fast5Data(args.test_path, args.recursive, args.reseg_path,
                    args.norm_method, args.motif, args.sample_size, args.window)

    if args.n_workers > 0:
        test_dl = DataLoader(test_ds, batch_size=args.batch_size,
                        num_workers=args.n_workers, pin_memory=True,
                        worker_init_fn=worker_init_fn,
                        prefetch_factor=args.prefetch_factor)
    else:
        test_dl = DataLoader(test_ds, batch_size=args.batch_size,
                        num_workers=args.n_workers, pin_memory=True,
                        worker_init_fn=worker_init_fn)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        model = DataParallel(model, device_ids=list(range(n_gpus)))
        model.to(f'cuda:{model.device_ids[0]}')

    model.eval()

    output_queue = mp.Queue()
    consumers = []
    abs_out_path = str(args.out_path.absolute())
    for i in range(args.output_workers):
        worker_path = TMP_PATH.format(final=abs_out_path, id=i)
        process = Process(target=output_worker, args=(worker_path, output_queue))
        process.start()

        consumers.append(process)

    with torch.no_grad():
        for info, sig, k_mer in tqdm(test_dl):
            pred = model(sig, k_mer).squeeze(-1)
            pred = pred.cpu().numpy()

            output_queue.put((info, pred))

    for _ in range(len(consumers)):
        output_queue.put(None)
    for c in consumers:
        c.join()

    with args.out_path.open('w') as out:
        for i in range(len(consumers)):
            worker_path = TMP_PATH.format(final=abs_out_path, id=i)
            with open(worker_path, 'r') as tmp_f:
                out.write(tmp_f.read())
            os.remove(worker_path)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('test_path', type=Path)
    parser.add_argument('out_path', type=Path)

    parser.add_argument('-r', '--recursive', action='store_true')

    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('-t', '--n_workers', type=int, default=0)
    parser.add_argument('--output_workers', type=int, default=1)
    parser.add_argument('--prefetch_factor', type=int, default=4)

    parser.add_argument('--reseg_path', type=str, default=DEFAULT_RESEGMENTATION_PATH,
                        help='''Path to resegmentation group in FAST5 file
                (default: Analyses/RawGenomeCorrected_000/BaseCalled_template)''')
    parser.add_argument('--norm_method', type=str, default='standardization',
                        help='Function name to use for signal normalization (default: standardization)')
    parser.add_argument('--motif', type=str, default='CG',
                        help='''Motif to be searched for in the sequences.
                Regular expressions can be used. (default: CG)''')
    parser.add_argument('--sample_size', type=int, default=20,
                        help='Sample size for every base in the given k-mer. (default: 20)')
    parser.add_argument('--window', type=int, default=8,
                        help='''Window size around central position.
                Total k-mer size is: K = 2*W + 1. (default: 8)''')

    return parser.parse_args()


def main():
    args = get_arguments()

    test(args)


if __name__ == "__main__":
    main()
