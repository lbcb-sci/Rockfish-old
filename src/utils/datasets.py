import argparse
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.distributed as dist

from pytorch_lightning import LightningDataModule

import numpy as np
import io
import os

from tqdm import tqdm


# DATA_DTYPE = np.dtype([('signal', np.float16, (340,)), ('kmer', np.uint8, (17,)), ('label', np.uint8)])


def dataset_size(path):
    with io.open(path.replace('data.bin', 'header.bin'), 'rb') as f:
        signal = np.fromfile(f, dtype=np.uint16)

    return len(signal)


def different_length_error(filename, read_name):
    msg = f'Different number of examples for read {read_name} in file {filename}.'
    raise RuntimeError(msg)


class MemoryDataset(Dataset):
    def __init__(self, path):
        self.data = io.open(path, 'rb')

        with open(path.replace('data.bin', 'info.txt')) as f:
            window = int(f.readlines()[4].split()[1])
            self.kmer = window * 2 + 1

        with io.open(path.replace('data.bin', 'header.bin'), 'rb') as f:
            self.signal = np.fromfile(f, dtype=np.uint16)

        self.example_bytes = self.signal * 2 + self.kmer * 3 + 1  # Lengths of examples in bytes
        self.offset = np.concatenate(([0], self.example_bytes)).cumsum()[:-1]

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, idx):
        data_dtype = np.dtype([('signal', np.float16, (self.signal[idx],)),
                               ('lens', np.uint16, (self.kmer,)),
                               ('kmer', np.uint8, (self.kmer,)),
                               ('label', np.uint8)])

        self.data.seek(self.offset[idx])
        example = np.frombuffer(self.data.read(self.example_bytes[idx]), data_dtype)[0]

        signal = torch.from_numpy(example['signal'])
        lens = example['lens']
        bases = torch.from_numpy(np.repeat(example['kmer'], lens).astype(int))
        label = example['label']

        return signal, bases, label


# def iter_worker_init(worker_id):
#     worker_info = torch.utils.data.get_worker_info()
#     dataset = worker_info.dataset
#
#     dataset.f = io.open(dataset.path, 'rb')
#
#
# class IterDataset(Dataset):
#     def __init__(self, path):
#         self.len = dataset_size(path)
#
#         self.path = path
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, idx):
#         self.f.seek(idx * DATA_DTYPE.itemsize)
#         example = np.frombuffer(self.f.read(DATA_DTYPE.itemsize), dtype=DATA_DTYPE)[0]
#
#         signal = torch.from_numpy(example['signal'])
#         bases = torch.from_numpy(np.repeat(example['kmer'], 20).astype(int))
#         label = example['label']
#
#         return signal, bases, label


class RockfishDataModule(LightningDataModule):
    def __init__(self, train_path, val_path, train_batch_size, val_batch_size, iterable):
        super().__init__()

        self.train_path = train_path
        self.val_path = val_path

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.iterable = iterable

        self.train_len = dataset_size(train_path)

    def train_dataloader(self):
        worker_init_fn = None
        # if self.iterable:
        #     train_ds = IterDataset(self.train_path)
        #     worker_init_fn = iter_worker_init
        # else:
        train_ds = MemoryDataset(self.train_path)

        return DataLoader(train_ds, self.train_batch_size, True,
                          num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

    def val_dataloader(self):
        worker_init_fn = None
        # if self.iterable:
        #     val_ds = IterDataset(self.val_path)
        #     worker_init_fn = iter_worker_init
        # else:
        val_ds = MemoryDataset(self.val_path)

        return DataLoader(val_ds, self.val_batch_size, num_workers=4,
                          pin_memory=True, worker_init_fn=worker_init_fn)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('train_path', type=str)
        parser.add_argument('val_path', type=str)

        parser.add_argument('-b', '--train_batch_size', type=int, default=128)
        parser.add_argument('--val_batch_size', type=int, default=1024)
        parser.add_argument('--iterable', action='store_true')

        return parser

    @classmethod
    def from_argparse_args(cls, args):
        return RockfishDataModule(args.train_path, args.val_path,
                                  args.train_batch_size, args.val_batch_size, args.iterable)
