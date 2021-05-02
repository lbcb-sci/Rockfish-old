import argparse
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.distributed as dist

from pytorch_lightning import LightningDataModule

import numpy as np
import io
import os

from tqdm import tqdm


DATA_DTYPE = np.dtype([('signal', np.float16, (340,)), ('kmer', np.uint8, (17,)), ('label', np.uint8)])


def dataset_size(path):
    length, r = divmod(os.path.getsize(path), DATA_DTYPE.itemsize)
    assert r == 0, 'Cannot calculate a number of examples. Possibly wrong data file.'

    return length


def different_length_error(filename, read_name):
    msg = f'Different number of examples for read {read_name} in file {filename}.'
    raise RuntimeError(msg)


class MemoryDataset(Dataset):
    def __init__(self, path):
        with io.open(path, 'rb') as f:
            self.data = np.fromfile(f, dtype=DATA_DTYPE)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        signal = torch.from_numpy(example['signal'])
        bases = torch.from_numpy(np.repeat(example['kmer'], 20).astype(int))
        label = example['label']

        return signal, bases, label


def iter_worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    dataset.f = io.open(dataset.path, 'rb')


class IterDataset(Dataset):
    def __init__(self, path):
        self.len = dataset_size(path)

        self.path = path

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        self.f.seek(idx * DATA_DTYPE.itemsize)
        example = np.frombuffer(self.f.read(DATA_DTYPE.itemsize), dtype=DATA_DTYPE)[0]

        signal = torch.from_numpy(example['signal'])
        bases = torch.from_numpy(np.repeat(example['kmer'], 20).astype(int))
        label = example['label']

        return signal, bases, label


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
        if self.iterable:
            train_ds = IterDataset(self.train_path)
            worker_init_fn = iter_worker_init
        else:
            train_ds = MemoryDataset(self.train_path)

        return DataLoader(train_ds, self.train_batch_size, True,
                        num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

    def val_dataloader(self):
        worker_init_fn = None
        if self.iterable:
            val_ds = IterDataset(self.val_path)
            worker_init_fn = iter_worker_init
        else:
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
