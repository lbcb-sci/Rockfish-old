import numpy as np
import os
import random
import io
import sys

from contextlib import ExitStack

from tqdm import tqdm

import argparse

DATA_DTYPE = np.dtype([('signal', np.float16, (340,)), ('kmer', np.uint8, (17,)), ('label', np.uint8)])


def data_split(string):
    data = string.strip().split(',')
    return data[0], int(data[1])


def check_size(splits, n_examples):
    total_size = sum(s for _, s in splits)
    if total_size != n_examples:
        print('Total splits size does not match the number of examples.', file=sys.err)
        sys.exit(-1)


def get_splits_indices(indices, splits, out_path):
    dirname, basename = os.path.split(out_path)

    idx_split = []
    start = 0
    for name, size in splits:
        path = f'{dirname}/{name}_{basename}'

        end = start + size
        idx_split.append((path, indices[start:end]))
        start = end

    return idx_split


def get_size(args):
    input_path = args.input_path

    data_size = os.path.getsize(input_path)
    n_examples, remainder = divmod(data_size, DATA_DTYPE.itemsize)
    assert remainder == 0

    print(f'Total {n_examples} examples in {input_path}.')


def shuffle(args):
    input_path, output_path = args.input_path, args.output_path
    splits = args.splits
    from_disk, seed = args.from_disk, args.seed

    random.seed(seed)

    data_size = os.path.getsize(input_path)
    n_examples, remainder = divmod(data_size, DATA_DTYPE.itemsize)
    assert remainder == 0

    if splits:
        check_size(splits, n_examples)

    indices = list(range(n_examples))
    random.shuffle(indices)

    if splits:
        idx_split = get_splits_indices(indices, splits, output_path)
    else:
        idx_split = [(output_path, indices)]

    with ExitStack() as stack:
        if from_disk:
            f_in = stack.enter_context(io.open(input_path, 'rb'))
            tqdm.write('Reading from disk.')
        else:
            tqdm.write('Loading input data into memory.')
            f_in = stack.enter_context(io.BytesIO(open(input_path, 'rb').read()))
            tqdm.write('Finished loading data into memory.')

        pbar = stack.enter_context(tqdm(n_examples))
        for path, all_idx in idx_split:
            f_out = stack.enter_context(io.open(path, 'wb'))
            for idx in all_idx:
                f_in.seek(idx * DATA_DTYPE.itemsize)
                example = f_in.read(DATA_DTYPE.itemsize)

                f_out.write(example)
                pbar.update(1)

            f_out.flush()
            f_out.close()


def create_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Check size command
    check_size_parser = subparsers.add_parser('size', help='Get dataset size.')
    check_size_parser.set_defaults(command=get_size)

    check_size_parser.add_argument('-i', '--input_path', type=str, required=True)
    
    #Shuffle command
    shuffle_parser = subparsers.add_parser('shuffle', help='Shuffle dataset.')
    check_size_parser.set_defaults(command=shuffle)

    shuffle_parser.add_argument('-i', '--input_path', type=str, required=True)
    shuffle_parser.add_argument('-o', '--output_path', type=str, required=True)

    shuffle_parser.add_argument('--get_data_size', action='store_true',
                help='Get dataset size instead of splitting the data.')

    shuffle_parser.add_argument('--splits', type=data_split, nargs='*', 
                help='''Provide one or more data splits in format (name,size).
                        If provided, the sum of the sizes should be equal to the
                        number of examples.''')

    shuffle_parser.add_argument('--from_disk', action='store_true')
    shuffle_parser.add_argument('--seed', type=int, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = create_args()
    
    args.command(args)