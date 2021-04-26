from pathlib import Path
import h5py

from tqdm import tqdm

import argparse


def get_files(path, recursive=False):
    if path.is_file():
        return {path}

    # Finding all input FAST5 files
    if recursive:
        files = path.glob('**/*.fast5')
    else:
        files = path.glob('*.fast5')

    return set(files)


def remove_analysis(path, recursive=False, group='/Analyses'):
    files = get_files(path, recursive)

    for name in tqdm(files):
        with h5py.File(str(name), 'a') as f:
            try:
                del f[group]
            except:
                continue


def create_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=Path, required=True)
    parser.add_argument('-r', '--recursive', action='store_true')

    parser.add_argument('-g', '--group', type=str, default='/Analyses')

    return parser.parse_args()


if __name__ == '__main__':
    args = create_arguments()

    remove_analysis(args.input_path, args.recursive, args.group)
