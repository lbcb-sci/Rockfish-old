from pathlib import Path
import sys
import random
import shutil

from tqdm import tqdm
# import tqdm.contrib

from typing import *

import argparse

SAMPLE_SIZE_ERROR = ('Cannot sample w/o replacement. '
                     'Number of files is {} and total size of '
                     'all sets is {}.')


def get_files(path: Path, recursive: bool = False) -> Set[Path]:
    if path.is_file():
        return {path}

    # Finding all input FAST5 files
    if recursive:
        files = path.glob('**/*.fast5')
    else:
        files = path.glob('*.fast5')

    return set(files)


def include_files(files, include_paths, recursive):
    included = set()

    for p in include_paths:
        p_files = get_files(p, recursive)
        inter = files & p_files

        included.update(inter)

    return included


def exclude_files(files, exclude_paths, recursive):
    result = set(files)

    for p in exclude_paths:
        p_files = get_files(p, recursive)
        result = result - p_files

    return result


def sample_fast5(args):
    files = get_files(args.input_path, args.recursive)

    if args.include:
        files = include_files(files, args.include, args.recursive)
    if args.exclude:
        files = exclude_files(files, args.exclude, args.recursive)

    tqdm.write(f'{len(files)} files for sampling.')

    # Create output dir
    try:
        args.out_folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f'Output folder {args.out_folder} exists.', file=sys.stderr)
        sys.exit(1)

    total_sample_size = sum(args.sizes)
    if total_sample_size > len(files):
        print(SAMPLE_SIZE_ERROR.format(len(files), total_sample_size), file=sys.stderr)
        sys.exit(1)

    for set_name, set_size in zip(tqdm(args.names), args.sizes):
        tqdm.write(f'Generating sample named {set_name}.')

        set_path = args.out_folder / set_name
        set_path.mkdir()

        sampled_files = random.sample(files, set_size)
        for sampled_file in tqdm(sampled_files, leave=False):
            dest_path = set_path / sampled_file.name
            shutil.copyfile(str(sampled_file), str(dest_path))

            files.remove(sampled_file)


def create_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', required=True, type=Path)
    parser.add_argument('-r', '--recursive', action='store_true')

    parser.add_argument('--include', type=Path, nargs='*')
    parser.add_argument('--exclude', type=Path, nargs='*')

    parser.add_argument('--names', type=str, nargs='+')
    parser.add_argument('--sizes', type=int, nargs='+')

    parser.add_argument('-o', '--out_folder', type=Path, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = create_arguments()

    sample_fast5(args)
