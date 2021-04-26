import sys
from pathlib import Path
from enum import Enum
import shutil
from collections import Counter, defaultdict
import h5py

from typing import *

from tqdm import tqdm

import argparse

FAST5_READS_PATH = 'Raw/Reads' 


class AlignStatus(Enum):
    SECONDARY = 256
    SUPPLEMENTARY = 2048
    UNMAPPED = 4
    OTHER = -1


def get_files(path: Path, recursive: bool = False) -> List[Path]:
    if path.is_file():
        return [path]

    # Finding all input FAST5 files
    if recursive:
        files = path.glob('**/*.fast5')
    else:
        files = path.glob('*.fast5')

    return list(files)


def get_chr_names(sam) -> Tuple[Set[str], str]:
    chr_names = {'unmapped'}

    line = sam.readline().strip()
    while line.startswith('@SQ'):
        data = line.split('\t')

        _, name = data[1].split(':')
        chr_names.add(name)

        line = sam.readline().strip()

    return chr_names, line


def mkdir_chr(out: Path, names: Set[str]) -> Dict[str, Path]:
    path_for_chr = {}

    for name in names:
        chr_path = Path(out, name)
        chr_path.mkdir(exist_ok=True)

        path_for_chr[name] = chr_path

    return path_for_chr


def check_flag(flag: int) -> AlignStatus:
    if flag & 256:
        return AlignStatus.SECONDARY

    if flag & 2048:
        return AlignStatus.SUPPLEMENTARY

    if flag & 4:
        return AlignStatus.UNMAPPED

    return AlignStatus.OTHER


def write_mappings(mappings: Dict[str, List[Path]], out_folder: Path) -> None:
    mappings_path = out_folder / 'mappings.txt'
    with mappings_path.open('w') as mapping_f:
        for chrom, paths in mappings.items():
            mapping_f.write(f'{chrom}\n')

            for p in paths:
                mapping_f.write(f'    {str(p)}\n')



def get_read_id(path):
    with h5py.File(path, 'r') as f:
        read_name = list(f[FAST5_READS_PATH].keys())
        read_group = f[f'{FAST5_READS_PATH}/{read_name}']
        return read_group.attrs['read_id']

def organize_fast5(fast5_path: Path, align_path: Path, out_folder: Path, recursive: bool = False) -> None:
    align_data = {}
    counter = Counter()

    with align_path.open('r') as sam:
        names, line = get_chr_names(sam)  # Getting ref seq names

        print(f'Found {len(names)} chromosomes.')

        out_folder.mkdir(parents=True, exist_ok=True)  # Creating output folder if it doesn't exist
        path_for_chr = mkdir_chr(out_folder, names)  # Creating ref seq folders and mapping paths for them

        # Skipping header lines
        while line.startswith('@'):
            line = sam.readline().strip()

        # Process alignments
        while line:
            data = line.split('\t')

            flag = int(data[1])
            align_status = check_flag(flag)

            if align_status == AlignStatus.SECONDARY or align_status == AlignStatus.SUPPLEMENTARY:
                line = sam.readline().strip()
                continue

            qname, rname = data[0], data[2]

            if align_status == AlignStatus.UNMAPPED:
                assert rname == '*'
                rname = 'unmapped'

            align_data[qname] = rname
            counter[rname] += 1

            line = sam.readline().strip()

    print(f'Extracted {len(align_data)} alignments.')
    print(counter)

    files = get_files(fast5_path, recursive)
    files = {p.absolute() for p in files}

    print(f'Found {len(files)} FAST5 files.')

    mappings = defaultdict(lambda: list())

    for p in tqdm(files):
        read_name = p.stem

        try:
            read_id = get_read_id(p)
            rname = align_data[read_id]
        except KeyError:
            try:
                rname = align_data[read_name]
            except KeyError:
                print(f'No alignment entry for {str(p)}', file=sys.stderr)
                continue

        read_name = p.name
        dest_path = path_for_chr[rname] / read_name

        shutil.move(p, dest_path)
        mappings[rname].append(dest_path.relative_to(out_folder))

    write_mappings(mappings, out_folder)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--fast5_path', type=Path, required=True)
    parser.add_argument('-r', '--recursive', required=False, action='store_true')

    parser.add_argument('-a', '--align_path', type=Path, required=True)

    parser.add_argument('-o', '--out_folder', type=Path, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    organize_fast5(args.fast5_path, args.align_path, args.out_folder, args.recursive)
