from pathlib import Path
import mappy as mp
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import numpy as np

from util import get_files, read_fastq


def plot_distribution(data, modification, bins=15):
    """ Distribution of mapping quality """
    plt.hist(data, bins=bins, color='green' if modification == 'nomod' else 'red')
    plt.title(f'Distribution of mapping quality ({modification})')
    plt.xlabel('mapping quality')
    plt.ylabel('count')
    plt.yticks(np.arange(0, 1001, 100))
    plt.savefig(f'../plots/mapq_{modification}.png')


def get_mapqs(dir_in, file_fasta):
    a = mp.Aligner(file_fasta, preset='map-ont')  # Load or build index
    if not a:
        raise Exception("ERROR: failed to load/build index")

    reads = get_files(dir_in)
    mapqs = []

    for read in tqdm(reads):
        with h5py.File(read, 'r', libver='latest') as fd:
            fastq = read_fastq(fd)
            mapq = 0

            for hit in a.map(fastq.seq):  # Traverse alignments
                if hit.is_primary:  # Check if the alignment is primary
                    mapq = hit.mapq
                    break

            mapqs.append(mapq)

    return mapqs


if __name__ == '__main__':
    modification = 'nomod'  # 'mod' or 'nomod'
    dir_in = Path(f'/home/sdeur/tmp-guppy/{modification}/workspace')
    file_fasta = '/home/sdeur/data/ecoli_k12_mg1655.fasta'

    mapqs = get_mapqs(dir_in, file_fasta)
    plot_distribution(mapqs, modification)
