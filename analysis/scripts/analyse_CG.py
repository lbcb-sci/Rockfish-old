from pathlib import Path
import mappy as mp
from tqdm import tqdm
import h5py
from Bio import SeqIO
import pandas as pd

from util import get_files, read_fastq


def count_matches(cs):
    matches = {'M': 0, 'X': 0, 'D': 0, 'I': 0}
    mode, value = 'M', 0

    for c in cs:
        if c == ':':
            matches[mode] += int(value) if mode == 'M' or mode == 'X' else len(value)
            mode, value = 'M', ''
            continue
        elif c == '*':
            matches[mode] += int(value) if mode == 'M' or mode == 'X' else len(value)
            mode, value = 'X', ''
            continue
        elif c == '-':
            matches[mode] += int(value) if mode == 'M' or mode == 'X' else len(value)
            mode, value = 'D', ''
            continue
        elif c == '+':
            matches[mode] += int(value) if mode == 'M' or mode == 'X' else len(value)
            mode, value = 'I', ''
            continue

        if mode == 'X':
            value = 1
        else:
            value += c

    matches[mode] += int(value) if mode == 'M' or mode == 'X' else len(value)
    return matches


def normalize(ref, query, cigar):
    r = list(ref)
    q = list(query)
    i = 0
    num = ''

    for c in cigar:
        if c.isnumeric():
            num += c
            continue

        elif c == 'M':
            i += int(num)

        elif c == 'D':
            for _ in range(int(num)):
                q.insert(i, '-')
                i += 1

        elif c == 'I':
            for _ in range(int(num)):
                r.insert(i, '-')
                i += 1

        num = ''

    return ''.join(r), ''.join(q)


def count_CG(ref, query):
    CG_cnt = {'M': 0, 'X': 0, 'D': 0, 'I': 0}

    for i in range(len(ref) - 1):
        if ref[i] == 'C' and ref[i + 1] == 'G':
            if query[i] == 'C' and query[i + 1] == 'G':
                CG_cnt['M'] += 1

            elif query[i] == '-' and query[i + 1] == 'G':
                CG_cnt['D'] += 1

            elif query[i] in {'A', 'G', 'T'} and query[i + 1] == 'G':
                CG_cnt['X'] += 1

        elif ref[i] == '-' and ref[i + 1] == 'G':
            if query[i] == 'C' and query[i + 1] == 'G':
                CG_cnt['I'] += 1

    return CG_cnt


def analyse_CG(dir_in, file_out, file_fasta):
    a = mp.Aligner(file_fasta)  # Load or build index
    if not a:
        raise Exception("ERROR: failed to load/build index")

    reads = get_files(dir_in)
    data = []

    for read in tqdm(reads):
        with h5py.File(read, 'r', libver='latest') as fd:
            matches = {'M': 0, 'X': 0, 'D': 0, 'I': 0}
            CG_cnt = {'M': 0, 'X': 0, 'D': 0, 'I': 0}

            fastq = read_fastq(fd)
            ref = ''
            mapq = 0

            for hit in a.map(fastq.seq, cs=True):  # Traverse alignments
                if hit.is_primary:  # Check if the alignment is primary
                    # Alignment
                    matches = count_matches(hit.cs)

                    # Reference
                    for seq_record in SeqIO.parse(file_fasta, 'fasta'):
                        ref = seq_record.seq[hit.r_st: hit.r_en]

                    # Query
                    query = fastq.seq[hit.q_st: hit.q_en]
                    if hit.strand == -1:
                        query = mp.revcomp(query)

                    # Normalize
                    ref, query = normalize(ref, query, hit.cigar_str)

                    # Analyse CG motif
                    CG_cnt = count_CG(ref, query)

                    mapq = hit.mapq
                    break

            data.append([fastq.id, len(ref), matches['M'], matches['X'], matches['D'], matches['I'],
                         CG_cnt['M'], CG_cnt['X'], CG_cnt['D'], CG_cnt['I'], mapq])

    data = pd.DataFrame(data, columns=['read_id', 'alignment_len', 'M', 'X', 'D', 'I',
                                       'M_CG', 'X_CG', 'D_CG', 'I_CG', 'mapq'])
    data.sort_values('read_id', inplace=True)
    data.to_csv(file_out, index=False)


if __name__ == '__main__':
    modification = 'nomod'  # 'mod' or 'nomod'
    dir_in = Path(f'/home/sdeur/tmp-guppy/{modification}/workspace')
    file_out = f'/home/sdeur/tmp-mappy/CG_{modification}.csv'
    file_fasta = '/home/sdeur/data/ecoli_k12_mg1655.fasta'

    analyse_CG(dir_in, file_out, file_fasta)
