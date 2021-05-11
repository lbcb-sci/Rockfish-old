from pathlib import Path
import mappy as mp
from tqdm import tqdm
import h5py
import io
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import re
import pandas as pd


def get_files(path, recursive=False):
    if path.is_file():
        return {path}

    # Finding all input FAST5 files
    if recursive:
        files = path.glob('**/*.fast5')
    else:
        files = path.glob('*.fast5')

    return set(files)


def read_fastq(read) -> SeqRecord:
    fastq_str = read[f'/Analyses/Basecall_1D_000/BaseCalled_template/Fastq'][()]
    fastq_str = fastq_str.decode()
    str_file = io.StringIO(fastq_str)

    record = SeqIO.read(str_file, 'fastq')
    return record


def align_mappy(dir_in, file_out, file_fasta):
    a = mp.Aligner(file_fasta)  # Load or build index
    if not a:
        raise Exception("ERROR: failed to load/build index")

    reads = get_files(dir_in)
    files_fastq = {}
    data = []

    for read in tqdm(reads):
        with h5py.File(read, 'r', libver='latest') as fd:
            no_alignment = True
            fastq = read_fastq(fd)
            files_fastq[fastq.id] = len(fastq.seq)

            for hit in a.map(fastq.seq):  # Traverse alignments
                if hit.is_primary:  # Check if the alignment is primary
                    # Reference
                    for seq_record in SeqIO.parse(file_fasta, 'fasta'):
                        ref = seq_record.seq[hit.r_st: hit.r_en]
                    r_CG_num = len(re.findall(r'(CG)', str(ref)))

                    # Query
                    query = fastq.seq[hit.q_st: hit.q_en]
                    if hit.strand == -1:
                        query = mp.revcomp(query)
                    q_CG_num = len(re.findall(r'(CG)', str(query)))

                    no_alignment = False
                    data.append([fastq.id, hit.r_st, hit.r_en, hit.q_st, hit.q_en, r_CG_num, q_CG_num, hit.cigar_str])
                    break

        if no_alignment:
            data.append([fastq.id, '', '', '', '', 0, 0, ''])

    data = pd.DataFrame(data, columns=['read_id', 'r_st', 'r_en', 'q_st', 'q_en', 'r_CG_num', 'q_CG_num', 'cigar_str'])
    data.sort_values('read_id', inplace=True)
    data.to_csv(file_out, index=False)

    print("Average length of fastq files:", sum(files_fastq.values()) / len(files_fastq.values()))


if __name__ == '__main__':
    modification = 'nomod'
    dir_in = Path(f'/home/sdeur/tmp-guppy/{modification}/workspace')
    file_out = f'/home/sdeur/tmp-mappy/mappy_{modification}.csv'
    file_fasta = '/home/sdeur/data/ecoli_k12_mg1655.fasta'

    align_mappy(dir_in, file_out, file_fasta)
