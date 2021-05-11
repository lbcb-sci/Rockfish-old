import io
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


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
