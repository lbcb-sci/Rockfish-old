import mappy
from Bio import SeqIO
import re

from typing import Tuple, Set


def make_aligner(reference_file: str) -> mappy.Aligner:
    aligner = mappy.Aligner(reference_file, preset='map-ont')  # Load or build index
    if not aligner:
        raise Exception("ERROR: failed to load/build index")

    return aligner


def get_reference(reference_file: str) -> str:
    for seq_record in SeqIO.parse(reference_file, 'fasta'):
        reference = str(seq_record.seq)

    return reference


def get_motif_positions(reference: str, motif: str, index: int) -> Tuple[Set[int], Set[int]]:
    r_len = len(reference)

    # Forward strand
    fwd_matches = re.finditer(motif, reference, re.I)
    fwd_pos = set(m.start() + index for m in fwd_matches)

    # Reverse strand
    rev_matches = re.finditer(motif, mappy.revcomp(reference), re.I)
    rev_pos = set(r_len - (m.start() + index) - 1 for m in rev_matches)

    return fwd_pos, rev_pos