import mappy
from Bio import SeqIO
import re

from typing import Tuple, Set, Dict


def make_aligner(reference_file: str) -> mappy.Aligner:
    aligner = mappy.Aligner(reference_file, preset='map-ont')  # Load or build index
    if not aligner:
        raise Exception("ERROR: failed to load/build index")

    return aligner


def get_motif_positions(reference_file: str, motif: str, index: int) -> Dict[str, Tuple[Set[int], Set[int]]]:
    chromosomes = SeqIO.to_dict(SeqIO.parse(reference_file, 'fasta'))
    motif_positions = dict()

    for chromosome, record in chromosomes.items():
        reference = str(record.seq)

        # Forward strand
        fwd_matches = re.finditer(motif, reference, re.I)
        fwd_pos = set(m.start() + index for m in fwd_matches)

        # Reverse strand
        rev_matches = re.finditer(motif, mappy.revcomp(reference), re.I)
        rev_pos = set(len(reference) - (m.start() + index) - 1 for m in rev_matches)

        motif_positions[chromosome] = fwd_pos, rev_pos

    return motif_positions


def get_reference(reference_file: str, contig: str) -> str:
    chromosomes = SeqIO.to_dict(SeqIO.parse(reference_file, 'fasta'))

    return str(chromosomes[contig].seq)
