from pathlib import Path

from typing import *

from .models import BEDPredicate, BEDData, Strand, GenomicPos, ModInfo


def extract_bed_positions(
        path: Path,
        bed_filter: Optional[BEDPredicate]=None) -> BEDData:
    """ The function extracts modification information from the given file.

    The function extracts modification information from the given file. Optionally, data can be filtered using given
    filter.

    :param path: Path to the BED file containing modification information
    :param bed_filter: Optional filter for BED entries
    :return: Modification information for the positions in the given file
    """
    bed_info = {}

    with path.open('r') as f:
        for line in f:
            data = line.strip().split('\t')

            chromosome = data[0]
            pos = int(data[1])
            strand = Strand.FORWARD if data[5] == '+' else Strand.REVERSE
            genomic_pos = GenomicPos(chromosome, pos, strand)

            n_reads = int(data[4])
            mod_freq = int(data[-1])
            mod_info = ModInfo(n_reads, mod_freq)

            if bed_filter is None or bed_filter(genomic_pos, mod_info):
                bed_info[genomic_pos] = mod_info

    return bed_info


def high_confidence_filter(
        genomic_pos: GenomicPos,
        mod_info: ModInfo) -> bool:
    """ Function that filters high confidence modifications.

    This function is used for filtering high confidence positions. High confidence position is defined as a position
    that has coverage of at least 10 reads and having all reads modified or unmodified
    (modification frequency of 0% or 100%).

    :param genomic_pos: Genomic positions that is tested. Not used
    :param mod_info: Modification information for the given positions
    :return: True if position is high confidence position, False otherwise
    """
    if mod_info.n_reads < 10:
        return False

    if mod_info.mod_freq == 0 or mod_info.mod_freq == 100:
        return True

    return False


def bed_filter_factory(method: Optional[str]) -> Optional[BEDPredicate]:
    """ Factory method that returns BED filter from the given string.

    :param method: BED filter method name
    :return: BED filter function
    """
    if method is None:
        return None

    if method == 'high_confidence':
        return high_confidence_filter

    raise ValueError(f"{method} BED filter method doesn't exist.")
