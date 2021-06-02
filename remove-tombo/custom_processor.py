import argparse
from pyguppyclient import get_fast5_files
from pyguppyclient.decode import ReadData, CalledReadData
import mappy
from Bio import SeqIO
import re
from tqdm import tqdm
import numpy as np

from typing import Tuple, Set, List, Optional

from basecall import basecall, sequence_to_raw
from util import Interval, ResegmentationData


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


def align(aligner: mappy.Aligner, query: str, mapq: int) -> Optional[mappy.Alignment]:
    for hit in aligner.map(query):  # Traverse alignments
        if hit.is_primary:  # Check if the alignment is primary
            if hit.mapq < mapq:  # Check if the mapping quality is below set threshold
                return None
            return hit
    return None


def get_relevant_motif_positions(motif_positions: Tuple[Set[int], Set[int]], alignment: mappy.Alignment) -> Set[int]:
    strand_pos = motif_positions[0] if alignment.strand == 1 else motif_positions[1]

    relevant_positions = strand_pos & set(range(alignment.r_st, alignment.r_en))

    if alignment.strand == 1:
        return {pos - alignment.r_st for pos in relevant_positions}
    else:
        return {alignment.r_en - 1 - pos for pos in relevant_positions}


def resolve_insertions(alignment: mappy.Alignment, seq_to_raw: List[Interval]) -> Tuple[List[Interval], List[Interval]]:
    cigar = alignment.cigar if alignment.strand == 1 else reversed(alignment.cigar)

    r_pos, q_pos = 0, alignment.q_st
    r_len = alignment.r_en - alignment.r_st

    signal_intervals = [None] * r_len
    insertion = False
    deletion_idx = []

    for length, operation in cigar:
        if operation in {0, 7, 8}:  # Match or mismatch
            if insertion:
                signal_intervals[r_pos] = Interval(center, seq_to_raw[q_pos].end)  # Base to the right
                insertion = False
                length -= 1
                r_pos += 1
                q_pos += 1

            for i in range(length):
                signal_intervals[r_pos + i] = seq_to_raw[q_pos + i]

            r_pos += length
            q_pos += length

        elif operation == 1:  # Insertion
            insertion_interval = Interval(seq_to_raw[q_pos].start, seq_to_raw[q_pos + length].start)
            center = int(np.mean(insertion_interval))

            signal_intervals[r_pos - 1] = Interval(signal_intervals[r_pos - 1].start, center)  # Base to the left
            insertion = True

            q_pos += length

        elif operation in {2, 3}:  # Deletion or skip
            deletion_idx.append(Interval(r_pos, r_pos + length))

            if insertion:
                signal_intervals[r_pos] = Interval(center, seq_to_raw[q_pos].start)  # Base to the right
                insertion = False
                length -= 1
                r_pos += 1

            for i in range(length):
                signal_intervals[r_pos + i] = Interval(seq_to_raw[q_pos].start, seq_to_raw[q_pos].start)

            r_pos += length

        else:
            raise ValueError('Invalid CIGAR operation')

    return signal_intervals, deletion_idx


def resolve_deletions(signal_intervals: List[Interval], deletion_idx: List[Interval]) -> List[Interval]:
    for idx_st, idx_en in deletion_idx:
        sig_st, sig_en = signal_intervals[idx_st - 1].start, signal_intervals[idx_en].end
        intervals = np.array_split(range(sig_st, sig_en), idx_en - idx_st + 2)

        if len(intervals[-1]) == 0:  # If there is not enough signal points to divide among bases
            while len(intervals[-1]) == 0:
                intervals.pop(-1)
            interval = intervals.pop(-1)
            signal_intervals[idx_en] = Interval(interval[0], interval[-1] + 1)  # Base to the right must have > 0 signal points

        for i in range(idx_st - 1, idx_en + 1):
            if len(intervals) == 0:  # Bases in the middle (deletions) which have 0 signal points
                if i < idx_en:
                    signal_intervals[i] = Interval(signal_intervals[idx_en].start, signal_intervals[idx_en].start)
                continue

            interval = intervals.pop(0)
            signal_intervals[i] = Interval(interval[0], interval[-1] + 1)

    return signal_intervals


def custom_processor(basecall_data: Tuple[ReadData, CalledReadData], aligner: mappy.Aligner, reference: str,
                     motif_positions: Set[int], mapq: int, window: int) -> ResegmentationData:
    read, called = basecall_data

    alignment = align(aligner, called.seq, mapq)
    if not alignment:
        return None

    relevant_motif_positions = get_relevant_motif_positions(motif_positions, alignment)
    if not relevant_motif_positions:
        return None

    seq_to_raw = sequence_to_raw(read, called)

    signal_intervals, deletion_idx = resolve_insertions(alignment, seq_to_raw)
    signal_intervals = resolve_deletions(signal_intervals, deletion_idx)

    resegmentation_data = []

    for motif_position in relevant_motif_positions:
        r_len = alignment.r_en - alignment.r_st
        if motif_position - window < 0 or motif_position + window >= r_len:
            continue

        position = alignment.r_st + motif_position if alignment.strand == 1 else alignment.r_en - 1 - motif_position

        event_intervals = signal_intervals[motif_position - window: motif_position + window + 1]
        event_lens = np.array([interval.end - interval.start for interval in event_intervals])

        region = reference[position - window: position + window + 1]
        bases = region if alignment.strand == 1 else mappy.revcomp(region)

        assert len(event_intervals) == len(event_lens) == len(bases)

        resegmentation_data.append(ResegmentationData(position, event_intervals, event_lens, bases))

    return resegmentation_data


def process_data(input_path: str, recursive: bool, reference_file: str,
                 mapq: int = 0, motif: str = 'CG', index: int = 0, window: int = 8):
    fast5_files = get_fast5_files(input_path, recursive=recursive)

    aligner = make_aligner(reference_file)
    reference = get_reference(reference_file)
    motif_positions = get_motif_positions(reference, motif, index)

    for basecall_data in tqdm(basecall(fast5_files)):
        resegmentation_data = custom_processor(basecall_data, aligner, reference, motif_positions, mapq, window)

        if resegmentation_data:
            print(resegmentation_data)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Custom processor')

    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='Path to the input file or folder containing FAST5 files')

    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Flag to indicate if folder will be searched recursively (default: False)')

    parser.add_argument('--reference', type=str, required=True,
                        help='Path to the reference file')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    process_data(args.input_path, args.recursive, args.reference)
