from pathlib import Path
from pyguppyclient import get_fast5_files
import mappy
from Bio import SeqIO
import re
from tqdm import tqdm
import numpy as np

from basecall import basecall, sequence_to_raw
from util import ResegmentationData


def make_aligner(reference_file):
    aligner = mappy.Aligner(reference_file, preset='map-ont')  # Load or build index
    if not aligner:
        raise Exception("ERROR: failed to load/build index")

    return aligner


def get_reference(reference_file):
    for seq_record in SeqIO.parse(reference_file, 'fasta'):
        reference = str(seq_record.seq)

    return reference


def get_motif_positions(reference, motif, index):
    r_len = len(reference)

    # Forward strand
    fwd_matches = re.finditer(motif, reference, re.I)
    fwd_pos = set(m.start() + index for m in fwd_matches)

    # Reverse strand
    rev_matches = re.finditer(motif, mappy.revcomp(reference), re.I)
    rev_pos = set(r_len - (m.start() + index) - 1 for m in rev_matches)

    return fwd_pos, rev_pos


def align(aligner, query, mapq):
    for hit in aligner.map(query):  # Traverse alignments
        if hit.is_primary:  # Check if the alignment is primary
            if hit.mapq < mapq:  # Check if the mapping quality is below set threshold
                return None
            return hit
    return None


def get_relevant_motif_positions(motif_positions, alignment):
    strand_pos = motif_positions[0] if alignment.strand == 1 else motif_positions[1]

    relevant_positions = strand_pos & set(range(alignment.r_st, alignment.r_en))

    if alignment.strand == 1:
        return {pos - alignment.r_st for pos in relevant_positions}
    else:
        return {alignment.r_en - 1 - pos for pos in relevant_positions}


def resolve_insertions(alignment, seq_to_raw):
    cigar = alignment.cigar if alignment.strand == 1 else reversed(alignment.cigar)

    r_pos, q_pos = 0, alignment.q_st
    r_len = alignment.r_en - alignment.r_st

    signal_intervals = [None] * r_len
    insertion = False
    deletion_idx = []

    for length, operation in cigar:
        if operation in {0, 7, 8}:  # Match or mismatch
            if insertion:
                signal_intervals[r_pos] = center, seq_to_raw[q_pos][1]  # Base to the right
                insertion = False
                length -= 1
                r_pos += 1
                q_pos += 1

            for i in range(length):
                signal_intervals[r_pos + i] = seq_to_raw[q_pos + i]

            r_pos += length
            q_pos += length

        elif operation == 1:  # Insertion
            insertion_interval = seq_to_raw[q_pos][0], seq_to_raw[q_pos + length][0]

            center = int(np.mean(insertion_interval))
            signal_intervals[r_pos - 1] = signal_intervals[r_pos - 1][0], center  # Base to the left
            insertion = True

            q_pos += length

        elif operation in {2, 3}:  # Deletion or skip
            deletion_idx.append((r_pos, r_pos + length))

            if insertion:
                signal_intervals[r_pos] = center, seq_to_raw[q_pos][0]  # Base to the right
                insertion = False
                length -= 1
                r_pos += 1

            for i in range(length):
                signal_intervals[r_pos + i] = seq_to_raw[q_pos][0], seq_to_raw[q_pos][0]

            r_pos += length

        else:
            raise ValueError('Invalid CIGAR operation')

    return signal_intervals, deletion_idx


def resolve_deletions(signal_intervals, deletion_idx):
    for idx_st, idx_en in deletion_idx:
        sig_st, sig_en = signal_intervals[idx_st - 1][0], signal_intervals[idx_en][1]
        intervals = np.array_split(range(sig_st, sig_en), idx_en - idx_st + 2)

        if len(intervals[-1]) == 0:  # If there is not enough signal points to divide among bases
            while len(intervals[-1]) == 0:
                intervals.pop(-1)
            interval = intervals.pop(-1)
            signal_intervals[idx_en] = interval[0], interval[-1] + 1  # Base to the right must have > 0 signal points

        for i in range(idx_st - 1, idx_en + 1):
            if len(intervals) == 0:  # Bases in the middle (deletions) which have 0 signal points
                if i < idx_en:
                    signal_intervals[i] = signal_intervals[idx_en][0], signal_intervals[idx_en][0]
                continue

            interval = intervals.pop(0)
            signal_intervals[i] = interval[0], interval[-1] + 1

    return signal_intervals


def custom_processor(basecall_data, aligner, reference, motif_positions, mapq, window):
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
        event_lens = [end - start for start, end in event_intervals]
        bases = reference[position - window: position + window + 1]

        assert len(event_intervals) == len(event_lens) == len(bases)

        resegmentation_data.append(ResegmentationData(position, event_intervals, event_lens, bases))

    return resegmentation_data


def process_data(dir_in, reference_file, mapq=0, motif='CG', index=0, window=8):
    fast5_files = get_fast5_files(dir_in, recursive=True)

    aligner = make_aligner(reference_file)
    reference = get_reference(reference_file)
    motif_positions = get_motif_positions(reference, motif, index)

    for basecall_data in tqdm(basecall(fast5_files)):
        resegmentation_data = custom_processor(basecall_data, aligner, reference, motif_positions, mapq, window)

        if resegmentation_data:
            print(resegmentation_data)


if __name__ == '__main__':
    modification = 'mod'  # 'mod' or 'nomod'
    dir_in = Path(f'/home/sdeur/data/{modification}')
    reference_file = '/home/sdeur/data/ecoli_k12_mg1655.fasta'

    process_data(dir_in, reference_file)
