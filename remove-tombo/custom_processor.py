from pathlib import Path
from tqdm import tqdm
from pyguppyclient import GuppyBasecallerClient, get_fast5_files, yield_reads
import mappy as mp
from Bio import SeqIO
import re
import numpy as np

from util import ResegmentationData


def make_aligner(reference_file):
    aligner = mp.Aligner(reference_file, preset='map-ont')  # Load or build index
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
    rev_matches = re.finditer(motif, mp.revcomp(reference), re.I)
    rev_pos = set(r_len - (m.start() + index) - 1 for m in rev_matches)

    return fwd_pos, rev_pos


# config='dna_r9.4.1_450bps_hac' for more precise basecalling
def basecall(read_file, config='dna_r9.4.1_450bps_fast'):
    with GuppyBasecallerClient(config_name=config, trace=True) as client:
        for read in yield_reads(read_file):
            called = client.basecall(read)[1]
            return read, called


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
        return {p - alignment.r_st for p in relevant_positions}
    else:
        return {alignment.r_en - 1 - p for p in relevant_positions}


def reference_to_basecall(alignment):
    cigar = alignment.cigar if alignment.strand == 1 else reversed(alignment.cigar)

    ref_to_bc = []
    qpos = alignment.q_st

    for length, operation in cigar:
        if operation in {0, 7, 8}:  # Match or mismatch
            for i in range(length):
                ref_to_bc.append((qpos + i, qpos + i + 1))
            qpos += length

        elif operation == 1:  # Insertion
            qpos += length

        elif operation in {2, 3}:  # Deletion or skip
            for _ in range(length):
                ref_to_bc.append((qpos, qpos))

    return ref_to_bc


def sequence_to_raw(read, called):
    first_signal_id = len(read.signal) - called.trimmed_samples
    move_index = np.nonzero(called.move)[0]

    seq_to_raw_start = first_signal_id + move_index * called.model_stride
    seq_to_raw_len = np.diff(seq_to_raw_start, append=len(read.signal))
    seq_to_raw_end = seq_to_raw_start + seq_to_raw_len

    return list(zip(seq_to_raw_start, seq_to_raw_end))


def resolve_insertions(position, ref_to_bc, seq_to_raw, window):
    base_intervals = ref_to_bc[position - window: position + window + 1]  # End is exclusive
    signal_intervals = seq_to_raw[base_intervals[0][0]: base_intervals[-1][1]]

    deletion_idx = []

    for i in range(len(base_intervals)):
        if base_intervals[i][0] == base_intervals[i][1]:  # Deletion or skip
            signal_intervals.insert(i, (seq_to_raw[base_intervals[i][0]][0], seq_to_raw[base_intervals[i][0]][0]))

            if deletion_idx and i == deletion_idx[-1][1]:
                deletion_idx[-1] = deletion_idx[-1][0], i + 1
            else:
                deletion_idx.append((i, i + 1))

        if i != len(base_intervals) - 1 and base_intervals[i][1] != base_intervals[i + 1][0]:  # Resolve insertions
            no_of_insertions = base_intervals[i + 1][0] - base_intervals[i][1]
            insertion_interval = signal_intervals[i][1], signal_intervals[i + 1 + no_of_insertions][0]

            center = int(np.mean(insertion_interval))
            del signal_intervals[i + 1: i + 1 + no_of_insertions]

            signal_intervals[i] = signal_intervals[i][0], center  # Base to the left
            signal_intervals[i + 1] = center, signal_intervals[i + 1][1]  # Base to the right

    return base_intervals, signal_intervals, deletion_idx


def resolve_deletions(signal_intervals, deletion_idx):
    for idx_st, idx_en in deletion_idx:  # Resolve deletions
        if idx_st == 0 and idx_en == len(signal_intervals):  # Region consisting only of deletions
            break

        if idx_st == 0:
            start, end = idx_st, idx_en
        elif idx_en == len(signal_intervals):
            start, end = idx_st - 1, idx_en - 1
        else:
            start, end = idx_st - 1, idx_en

        sig_st, sig_en = signal_intervals[start][0], signal_intervals[end][1]
        intervals = np.array_split(range(sig_st, sig_en), end - start + 1)

        for i in range(start, end + 1):
            interval = intervals.pop(0)
            if len(interval) == 0:  # If there is not enough signal points for the remaining bases
                break
            signal_intervals[i] = interval[0], interval[-1] + 1

    return signal_intervals


def resolve_region(position, ref_to_bc, seq_to_raw, window):
    if position - 8 < 0 or position + 8 >= len(ref_to_bc):
        return None

    base_intervals, signal_intervals, deletion_idx = resolve_insertions(position, ref_to_bc, seq_to_raw, window)
    signal_intervals = resolve_deletions(signal_intervals, deletion_idx)

    assert len(base_intervals) == len(signal_intervals)

    return signal_intervals


def custom_processor(fast5_file, aligner, reference, motif_positions, mapq, window):
    read, called = basecall(fast5_file)

    alignment = align(aligner, called.seq, mapq)
    if not alignment:
        return None

    relevant_motif_positions = get_relevant_motif_positions(motif_positions, alignment)

    ref_to_bc = reference_to_basecall(alignment)
    seq_to_raw = sequence_to_raw(read, called)

    for motif_position in relevant_motif_positions:
        event_intervals = resolve_region(motif_position, ref_to_bc, seq_to_raw, window)
        if event_intervals is None:
            continue

        position = alignment.r_st + motif_position
        event_lens = [end - start for start, end in event_intervals]
        bases = reference[position - window: position + window + 1]

        return ResegmentationData(position, event_intervals, event_lens, bases)


def process_data(dir_in, reference_file, mapq=0, motif='CG', index=0, window=8):
    fast5_files = get_fast5_files(dir_in, recursive=True)

    aligner = make_aligner(reference_file)
    reference = get_reference(reference_file)
    motif_positions = get_motif_positions(reference, motif, index)

    for fast5_file in tqdm(fast5_files):
        resegmentation_data = custom_processor(fast5_file, aligner, reference, motif_positions, mapq, window)
        if resegmentation_data is None:
            continue

        print(resegmentation_data)


if __name__ == '__main__':
    modification = 'mod'  # 'mod' or 'nomod'
    dir_in = Path(f'/home/sdeur/data/{modification}')
    reference_file = '/home/sdeur/data/ecoli_k12_mg1655.fasta'

    process_data(dir_in, reference_file)
