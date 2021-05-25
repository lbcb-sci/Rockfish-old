from pathlib import Path
from tqdm import tqdm
from pyguppyclient import GuppyBasecallerClient, get_fast5_files, yield_reads
import mappy as mp
from Bio import SeqIO
import re
import numpy as np


# config='dna_r9.4.1_450bps_hac.cfg' for more precise basecalling
def basecall(read_file, config='dna_r9.4.1_450bps_fast'):
    with GuppyBasecallerClient(config_name=config, trace=True) as client:
        for read in yield_reads(read_file):
            called = client.basecall(read)[1]
            return read, called


def align(reference_file, query, mapq):
    a = mp.Aligner(reference_file, preset='map-ont')  # Load or build index
    if not a:
        raise Exception("ERROR: failed to load/build index")

    for hit in a.map(query):  # Traverse alignments
        if hit.is_primary:  # Check if the alignment is primary
            if hit.mapq < mapq:  # Check if the mapping quality is below set threshold
                return None
            return hit

    return None


def get_motif_positions(reference_file, alignment, motif, index):
    for seq_record in SeqIO.parse(reference_file, 'fasta'):
        reference = str(seq_record.seq[alignment.r_st: alignment.r_en])

    if alignment.strand == -1:  # Reverse alignment strand
        reference = mp.revcomp(reference)

    return [m.start() + index for m in re.finditer(motif, reference, re.I)]


def sequence_to_raw(read, called):
    first_signal_id = len(read.signal) - called.trimmed_samples
    move_index = np.nonzero(called.move)[0]

    seq_to_raw_start = first_signal_id + move_index * called.model_stride
    seq_to_raw_len = np.diff(seq_to_raw_start, append=len(read.signal))
    seq_to_raw_end = seq_to_raw_start + seq_to_raw_len

    return list(zip(seq_to_raw_start, seq_to_raw_end))


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


def custom_processor(fast5_file, reference_file, mapq=0, motif='CG', index=0, window=8):
    read, called = basecall(fast5_file)

    alignment = align(reference_file, called.seq, mapq)
    if not alignment:
        return

    motif_positions = get_motif_positions(reference_file, alignment, motif, index)

    ref_to_bc = reference_to_basecall(alignment)
    seq_to_raw = sequence_to_raw(read, called)

    for motif_position in motif_positions:
        event_intervals = resolve_region(motif_position, ref_to_bc, seq_to_raw, window)
        if event_intervals is None:
            continue

        position = alignment.r_st + motif_position
        event_lens = [end - start for start, end in event_intervals]

        print(f'position: {position}\nevent_intervals: {event_intervals}\nevent_lens:{event_lens}\n')


if __name__ == '__main__':
    modification = 'mod'  # 'mod' or 'nomod'
    dir_in = Path(f'/home/sdeur/data/{modification}')
    reference_file = '/home/sdeur/data/ecoli_k12_mg1655.fasta'

    fast5_files = get_fast5_files(dir_in, recursive=True)

    for fast5_file in tqdm(fast5_files):
        custom_processor(fast5_file, reference_file)
