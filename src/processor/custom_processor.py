from pyguppyclient.decode import ReadData, CalledReadData
import mappy
import numpy as np

from typing import Tuple, Set, List, Optional

from .alignment import make_aligner, get_reference, get_motif_positions
from .basecall import sequence_to_raw
from .util import Interval, ResegmentationData


class CustomProcessor:
    def __init__(self,
                 basecall_data: Tuple[ReadData, CalledReadData],
                 reference_file: str,
                 mapq: int = 0,
                 motif: str = 'CG',
                 index: int = 0,
                 window: int = 8):
        self.basecall_data = basecall_data
        self.reference_file = reference_file

        self.mapq = mapq
        self.window = window

        self.aligner = make_aligner(reference_file)
        self.motif_positions = get_motif_positions(reference_file, motif, index)

    def align(self, query: str) -> Optional[mappy.Alignment]:
        for hit in self.aligner.map(query):  # Traverse alignments
            if hit.is_primary:  # Check if the alignment is primary
                if hit.mapq < self.mapq:  # Check if the mapping quality is below set threshold
                    return None
                return hit
        return None

    def _get_relevant_motif_positions(self, alignment: mappy.Alignment) -> Set[int]:
        contig_positions = self.motif_positions[alignment.ctg]
        strand_positions = contig_positions[0] if alignment.strand == 1 else contig_positions[1]

        relevant_positions = strand_positions & set(range(alignment.r_st, alignment.r_en))

        if alignment.strand == 1:
            return {pos - alignment.r_st for pos in relevant_positions}
        else:
            return {alignment.r_en - 1 - pos for pos in relevant_positions}

    @staticmethod
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

    @staticmethod
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

    def process(self) -> Optional[List[ResegmentationData]]:
        read, called = self.basecall_data

        alignment = self.align(called.seq)
        if not alignment:
            return None

        relevant_motif_positions = self._get_relevant_motif_positions(alignment)
        if not relevant_motif_positions:
            return None

        seq_to_raw = sequence_to_raw(read, called)

        signal_intervals, deletion_idx = CustomProcessor.resolve_insertions(alignment, seq_to_raw)
        signal_intervals = CustomProcessor.resolve_deletions(signal_intervals, deletion_idx)

        resegmentation_data = []

        for motif_position in relevant_motif_positions:
            r_len = alignment.r_en - alignment.r_st
            if motif_position - self.window < 0 or motif_position + self.window >= r_len:
                continue

            position = alignment.r_st + motif_position if alignment.strand == 1 else alignment.r_en - 1 - motif_position

            event_intervals = signal_intervals[motif_position - self.window: motif_position + self.window + 1]
            event_lens = np.array([interval.end - interval.start for interval in event_intervals])

            reference = get_reference(self.reference_file, alignment.ctg)
            region = reference[position - self.window: position + self.window + 1]
            bases = region if alignment.strand == 1 else mappy.revcomp(region)

            assert len(event_intervals) == len(event_lens) == len(bases)

            resegmentation_data.append(ResegmentationData(position, event_intervals, event_lens, bases))

        return resegmentation_data
