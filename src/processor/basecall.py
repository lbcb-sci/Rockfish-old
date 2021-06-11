from pyguppyclient import GuppyBasecallerClient, yield_reads
from pyguppyclient.decode import ReadData, CalledReadData
import numpy as np

from typing import Tuple, List

from .util import Interval


# config='dna_r9.4.1_450bps_hac' for more precise basecalling
def basecall(files: List[str], config: str = 'dna_r9.4.1_450bps_fast') -> Tuple[ReadData, CalledReadData]:
    with GuppyBasecallerClient(config_name=config, trace=True) as client:
        for file in files:
            for read in yield_reads(file):
                called = client.basecall(read)[1]
                yield read, called


def sequence_to_raw(read: ReadData, called: CalledReadData) -> List[Interval]:
    first_signal_id = len(read.signal) - called.trimmed_samples
    move_index = np.nonzero(called.move)[0]

    seq_to_raw_start = first_signal_id + move_index * called.model_stride
    seq_to_raw_len = np.diff(seq_to_raw_start, append=len(read.signal))
    seq_to_raw_end = seq_to_raw_start + seq_to_raw_len

    return [Interval(st, en) for st, en in zip(seq_to_raw_start, seq_to_raw_end)]
