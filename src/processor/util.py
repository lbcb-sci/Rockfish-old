from collections import namedtuple
from dataclasses import dataclass
import numpy as np

from typing import List


Interval = namedtuple('Interval', ['start', 'end'])  # Start - inclusive, end - exclusive


@dataclass
class ResegmentationData:
    position: int
    event_intervals: List[Interval]  # Intervals of signal points
    event_lens: np.ndarray  # Lengths of intervals of signal points
    bases: str
