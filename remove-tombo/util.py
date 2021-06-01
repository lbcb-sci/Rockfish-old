from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class ResegmentationData:
    position: int
    event_intervals: List[Tuple]  # Intervals of signal points
    event_lens: np.ndarray  # Lengths of intervals of signal points
    bases: str
