from math import isclose
from typing import NamedTuple

from .utils.decompose import sliceable


@sliceable
class TokenBB(NamedTuple):
    """1D bounding box for a token"""

    # All positions relative to token start
    width: float  # Total width of token
    first_char: float  # Position of first char midpoint
    mid: float  # Midpoint of token
    last_char: float  # Position of last char midpoint

    @property
    def is_wide(self):
        return not isclose(self.first_char, self.last_char, rel_tol=0.05)
