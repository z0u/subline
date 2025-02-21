from dataclasses import dataclass
import math
import numpy as np


@dataclass
class Series:
    """A series of values for visualization with optional styling."""

    raw: np.ndarray
    color: str = ""
    dasharray: str = ""
    label: str = ""

    @property
    def values(self) -> np.ndarray:
        """Normalized values ready for visualization."""
        return self.normalize(self.raw)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Default implementation just passes through raw values."""
        return x


@dataclass
class EntropySeries(Series):
    """A series that normalizes values relative to maximum possible entropy."""

    vocab_size: int = 256

    def normalize(self, x: np.ndarray) -> np.ndarray:
        max_entropy = math.log(self.vocab_size)
        return x / max_entropy
