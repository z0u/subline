from dataclasses import dataclass
import math
import torch

@dataclass
class Series:
    raw: torch.Tensor
    color: str = ""
    dasharray: str = ""
    label: str = ""
    
    @property
    def values(self) -> torch.Tensor:
        """Normalized values ready for visualization"""
        return self.normalize(self.raw)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Default implementation just passes through raw values"""
        return x


@dataclass
class EntropySeries(Series):
    vocab_size: int = 256
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        max_entropy = math.log(self.vocab_size)
        return x / max_entropy
