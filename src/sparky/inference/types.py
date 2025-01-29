from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TokenMetrics:
    """Metrics computed for each token in a batch of sequences."""

    # Per-token metrics [B, T]
    tokens: List[List[str]]  # Actual tokens, jagged array
    surprisal: np.ndarray  # -log P(token|context)
    entropy: np.ndarray  # Expected information content

    # Per-sequence metrics [B]
    sequence_entropy: np.ndarray  # Mean entropy across tokens
    sequence_perplexity: np.ndarray  # exp(mean surprisal)
    sequence_length: np.ndarray  # Number of tokens per sequence

    # Model info
    vocab_size: int  # Size of tokenizer vocabulary

    def __post_init__(self):
        """Convert torch tensors to numpy arrays where appropriate."""
        self.surprisal = self.surprisal.numpy()
        self.entropy = self.entropy.numpy()
        self.sequence_entropy = self.sequence_entropy.numpy()
        self.sequence_perplexity = self.sequence_perplexity.numpy()
        self.sequence_length = self.sequence_length.numpy()
