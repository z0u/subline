import math

from .decompose import select
from .sparky import Sparky


def visualize_batch(metrics: dict) -> None:
    """Draw token-level metrics for each sequence in a batch: sparklines, KDE plots"""
    import ml_utils as mu
    tokens, surprisals, entropies, vocab_size = select(metrics, 'tok_str, tok_surp, tok_ent, vocab_size')
    for seq, surp, ent in zip(tokens, surprisals, entropies):
        n = len(seq)
        surp = surp[:n]
        ent = ent[:n]
        vis = Sparky(chars_per_line=60)
        vis.visualize(seq, [
            # Base metrics
            # mu.EntropySeries(surp, vocab_size=vocab_size, label='Surprisal'),
            # mu.EntropySeries(ent, vocab_size=vocab_size, label='Entropy'),

            # Surprise-surprise as two series, with negative values mirrored around the baseline
            mu.Series((surp - ent)/math.log(vocab_size), label="S₂"),
            mu.Series(-(surp - ent)/math.log(vocab_size), label="-S₂", dasharray='3'),

            # Earlier attempts at relative surprisal
            # mu.Series(surp / ent / 2, label='ERS'),
            # mu.Series(torch.log(surp / (ent+1) + 1), label='ERS'),
            # mu.Series(torch.log(torch.log(surp / (ent+1) + 1) + 1), label='ERS'),
        ])
