import math
from typing import List, Literal

from ..sparky import Sparky
from ..series import Series, EntropySeries
from .types import TokenMetrics


# Available metrics to visualize
MetricType = Literal["surprisal", "entropy", "s2"]


def create_series(
    metrics: TokenMetrics, metric: MetricType, index: int
) -> List[Series]:
    """Create visualization series for the specified metric.

    Args:
        metrics: Token metrics for a batch of sequences
        metric: Which metric to visualize
        index: Batch index to extract metrics for

    Returns:
        List of Series objects to visualize (some metrics like s2 return multiple series)
    """
    n = len(metrics.tokens[index])
    match metric:
        case "entropy":
            return [
                EntropySeries(
                    metrics.entropy[index, :n],
                    vocab_size=metrics.vocab_size,
                    label="Entropy",
                )
            ]
        case "surprisal":
            return [
                EntropySeries(
                    metrics.surprisal[index, :n],
                    vocab_size=metrics.vocab_size,
                    label="Surprisal",
                )
            ]
        case "s2":
            # Return both positive and negative series for S2
            s2 = (metrics.surprisal[index, :n] - metrics.entropy[index, :n]) / math.log(
                metrics.vocab_size
            )
            return [Series(s2, label="+S₂"), Series(-s2, label="-S₂", dasharray="3")]
        case _:
            raise ValueError(f"Unknown metric: {metric}")


def visualize_batch(
    metrics: TokenMetrics,
    line_width: int = 80,
    metrics_to_show: List[MetricType] = ["s2"],
) -> List[str]:
    """Draw token-level metrics for each sequence in a batch.

    Creates sparkline visualizations aligned with text, showing how metrics
    like surprisal and entropy vary across each sequence.

    Args:
        metrics: Token-level metrics for a batch of sequences
        line_width: Maximum characters per line (default: 80)
        metrics_to_show: Which metrics to visualize, in order. Available metrics:
            - 'entropy': Expected information content
            - 's2': Surprise-surprise (surprisal - entropy)
            - 'surprisal': Token surprisal (-log probability)

            Duplicates are allowed if you want to compare variations.

    Returns:
        List of SVG strings, one per sequence in the batch.

    Example:
        >>> metrics = calc_token_metrics(["Hello world"], model, tokenizer)
        >>> svgs = visualize_batch(metrics, metrics_to_show=['entropy', 's2'])
        >>> display_svg(svgs[0])  # Show first sequence
    """
    plots = []
    for i in range(len(metrics.tokens)):
        # Create series for each requested metric
        all_series = []
        for metric in metrics_to_show:
            all_series.extend(create_series(metrics, metric, i))

        # Create sparkline visualization
        vis = Sparky(chars_per_line=line_width)
        vis.margin = 0
        plot = vis.visualize(metrics.tokens[i], all_series)
        plots.append(plot)

    return plots
