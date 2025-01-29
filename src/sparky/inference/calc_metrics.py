from typing import List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .types import TokenMetrics


def clean_tokens_for_display(token_ids: List[int], tokenizer) -> List[str]:
    """Convert token IDs to readable strings, making whitespace visible."""
    texts = tokenizer.batch_decode(token_ids, clean_up_tokenization_spaces=True)
    return [text.replace("\n", "↵").replace("\t", "→") for text in texts]


def calc_token_metrics(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    truncation=False,
) -> TokenMetrics:
    """Calculate per-token metrics for a batch of text sequences using a language model."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    B = len(texts)
    device = next(model.parameters()).device

    # Encode text to tokens
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=truncation,
        return_length=True,
    ).to(device)

    # Get model predictions efficiently
    torch.cuda.empty_cache()  # Clear any leftover memory
    with torch.no_grad(), torch.amp.autocast(device.type):
        outputs = model(input_ids=inputs.input_ids, labels=inputs.input_ids)
        logits = outputs.logits[:, :-1].float()  # [B, T-1, V]

    # Calculate token probabilities
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, T-1, V]
    next_tokens = inputs.input_ids[:, 1:]  # [B, T-1]
    attention_mask = inputs.attention_mask[:, 1:]  # [B, T-1]

    # Get surprisal (negative log probability) of actual next tokens [B, T-1]
    token_log_probs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
    surprisals = -token_log_probs * attention_mask

    # Calculate entropy (expected information content of distribution) [B, T-1]
    probs = torch.exp(log_probs).clamp(min=1e-10)  # Prevent exactly 0
    entropies = -torch.sum(probs * log_probs, dim=-1) * attention_mask

    # Aggregate sequence-level metrics
    # Note: torch.nanmean handles the padding properly
    sequence_entropy = torch.nanmean(entropies * attention_mask, dim=1)
    mean_neg_log_prob = -torch.masked.mean(token_log_probs, dim=1, mask=attention_mask)
    sequence_perplexity = torch.exp(mean_neg_log_prob)

    # Add NaN for first token (no context)
    null_first = torch.full((B, 1), float("nan"), device=device)
    surprisals = torch.cat((null_first, surprisals), 1)  # [B, T]
    entropies = torch.cat((null_first, entropies), 1)  # [B, T]

    # Get readable tokens
    tokens = [
        clean_tokens_for_display(seq[:length], tokenizer)
        for seq, length in zip(inputs.input_ids, inputs.length)
    ]

    return TokenMetrics(
        tokens=tokens,
        surprisal=surprisals.cpu(),
        entropy=entropies.cpu(),
        sequence_entropy=sequence_entropy.cpu(),
        sequence_perplexity=sequence_perplexity.cpu(),
        sequence_length=inputs.length.cpu(),
        vocab_size=len(tokenizer),
    )
