import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizerimport torch


def load_model(name='gpt2'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel.from_pretrained(name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(name, clean_up_tokenization_spaces=True)
    return model, tokenizer


def calc_token_metrics(texts: list[str], model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, truncation=False) -> dict:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    B = len(texts)
    device = next(model.parameters()).device

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=truncation, return_length=True).to(device)
    
    torch.cuda.empty_cache()  # Clear any leftover memory
    with torch.no_grad(), torch.amp.autocast(device.type):  # Enable mixed precision
        outputs = model(input_ids=inputs.input_ids, labels=inputs.input_ids)
        logits = outputs.logits[:, :-1].float()

    log_probs = torch.log_softmax(logits, dim=-1)
    next_tokens = inputs.input_ids[:, 1:]
    attention_mask = inputs.attention_mask[:, 1:]
    
    # Get token log probs directly without going through exp
    # unsqueeze -> gather -> squeeze lets us select one item per token per sequence
    token_log_probs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
    
    # Calculate perplexity in log space as much as possible
    # perplexity = exp(-log_prob) = exp(abs(log_prob)) since log_probs are negative
    surprisals = -token_log_probs * attention_mask
    perplexities = torch.exp(torch.abs(token_log_probs)) * attention_mask
    
    # For entropy, we actually need the probs
    probs = torch.exp(log_probs).clamp(min=1e-10)  # Prevent exactly 0
    entropies = -torch.sum(probs * log_probs, dim=-1) * attention_mask

    # Entropies can be aggregated directly, because they're linear...
    sequence_entropy = torch.nanmean(entropies * attention_mask, dim=1)

    # ...But perplexities are exponential. To improve precision, we go back to the sampled log probabilities
    mean_neg_log_prob = -torch.masked.mean(token_log_probs, dim=1, mask=attention_mask)
    sequence_perplexity = torch.exp(mean_neg_log_prob)

    # Add the first token back on (for which we don't have metrics)
    null_first = torch.full((B, 1), float('nan'), device=device)
    surprisals = torch.cat((null_first, surprisals), 1)
    perplexities = torch.cat((null_first, perplexities), 1)
    entropies = torch.cat((null_first, entropies), 1)

    # Get tokens for each sequence (without padding)
    tokens = [
        clean_tokens_for_display(seq[:mask.sum()], tokenizer)
        for seq, mask in zip(inputs.input_ids, inputs.attention_mask)
    ]

    assert all(len(s) == n for s, n in zip(tokens, inputs.length))
    return {
        'tok_str':  tokens,  # list[list[str]] jagged shape [B, Tb<=T]
        'tok_surp': surprisals.cpu(),  # tensor[B, T]
        'tok_perp': perplexities.cpu(),  # tensor[B, T]
        'tok_ent':  entropies.cpu(),  # tensor[B, T]
        'tok_attn': attention_mask.cpu(),  # tensor[B, T]
        'seq_ent':  sequence_entropy.cpu(),  # tensor[B]
        'seq_perp': sequence_perplexity.cpu(),  # tensor[B]
        'seq_len':  inputs.length.cpu(),  # tensor[B]
        'vocab_size': len(tokenizer),  # int
    }


def clean_tokens_for_display(token_ids: list[int], tokenizer) -> list[str]:
    # Get all the texts at once
    texts = tokenizer.batch_decode(token_ids, clean_up_tokenization_spaces=True)
    # Make whitespace visible in each one
    return [text.replace('\n', '↵').replace('\t', '→') for text in texts]
