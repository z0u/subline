import time

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(name="gpt2"):
    print(f"Loading model {name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model from pretrained...")
    t0 = time.perf_counter()
    model = GPT2LMHeadModel.from_pretrained(name).to(device)
    t1 = time.perf_counter()
    print(f"Model loaded in {t1 - t0:.1f}s")

    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(name, clean_up_tokenization_spaces=True)
    t2 = time.perf_counter()
    print(f"Tokenizer loaded in {t2 - t1:.1f}s")

    # Truncate from start to preserve adversarial suffixes.
    tokenizer.truncation_side = "left"

    return model, tokenizer
