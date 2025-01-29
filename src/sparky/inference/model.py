import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(name="gpt2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(name, clean_up_tokenization_spaces=True)
    return model, tokenizer
