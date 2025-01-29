from .calc_metrics import calc_token_metrics, load_model
from .visualize import visualize_batch

model, tokenizer = load_model('gpt2')

texts = [
    "I am a language model.",
    "The quick brown fox jumped over the lazy dog.",    # Short
    "In a shocking turn of events, the seemingly impossible task", # Long, truncated
    "In a shocking turn of table, the seemingly impossible task", # Long, truncated
    "Hello"                   # Tiny!
]
metrics = calc_token_metrics(texts, model, tokenizer)
visualize_batch(metrics)
