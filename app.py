from sparky.inference import load_model, calc_token_metrics, visualize_batch

model, tokenizer = load_model("gpt2")

texts = [
    "I am a language model.",
    # "In a shocking turn of events, the seemingly impossible task",
    # "In a shocking turn of table, the seemingly impossible task",
    # "Hello",
]
metrics = calc_token_metrics(texts, model, tokenizer)
print(visualize_batch(metrics))
