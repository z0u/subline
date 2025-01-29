from sparky.inference.visualize import visualize_batch

model, tokenizer = load_model('gpt2')

texts = [
    "I am a language model.",
    "In a shocking turn of events, the seemingly impossible task",
    "In a shocking turn of table, the seemingly impossible task",
    "Hello",
]
metrics = calc_token_metrics(texts, model, tokenizer)

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
