---
title: Sparky
emoji: 📈
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.13.2
app_file: app.py
pinned: false
python_version: "3.12"
license: mit
short_description: Token metrics visualization
---

Visualize how predictable each token is according to GPT-2, with beautiful sparklines that align with text. This tool helps you understand the information content of text at a token level, showing:

- **Surprisal**: How much information each token contains (-log probability)
- **Entropy**: How much uncertainty the model has before seeing each token
- **S₂** (surprise-surprise): Whether tokens are more or less surprising than expected

Perfect for:
- Analyzing writing style and flow
- Finding unusual word combinations
- Understanding how language models process text


## Examples

Try these examples to see different aspects of the visualization:

1. Classic sentence: "The quick brown fox jumps over the lazy dog"
   - Notice how common words like "the" have low surprisal
   - The unusual sequence "brown fox" shows higher entropy

2. Compare these two sentences:
   - "In a shocking turn of events, the seemingly impossible task"
   - "In a shocking turn of table, the seemingly impossible task"
   - Watch how S₂ spikes on the unexpected word!


## Local Development

To run locally:

```bash
uv venv
uv pip install -r requirements-dev.txt
uv run app.py
```


## Deployment

The app is deployed as a Hugging Face Space. To deploy your own instance:

1. Fork this repository
2. Create a new Space on Hugging Face
3. Link the repository to your Space
4. Push changes to deploy

```bash
huggingface-cli login
git remote add space https://huggingface.co/spaces/your-username/sparky
git push space main
```


## Citation

If you use this visualization in your research, please cite:

```bibtex
@software{text_metrics_viz,
  author = {Sandy Fraser},
  title = {Sparky: A Text Metrics Visualizer},
  year = {2025},
  url = {https://github.com/z0u/sparky}
}
```


## License

MIT
