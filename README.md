# Sparky ⚡

A tool for visualizing the information structure of text using metrics derived from language models. It creates inline sparkline visualizations that show how surprisal and entropy vary across tokens, helping writers and researchers analyze text flow and complexity.

![Example visualization](assets/example.png)

## Features

- **Token-level metrics**: Calculate surprisal and entropy for each token using GPT-2
- **Inline visualization**: See metrics directly aligned with text using SVG sparklines
- **Interactive interface**: Web UI for analyzing your own text
- **Theme support**: Automatic light/dark mode detection with manual toggle
- **Exportable SVGs**: Visualization output as clean SVG for use in documents

## Technical Details

The visualization uses GPT-2's token probabilities to compute two key metrics:

- **Surprisal** (red line): How unexpected each token is given the context
- **Entropy** (blue line): How uncertain the model is about what comes next

These are rendered as sparklines aligned with the text, making it easy to spot patterns in information flow and identify potentially confusing or unexpected passages.

## Local Development

```bash
# Clone the repository
git clone https://github.com/z0u/sparky.git
cd text-metrics

# Set up environment with uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run the Gradio interface locally
python app.py
```


## Deployment

The app is deployed as a Hugging Face Space. To deploy your own instance:

1. Fork this repository
2. Create a new Space on Hugging Face
3. Link the repository to your Space
4. Push changes to deploy


## Citation

If you use this visualization in your research, please cite:

```bibtex
@software{text_metrics_viz,
  author = {Sandy Fraser},
  title = {Sparky: A Text Metrics Visualizer},
  year = {2024},
  url = {https://github.com/z0u/sparky}
}
```


## License

MIT