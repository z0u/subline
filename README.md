Visualize metrics as sparklines under text.

![Screenshot of text that reads, "A long time ago, in a galaxy somewhat far away..." A sparkline beneath the text shows that the word "somewhat" is clearly out of distribution (i.e. unexpected) in this context.](doc/example.svg)


## Local Development

```bash
uv venv
uv sync --all-extras
uv run ruff check
uv run pytest
```


## Citation

If you use this visualization in your research, please cite:

```bibtex
@software{text_metrics_viz,
  author = {Sandy Fraser},
  title = {Subline: A Text Metrics Visualizer},
  year = {2025},
  url = {https://github.com/z0u/subline}
}
```


## License

MIT
