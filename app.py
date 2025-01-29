import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import gradio as gr
from sparky.inference import load_model, calc_token_metrics, visualize_batch

# Load model on startup (cached)
model, tokenizer = load_model("gpt2")


def analyze_text(
    text: str,
    line_width: int = 80,
    surprisal: bool = False,
    entropy: bool = False,
    s2: bool = True,
) -> str:
    # Default to S2 if nothing selected
    if not any([s2, entropy, surprisal]):
        s2 = True

    # Build list of metrics to show
    metrics_to_show = []
    if surprisal:
        metrics_to_show.append("surprisal")
    if entropy:
        metrics_to_show.append("entropy")
    if s2:
        metrics_to_show.append("s2")

    # Calculate metrics and generate visualization
    metrics = calc_token_metrics([text], model, tokenizer)
    svgs = visualize_batch(
        metrics, metrics_to_show=metrics_to_show, line_width=line_width
    )
    return svgs[0]  # Return first (only) SVG


# Create Gradio interface
demo = gr.Interface(
    fn=analyze_text,
    inputs=[
        gr.Textbox(
            label="Text to analyze",
            placeholder="Enter some text to analyze its information content...",
            lines=3,
            value="The quick brown fox jumps over the lazy dog.",
        ),
        gr.Slider(
            label="Line width (characters)",
            minimum=30,
            maximum=200,
            step=10,
            value=30,
        ),
        gr.Checkbox(label="Show surprisal", value=True),
        gr.Checkbox(label="Show entropy", value=True),
        gr.Checkbox(label="Show S₂ (surprise-surprise)"),
    ],
    outputs=gr.HTML(),
    title="Token Information Content Visualization",
    description="""
    Visualize how predictable each token is according to GPT-2. The metrics shown are:
    
    - **Surprisal**: Actual information content (-log probability) of each token
    - **Entropy**: Expected information content (uncertainty) at each position
    - **S₂** (surprise-surprise): How much more/less surprising a token is than expected
    """,
    examples=[
        ["The quick brown fox jumps over the lazy dog.", 30, True, True, False],
        [
            "In a shocking turn of events, the seemingly impossible task",
            30,
            False,
            False,
            True,
        ],
        [
            "In a shocking turn of table, the seemingly impossible task",
            30,
            False,
            False,
            True,
        ],
        ["A long time ago, in a galaxy far, far away...", 50, True, False, True],
    ],
)

if __name__ == "__main__":
    demo.launch(show_error=True)
