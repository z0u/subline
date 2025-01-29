import gradio as gr
from sparky.inference import load_model, calc_token_metrics, visualize_batch

# Load model on startup (cached)
model, tokenizer = load_model("gpt2")


def analyze_text(
    text: str, s2: bool = True, entropy: bool = False, surprisal: bool = False
) -> str:
    # Default to S2 if nothing selected
    if not any([s2, entropy, surprisal]):
        s2 = True

    # Build list of metrics to show
    metrics_to_show = []
    if s2:
        metrics_to_show.append("s2")
    if entropy:
        metrics_to_show.append("entropy")
    if surprisal:
        metrics_to_show.append("surprisal")

    # Calculate metrics and generate visualization
    metrics = calc_token_metrics([text], model, tokenizer)
    svgs = visualize_batch(metrics, metrics_to_show=metrics_to_show)
    return svgs[0]  # Return first (only) SVG


# Create Gradio interface
demo = gr.Interface(
    fn=analyze_text,
    inputs=[
        gr.Textbox(
            label="Text to analyze",
            placeholder="Enter some text to analyze its information content...",
            lines=3,
        ),
        gr.Checkbox(label="Show S₂ (surprise-surprise)", value=True),
        gr.Checkbox(label="Show entropy"),
        gr.Checkbox(label="Show surprisal"),
    ],
    outputs=gr.HTML(),
    title="Token Information Content Visualization",
    description="""
    Visualize how predictable each token is according to GPT-2. The metrics shown are:
    
    - **S₂** (surprise-surprise): How much more/less surprising a token is than expected
    - **Entropy**: Expected information content (uncertainty) at each position
    - **Surprisal**: Actual information content (-log probability) of each token
    """,
    examples=[
        ["The quick brown fox jumps over the lazy dog.", True, False, False],
        [
            "In a shocking turn of events, the seemingly impossible task",
            True,
            True,
            False,
        ],
        ["Once upon a time, in a galaxy far, far away...", True, False, True],
    ],
)

if __name__ == "__main__":
    demo.launch()
