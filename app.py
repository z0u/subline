import sys
from functools import lru_cache, wraps
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import gradio as gr
from pydantic import Field, ValidationError, validate_call

from subline.inference import calc_token_metrics, load_model, visualize_batch
from subline.inference.visualize import MetricType

model, tokenizer = load_model("gpt2")


# Example inputs showing different linguistic patterns
examples = [
    ["A long time ago, in a galaxy somewhat far away..."],
    ["In a shocking turn of table, the seemingly impossible task"],
    ["The quick brown fox jumps over the lazy dog."],
]


def catch_all(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs) -> str:
        try:
            return fn(*args, **kwargs)
        except ValidationError as e:
            return "<br>".join(
                f"{', '.join(err['loc'])}: {err['msg']}"
                for err in e.errors(
                    include_url=False,
                    include_context=False,
                    include_input=True,
                )
            )
        except Exception as e:
            print(f"Error processing text: {str(e)}", file=sys.stderr)
            return "Sorry, there was an error processing your text. Please try a different input."

    return wrapper


@catch_all
@validate_call(validate_return=True)
def _analyze_text(
    text: str = Field(..., min_length=1, max_length=2000),
    line_width: int = Field(..., ge=20, le=200),
    metrics_to_show: tuple[MetricType, ...] = Field(..., min_length=1),
) -> str:
    metrics = calc_token_metrics([text], model, tokenizer)
    svgs = visualize_batch(metrics, metrics_to_show=metrics_to_show, line_width=line_width)
    return svgs[0]


@lru_cache(128)
def analyze_text(
    text: str,
    line_width: int,
    surprisal: bool,
    entropy: bool,
    s2: bool,
) -> str:
    metrics_to_show = tuple()
    if surprisal:
        metrics_to_show += ("surprisal",)
    if entropy:
        metrics_to_show += ("entropy",)
    if s2:
        metrics_to_show += ("s2",)

    return _analyze_text(text=text, line_width=line_width, metrics_to_show=metrics_to_show)


# Define UI components
text_input = gr.Textbox(
    label="text",
    placeholder="Enter some text to analyze...",
    lines=3,
    value=examples[0][0],
)

width_slider = gr.Slider(
    label="line_width",
    minimum=20,
    maximum=120,
    step=5,
    value=30,
)

metric_toggles = [
    gr.Checkbox(label="Surprisal", value=False),
    gr.Checkbox(label="Entropy", value=False),
    gr.Checkbox(label="S₂", value=True),
]

inputs = [text_input, width_slider, *metric_toggles]

empty_sample = [None] * len(inputs)
examples = [sample + empty_sample[len(sample) :] for sample in examples]

# Create interface
demo = gr.Interface(
    fn=analyze_text,
    inputs=inputs,
    outputs=gr.HTML(),
    title="Subline: Token Information Content Visualization",
    description="""
    Visualize how predictable each token is according to GPT-2. The metrics shown are:
    
    - **Surprisal**: Actual information content (-log probability) of each token
    - **Entropy**: Expected information content (uncertainty) at each position
    - **S₂** (surprise-surprise): How much more/less surprising a token is than expected (surprisal - entropy)

    Read the paper for more details about this visualization and S₂:
    [Detecting out of distribution text with surprisal and entropy][s2]

    [s2]: https://www.lesswrong.com/posts/Kjo64rSWkFfc3sre5/detecting-out-of-distribution-text-with-surprisal-and
    """,
    examples=examples,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
