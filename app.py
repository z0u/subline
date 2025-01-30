import sys
from pathlib import Path
from functools import lru_cache, wraps

sys.path.append(str(Path(__file__).parent / "src"))

from pydantic import validate_call, Field, ValidationError
import gradio as gr

from sparky.inference import load_model, calc_token_metrics, visualize_batch
from sparky.inference.visualize import MetricType


model, tokenizer = load_model("gpt2")


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
    svgs = visualize_batch(
        metrics, metrics_to_show=metrics_to_show, line_width=line_width
    )
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

    return _analyze_text(
        text=text, line_width=line_width, metrics_to_show=metrics_to_show
    )


# Define UI components
text_input = gr.Textbox(
    label="text",
    placeholder="Enter some text to analyze...",
    lines=3,
    value="The quick brown fox jumps over the lazy dog.",
)

width_slider = gr.Slider(
    label="line_width",
    minimum=20,
    maximum=120,
    step=10,
    value=30,
)

metric_toggles = [
    gr.Checkbox(label="Surprisal", value=False),
    gr.Checkbox(label="Entropy", value=False),
    gr.Checkbox(label="S₂", value=True),
]

inputs = [text_input, width_slider, *metric_toggles]

# Example inputs showing different linguistic patterns
examples = [
    ["The quick brown fox jumps over the lazy dog."],
    ["In a shocking turn of events, the seemingly impossible task"],
    ["In a shocking turn of table, the seemingly impossible task"],
    ["A long time ago, in a galaxy far, far away..."],
]
empty_sample = [None] * len(inputs)
examples = [sample + empty_sample[len(sample) :] for sample in examples]

# Create interface
demo = gr.Interface(
    fn=analyze_text,
    inputs=inputs,
    outputs=gr.HTML(),
    title="Token Information Content Visualization",
    description="""
    Visualize how predictable each token is according to GPT-2. The metrics shown are:
    
    - **Surprisal**: Actual information content (-log probability) of each token
    - **Entropy**: Expected information content (uncertainty) at each position
    - **S₂** (surprise-surprise): How much more/less surprising a token is than expected (surprisal - entropy)

    Read the paper for more details about this visualization and S₂: [Detecting out of distribution text with surprisal and entropy](https://www.lesswrong.com/posts/Kjo64rSWkFfc3sre5/detecting-out-of-distribution-text-with-surprisal-and#)
    """,
    examples=examples,
)

if __name__ == "__main__":
    demo.launch()
