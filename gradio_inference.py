from dataclasses import dataclass
from pathlib import Path

import modal
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = modal.App(name="flux-dev")
USE_WANDB = False
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "accelerate==0.30.0",
    "datasets~=2.13.0",
    "ftfy~=6.1.0",
    "gradio~=3.50.2",
    "smart_open~=6.4.0",
    "transformers~=4.41.2",
    "sentencepiece>=0.1.91,!=0.1.92",
    "torch~=2.2.0",
    "torchvision~=0.16",
    "triton~=2.2.0",
    "peft==0.11.1",
    "wandb==0.17.6",
    "diffusers==0.30.0",
)


@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    # The instance name is the "proper noun" we're teaching the model
    instance_name: str = "Qwerty"
    # That proper noun is usually a member of some class (person, bird),
    # and sharing that information with the model helps it generalize better.
    class_name: str = "Golden Retriever"
    # identifier for pretrained models on Hugging Face
    model_name: str = "black-forest-labs/FLUX.1-dev"

    USE_WANDB: bool = False  # whether to use Weights & Biases for tracking training


# def download_models():
#     from diffusers import DiffusionPipeline, FluxPipeline
#     from transformers.utils import move_cache

#     config = SharedConfig()

#     FluxPipeline.from_pretrained(config.model_name)
#     move_cache()


# image = image.run_function(
#     download_models, secrets=[modal.Secret.from_name("my-huggingface-secret")]
# )


volume = modal.Volume.from_name("flux-lora-private")
model_vol = modal.Volume.from_name("flux-model")
MODEL_DIR = "/flux-model"
CHECKPOINT_DIR = "/flux-lora-private"


def load_images(image_urls: list[str]) -> Path:
    import PIL.Image
    from smart_open import open

    img_path = Path("/img")

    img_path.mkdir(parents=True, exist_ok=True)
    for ii, url in enumerate(image_urls):
        with open(url, "rb") as f:
            image = PIL.Image.open(f)
            image.save(img_path / f"{ii}.png")
    print(f"{ii + 1} images loaded")

    return img_path


def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False


@app.cls(image=image, gpu="H100", volumes={MODEL_DIR: model_vol, CHECKPOINT_DIR: volume})
class Model:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import DiffusionPipeline, FluxPipeline
        from safetensors.torch import load_file  # Import safetensors module
        import os, sys


        # Reload the modal.Volume to ensure the latest state is accessible.
        volume.reload()

        print(os.system("ls /flux-model"))

        # Set up a Hugging Face inference pipeline using our model
        pipe = FluxPipeline.from_pretrained(
            f"{MODEL_DIR}/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        pipe.safety_checker = disabled_safety_checker

        # Load LoRA weights from safetensors file
        lora_weights_path = (
            f"{CHECKPOINT_DIR}/flux_lora_private/flux_lora_private.safetensors"
        )
        lora_weights = load_file(lora_weights_path)
        pipe.load_lora_weights(lora_weights)

        self.pipe = pipe

    @modal.method()
    def inference(self, text, config):
        image = self.pipe(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]

        return image


web_app = FastAPI()
assets_path = Path(__file__).parent / "assets"


@dataclass
class AppConfig(SharedConfig):
    """Configuration information for inference."""

    num_inference_steps: int = 50
    guidance_scale: float = 2


@app.function(
    image=image,
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
    mounts=[modal.Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@modal.asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    # Call out to the inference in a separate Modal environment with a GPU
    def go(text=""):
        if not text:
            text = example_prompts[0]
        return Model().inference.remote(text, config)

    # set up AppConfig
    config = AppConfig()

    instance_phrase = f"{config.instance_name} the {config.class_name}"

    example_prompts = [
        f"{instance_phrase}",
        f"a painting of {instance_phrase.title()} With A Pearl Earring, by Vermeer",
        f"oil painting of {instance_phrase} flying through space as an astronaut",
        f"a painting of {instance_phrase} in cyberpunk city. character design by cory loftis. volumetric light, detailed, rendered in octane",
        f"drawing of {instance_phrase} high quality, cartoon, path traced, by studio ghibli and don bluth",
    ]

    # custom styles: an icon, a background, and a theme
    @web_app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse("/assets/favicon.svg")

    @web_app.get("/assets/background.svg", include_in_schema=False)
    async def background():
        return FileResponse("/assets/background.svg")

    with open("/assets/index.css") as f:
        css = f.read()

    theme = gr.themes.Default(
        primary_hue="green", secondary_hue="emerald", neutral_hue="neutral"
    )

    # add a gradio UI around inference
    with gr.Blocks(theme=theme, css=css, title="Inference") as interface:
        with gr.Row():
            inp = gr.Textbox(  # input text component
                label="",
                placeholder=f"Describe the version of {instance_phrase} you'd like to see",
                lines=10,
            )
            out = gr.Image(  # output image component
                height=512, width=512, label="", min_width=512, elem_id="output"
            )
        with gr.Row():
            btn = gr.Button("Dream", variant="primary", scale=2)
            btn.click(
                fn=go, inputs=inp, outputs=out
            )  # connect inputs and outputs with inference function

        with gr.Column(variant="compact"):
            # add in a few examples to inspire users
            for ii, prompt in enumerate(example_prompts):
                btn = gr.Button(prompt, variant="secondary")
                btn.click(fn=lambda idx=ii: example_prompts[idx], outputs=inp)

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


@app.local_entrypoint()
def run(  # add more config params here to make training configurable
    max_train_steps: int = 250,
):
    fastapi_app.remote()


# to run: modal serve flux_gradio.py
