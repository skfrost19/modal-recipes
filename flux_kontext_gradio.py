# -*- coding: utf-8 -*-
# output-directory: /tmp/stable-diffusion

# # Edit images with Flux Kontext

# In this example, we run the Flux Kontext model in _image-to-image_ mode:
# the model takes in a prompt and an image and edits the image to better match the prompt.

# For example, the model edited the first image into the second based on the prompt
# "A cute dog wizard inspired by Gandalf from Lord of the Rings, featuring detailed fantasy elements in Studio Ghibli style".

#  <img src="https://modal-cdn.com/dog-wizard-ghibli-flux-kontext.jpg" alt="A photo of a dog transformed into a cartoon of a cute dog wizard" />

# The model is Black Forest Labs' [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).
# Learn more about the model [here](https://bfl.ai/announcements/flux-1-kontext-dev).

# ## Credits
# Original script by Modal Labs
# Gradio UI implementation added by skfrost19 (https://github.com/skfrost19)

# ## Define a container image

# First, we define the environment the model inference will run in,
# the [container image](https://modal.com/docs/guide/custom-container).

from io import BytesIO
from pathlib import Path

import modal

diffusers_commit_sha = "00f95b9755718aabb65456e791b8408526ae6e76"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("git")
    .uv_pip_install(
        "accelerate~=1.8.1",
        "git+https://github.com/huggingface/diffusers.git@" + diffusers_commit_sha,
        "huggingface-hub[hf-transfer]~=0.33.1",
        "Pillow~=11.2.1",
        "safetensors~=0.5.3",
        "transformers~=4.53.0",
        "sentencepiece~=0.2.0",
        "torch==2.7.1",
        "optimum-quanto==0.2.7",
        extra_options="--index-strategy unsafe-best-match",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
)

MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"
MODEL_REVISION = "f9fdd1a95e0dfd7653cb0966cda2486745122695"

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {str(CACHE_DIR): cache_volume}

secrets = [modal.Secret.from_name("huggingface-secret")]


image = image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Allows faster model downloads
        "HF_HOME": str(CACHE_DIR),  # Points the Hugging Face cache to a Volume
    }
)

# Web UI image with Gradio dependencies
web_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi",
    "gradio",  # Using a more stable version
    "pillow",
)

app = modal.App("example-image-to-image")

with image.imports():
    import torch
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    from PIL import Image


# ## Setting up and running Flux Kontext

# The Modal `Cls` defined below contains all the logic to set up and run Flux Kontext.

# The [container lifecycle](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta) decorator
# (`@modal.enter()`) ensures that the model is loaded into memory when a container starts, before it picks up any inputs.

# The `inference` method runs the actual model inference. It takes in an image as a collection of `bytes` and a string `prompt` and returns
# a new image (also as a collection of `bytes`).

# To avoid excessive cold-starts, we set the `scaledown_window` to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down.


@app.cls(
    image=image, gpu="B200", volumes=volumes, secrets=secrets, scaledown_window=240
)
class Model:
    @modal.enter()
    def enter(self):
        print(f"Downloading {MODEL_NAME} if necessary...")

        dtype = torch.bfloat16

        self.seed = 42
        self.device = "cuda"

        self.pipe = FluxKontextPipeline.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,  # This is crucial for the correct model version
            torch_dtype=dtype,
            cache_dir=CACHE_DIR,
        ).to(self.device)

    @modal.method()
    def inference(
        self,
        image_bytes: bytes,
        prompt: str,
        guidance_scale: float = 2.5,
        num_inference_steps: int = 28,
    ) -> bytes:
        # Load the original image without resizing to preserve dimensions
        init_image = load_image(Image.open(BytesIO(image_bytes)))

        # Get original image dimensions to maintain exact size
        original_width, original_height = init_image.size

        # Use a different seed for each inference to avoid repetitive results
        import random

        seed = random.randint(0, 2**32 - 1)

        image = self.pipe(
            image=init_image,
            guidance_scale=guidance_scale,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            output_type="pil",
            generator=torch.Generator(device=self.device).manual_seed(seed),
            max_sequence_length=512,  # Maximum for best quality
        ).images[0]

        byte_stream = BytesIO()
        # Save with maximum quality to preserve details
        image.save(byte_stream, format="PNG", optimize=False, compress_level=0)
        image_bytes = byte_stream.getvalue()

        return image_bytes


# ## Gradio Web UI

# You can deploy the Gradio web interface with:
# ```bash
# modal deploy flux_kontext.py
# ```
# This will create a web interface accessible to anyone with the URL.


@app.function(
    image=web_image,
    min_containers=1,
    scaledown_window=60 * 20,
    # Gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 100 concurrent inputs
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    """A Gradio interface for Flux Kontext image editing."""
    import io
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from PIL import Image

    web_app = FastAPI()

    def edit_image(input_image, prompt, guidance_scale, num_inference_steps):
        """Process the image editing request."""
        if input_image is None:
            return None

        try:
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            input_image.save(img_byte_arr, format="PNG")
            input_image_bytes = img_byte_arr.getvalue()

            # Call the inference function
            output_image_bytes = Model().inference.remote(
                input_image_bytes,
                prompt,
                float(guidance_scale),
                int(num_inference_steps),
            )

            # Convert bytes back to PIL image
            output_image = Image.open(io.BytesIO(output_image_bytes))
            return output_image

        except Exception:
            return None

    # Create the Gradio interface with more explicit typing
    with gr.Blocks(
        title="Flux Kontext Image Editor",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# Flux Kontext Image Editor")
        gr.Markdown(
            "Upload an image and provide a prompt to edit it using the Flux Kontext model. "
            "The model will transform your image to better match your prompt while preserving its structure and quality."
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image", height=400)
                prompt = gr.Textbox(
                    label="Edit Prompt",
                    placeholder="Describe how you want to transform the image...",
                    value="A cute dog wizard inspired by Gandalf from Lord of the Rings, featuring detailed fantasy elements in Studio Ghibli style",
                    lines=3,
                )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=2.5,
                        step=0.1,
                        label="Guidance Scale",
                    )
                    num_inference_steps = gr.Slider(
                        minimum=15,
                        maximum=50,
                        value=28,
                        step=1,
                        label="Inference Steps",
                    )

                edit_btn = gr.Button("Edit Image", variant="primary", size="lg")

            with gr.Column():
                output_image = gr.Image(
                    label="Edited Image",
                    height=600,
                    width=None,  # Let width auto-adjust to maintain aspect ratio
                    show_download_button=True,
                )

        # Example images and prompts
        gr.Markdown("## Example Prompts")

        # Simplified examples without potential problematic schema
        example_prompts = [
            "A majestic lion in the style of a Renaissance painting",
            "A cyberpunk cityscape with neon lights and flying cars",
            "An oil painting of a serene forest with magical glowing mushrooms",
            "A professional headshot in corporate style",
        ]

        for i, example_prompt in enumerate(example_prompts):
            with gr.Row():
                btn = gr.Button(
                    f"Example {i+1}: {example_prompt[:50]}...", variant="secondary"
                )
                # Use default parameter to capture the value
                btn.click(lambda _, p=example_prompt: p, outputs=prompt)

        # Set up the event handler
        edit_btn.click(
            fn=edit_image,
            inputs=[input_image, prompt, guidance_scale, num_inference_steps],
            outputs=output_image,
            show_progress=True,
        )

        # Add footer
        footer_text = """---
Powered by [Flux Kontext](https://bfl.ai/announcements/flux-1-kontext-dev) and [Modal](https://modal.com)

This interface automatically scales based on usage and only charges for compute time used."""
        gr.Markdown(footer_text)

    # Enable queueing for handling multiple requests
    demo.queue(max_size=10)

    return mount_gradio_app(app=web_app, blocks=demo, path="/")
