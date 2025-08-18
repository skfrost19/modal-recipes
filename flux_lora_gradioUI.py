from pathlib import Path
import modal
from fastapi import FastAPI
from typing import List


app = modal.App(name="flux-dev")
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
    "diffusers",
)

volume = modal.Volume.from_name("flux-lora-private")
model_vol = modal.Volume.from_name("flux-model")
loras_volume = modal.Volume.from_name("loras", create_if_missing=True)
MODEL_DIR = "/flux-model"
CHECKPOINT_DIR = "/flux-lora-private"
LORAS_DIR = "/loras"


# Please set the secrets in the modal dashboard
@app.function(image=image, volumes={MODEL_DIR: model_vol, LORAS_DIR: loras_volume}, secrets=[modal.Secret.from_name("my-huggingface-secret")], timeout=32000)
def download_models(models: List[str]):
    # using huggingface-cli
    import os

    for model in models:
        local_dir_name = model.split("/")[-1]
        os.system(f"huggingface-cli download {model} --local-dir {LORAS_DIR}/{local_dir_name}")
        model_vol.commit()

        print(f"Downloaded {model}")


@app.cls(image=image, gpu="H100", volumes={MODEL_DIR: model_vol, CHECKPOINT_DIR: volume, LORAS_DIR: loras_volume})
class Model:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import FluxPipeline
        from safetensors.torch import load_file
        
        volume.reload()
        
        pipe = FluxPipeline.from_pretrained(
            f"{MODEL_DIR}/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        lora_weights_path = f"{LORAS_DIR}/Flux-Uncensored-V2/lora.safetensors"
        lora_weights = load_file(lora_weights_path)
        pipe.load_lora_weights(lora_weights)

        self.pipe = pipe

    @modal.method()
    def inference(self, text, num_inference_steps, guidance_scale):
        image = self.pipe(
            text,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        return image

web_app = FastAPI()

@app.function(
    image=image,
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
)
@modal.asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    def generate(text, num_steps, guidance):
        return Model().inference.remote(text, num_steps, guidance)

    with gr.Blocks(title="Inference") as interface:
        gr.Markdown("<h1 style='text-align: center'>Flux + LoRA Image Generation</h1>")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here",
                    lines=5  # Increased from 3 to 5
                )
                num_steps = gr.Slider(
                    minimum=1, 
                    maximum=50, 
                    value=50, 
                    step=1, 
                    label="Number of Inference Steps"
                )
                guidance = gr.Slider(
                    minimum=1.0, 
                    maximum=20.0, 
                    value=2.0, 
                    step=0.1, 
                    label="Guidance Scale"
                )
                generate_btn = gr.Button("Generate")
            
            with gr.Column(scale=2):
                out = gr.Image(
                    height=512, 
                    width=512, 
                    label="Generated Image"
                )
        
        generate_btn.click(
            fn=generate, 
            inputs=[inp, num_steps, guidance], 
            outputs=out
        )

    return mount_gradio_app(app=web_app, blocks=interface, path="/")
@app.local_entrypoint()
def run():
    # download_models.remote(["enhanceaiteam/Flux-Uncensored-V2"])
    fastapi_app.remote()


# to run: modal serve flux_gradio.py
