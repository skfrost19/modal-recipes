from dataclasses import dataclass
from pathlib import Path
from typing import List
import modal
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = modal.App(name="flux-dev-upscaler")
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

model_vol = modal.Volume.from_name("flux-model")
MODEL_DIR = "/flux-model"

# Please set the secrets in the modal dashboard
@app.function(image=image, volumes={MODEL_DIR: model_vol}, secrets=[modal.Secret.from_name("my-huggingface-secret")], timeout=32000)
def download_models(models: List[str]):
    # using huggingface-cli
    import os

    for model in models:
        local_dir_name = model.split("/")[-1]
        os.system(f"huggingface-cli download {model} --local-dir {MODEL_DIR}/{local_dir_name}")
        model_vol.commit()

        print(f"Downloaded {model}")


@app.cls(image=image, gpu=modal.gpu.H100(count=1), volumes={MODEL_DIR: model_vol}, timeout=3200)
class UpscalerModel:
    @modal.enter()
    def load_model(self):
        import torch
        import os
        from diffusers import FluxControlNetModel, FluxControlNetPipeline
        from accelerate import dispatch_model
        import gc

        print(os.system("nvidia-smi"))
    
        try:
            gpu_count = torch.cuda.device_count()
            print(f"Available GPUs: {gpu_count}")

            print("Loading ControlNet...")
            controlnet = FluxControlNetModel.from_pretrained(
                f"{MODEL_DIR}/Flux.1-dev-Controlnet-Upscaler",
                torch_dtype=torch.bfloat16
            ).to("cuda:0")
    
            print("Loading Pipeline...")
            pipe = FluxControlNetPipeline.from_pretrained(
                f"{MODEL_DIR}/FLUX.1-dev",
                controlnet=controlnet,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            ).to("cuda:0")
            # Enable optimizations
            pipe.enable_attention_slicing()
            self.pipe = pipe
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
            print("GPU Memory Usage:")
            print(os.system("nvidia-smi"))
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @modal.method()
    def upscale(self, image, scale_factor=2, conditioning_scale=0.6):
        import PIL
        import os
        
        # Resize image
        w, h = image.size
        w /= 2
        h /= 2
        image = image.resize((w * scale_factor, h * scale_factor))
        
        # Run upscaling
        result = self.pipe(
            prompt="",
            control_image=image,
            controlnet_conditioning_scale=conditioning_scale,
            num_inference_steps=28,
            guidance_scale=3.5,
            height=image.size[1],
            width=image.size[0]
        ).images[0]
        
        # save the IMAGE to the MODEL_DIR/upscaled folder, create if doesn't exist
        os.makedirs(f"{MODEL_DIR}/upscaled", exist_ok=True)
        result.save(f"{MODEL_DIR}/upscaled/upscaled.png")
        model_vol.commit()

        return result

web_app = FastAPI()

@app.function(
    image=image,
    concurrency_limit=1,
    allow_concurrent_inputs=1000
)
@modal.asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    # Add upscaler function
    def upscale(image, scale_factor, conditioning_scale):
        return UpscalerModel().upscale.remote(image, scale_factor, conditioning_scale)
    
    with gr.Blocks(title="Flux") as interface:
        with gr.Tab("Upscale"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(label="Input Image", type="pil", height=256, width=256)
                        scale_factor = gr.Slider(minimum=2, maximum=4, value=4, step=1, label="Scale Factor")
                        conditioning = gr.Slider(minimum=0.1, maximum=1.0, value=0.6, step=0.1, label="Conditioning Scale")
                        upscale_btn = gr.Button("Upscale")
                    with gr.Column():
                        img_output = gr.Image(label="Upscaled Result")
                
                upscale_btn.click(
                    fn=upscale,
                    inputs=[img_input, scale_factor, conditioning],
                    outputs=img_output
                )

    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )



@app.local_entrypoint()
def run():
    # download_models.remote(["jasperai/Flux.1-dev-Controlnet-Upscaler", "black-forest-labs/FLUX.1-dev"])
    fastapi_app.remote()
    


# to run upscaled UI: modal serve flux_gradio.py
# to download model, uncomment line 158 and comment 159 and run: modal run flux_upscaler.py
