import modal

modal_volume = modal.Volume.from_name("evaluation-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness && cd lm-evaluation-harness && pip install -e .",
    )
)
MOUNT_DIR = "/root/lm-eval-results"

app = modal.App(
    name="lm-evaluation-harness",
    image=image,
    volumes={MOUNT_DIR: modal_volume},
)


@app.function(
    gpu="A100", timeout=7200, secrets=[modal.Secret.from_name("my-huggingface-secret")]
)
def evaluate(pretrained_models: list):
    """
    Evaluates a list of pretrained models using the lm-evaluation-harness.

    This function clones the lm-evaluation-harness repository, installs the necessary dependencies,
    and runs the evaluation for each model in the provided list. The results are stored in the specified
    output directory.

    Args:
        pretrained_models (list): A list of pretrained model identifiers to be evaluated.function

    Raises:
        FileNotFoundError: If the lm_eval command is not found.
    """
    import os
    import subprocess

    os.chdir("/lm-evaluation-harness")
    print(f"Current working directory: {os.getcwd()}")
    # os.system("huggingface-cli login --token $HF_TOKEN")
    for model in pretrained_models:
        process = subprocess.Popen(
            [
                "lm_eval",
                "--model",
                "hf",
                "--model_args",
                f"pretrained={model},trust_remote_code=True",
                "--tasks",
                "medmcqa,medqa_4options,mmlu_anatomy,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_medicine,mmlu_medical_genetics,mmlu_professional_medicine,pubmedqa",
                "--device",
                "cuda:0",
                "--batch_size",
                "16",
                "--log_samples",
                "--output_path",
                "/root/lm-eval-results",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Send 'Y' to the subprocess if it asks for confirmation
        stdout, stderr = process.communicate(input="Y\n")

        print(f"Output for model {model}:\n{stdout}")
        print(f"Error for model {model}:\n{stderr}")

        modal_volume.commit()


@app.local_entrypoint()
def main():
    """
    Main entry point for the evaluation script.

    This function defines a list of pretrained models to be evaluated and calls the evaluate function
    to perform the evaluation.

    Args:
        None
    """
    pretrained_models = ["meta-llama/Llama-3.2-1B"]
    evaluate.remote(pretrained_models)


# To run this:
# - Make sure you have modal api-key configured and huggingface token configured in modal secrets.
# - Run `modal run evaluation.py`
# - You can check the results in the `evaluation-results` volume
# - If you want to download the results, you can use the `modal volume get evaluation-results /` command and it will download the entire voulme content to your local machine.
