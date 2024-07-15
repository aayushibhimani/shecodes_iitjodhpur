import os
import subprocess
import torch
import gc
from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline
from huggingface_hub import login

def install_dependencies():
    print("Installing dependencies...")
    try:
        subprocess.run(["pip", "install", "-q", "-U", "git+https://github.com/huggingface/diffusers"], check=True)
        subprocess.run(["pip", "install", "-q", "-U", "transformers", "accelerate", "wandb", "bitsandbytes", "peft"], check=True)
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")

def huggingface_login(api_key):
    print("Logging in to Hugging Face...")
    try:
        login(token=api_key)
        print("Logged in to Hugging Face.")
    except Exception as e:
        print(f"Error logging in to Hugging Face: {e}")

def clone_diffusers_repo():
    print("Cloning Diffusers repository...")
    try:
        repo_dir = "diffusers"
        if os.path.exists(repo_dir):
            print(f"Repository directory {repo_dir} already exists. Removing it to clone afresh.")
            subprocess.run(["rm", "-rf", repo_dir], check=True)
        
        subprocess.run(["git", "clone", "https://github.com/huggingface/diffusers"], check=True)
        os.chdir("diffusers/examples/research_projects/sd3_lora_colab")
        print(f"Changed directory to: {os.getcwd()}")
        print("Diffusers repository cloned and working directory changed.")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning Diffusers repository: {e}")
    except FileNotFoundError as e:
        print(f"Error changing directory: {e}")

def download_instance_data():
    print("Downloading instance data...")
    try:
        local_dir = "./outfits"
        snapshot_download(
            "ritutweets46/outfits",
            local_dir=local_dir, repo_type="dataset",
            ignore_patterns=".gitattributes",
        )
        subprocess.run(["rm", "-rf", "outfits/.huggingface"], check=True)
        print("Instance data downloaded and cleaned.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading instance data: {e}")

def compute_embeddings():
    print("Computing embeddings...")
    try:
        subprocess.run(["python", "compute_embeddings.py"], check=True)
        print("Embeddings computed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error computing embeddings: {e}")

def flush_memory():
    print("Flushing memory...")
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleared.")

def train_model():
    print("Training model...")
    try:
        # Ensure train.py is executable in the home directory
        home_dir = os.path.expanduser("~")
        script_path = os.path.join(home_dir, "train.py")
        if not os.access(script_path, os.X_OK):
            print(f"Making {script_path} executable")
            subprocess.run(["chmod", "+x", script_path], check=True)
        else:
            print(f"{script_path} is already executable")

        result = subprocess.run([
            "accelerate", "launch", script_path,
            "--pretrained_model_name_or_path=stabilityai/stable-diffusion-3-medium-diffusers",
            "--instance_data_dir=outfits",
            "--data_df_path=sample_embeddings.parquet",
            "--output_dir=trained-sd3-lora-miniature",
            "--mixed_precision=fp16",
            "--instance_prompt=photos of trendy genz outfits",
            "--resolution=1024",
            "--train_batch_size=1",
            "--gradient_accumulation_steps=4",
            "--gradient_checkpointing",
            "--use_8bit_adam",
            "--learning_rate=1e-4",
            "--report_to=wandb",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            "--max_train_steps=10",
            "--seed=0",
            "--checkpointing_steps=100",
            "--push_to_hub",  # Added this line to enable pushing to Hub
            "--hub_model_id=ritutweets46/sd3_finetuned_shecodes",  # Replace with your repository ID
            #"--hub_token=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"  # Added this line to specify checkpointing steps
        ], check=True, capture_output=True, text=True)
        print("Model trained successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error training model: {e}")
        # Print detailed error information
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")

def run_inference():
    print("Running inference...")
    try:
        if not os.path.exists("trained-sd3-lora-miniature"):
            raise FileNotFoundError("The directory 'trained-sd3-lora-miniature' does not exist. Ensure the training step completed successfully.")

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16
        )
        lora_output_path = "trained-sd3-lora-miniature"
        pipeline.load_lora_weights(lora_output_path)
        pipeline.enable_sequential_cpu_offload()
        image = pipeline("Beige-coloured & black regular wrap top, Animal printed, V-neck, three-quarter,regular sleeves").images[0]
        image.save("animal_print.png")
        image2 = pipeline("Beige-coloured & black regular wrap top, Animal printed, V-neck, three-quarter,regular sleeves").images[0]
        image3 = pipeline("Beige-coloured & black regular wrap top, Animal printed, V-neck, three-quarter,regular sleeves").images[0]
        image4 = pipeline("Beige-coloured & black regular wrap top, Animal printed, V-neck, three-quarter,regular sleeves").images[0]
        image5 = pipeline("Beige-coloured & black regular wrap top, Animal printed, V-neck, three-quarter,regular sleeves").images[0]
        image2.save("animal_print2.png")
        image3.save("animal_print3.png")
        image4.save("animal_print4.png")
        image5.save("animal_print5.png")
        print("Inference completed and image saved as 'animal_print.png'.")
    except Exception as e:
        print(f"Error running inference: {e}")

def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

if __name__ == "__main__":
    print("Hi")
    install_dependencies()
    
    api_key = "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"
    huggingface_login(api_key)

    check_device()
    
    # download_instance_data()
    # compute_embeddings()
    # flush_memory()
    #train_model()
    #flush_memory()
    run_inference()
