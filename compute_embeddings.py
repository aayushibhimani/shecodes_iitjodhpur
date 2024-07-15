import argparse
import glob
import hashlib
import os

import pandas as pd
import torch
from transformers import T5EncoderModel
from diffusers import StableDiffusion3Pipeline

PROMPT = "photos of trendy genz outfits"
MAX_SEQ_LENGTH = 77
LOCAL_DATA_DIR = "outfits"
OUTPUT_PATH = "sample_embeddings.parquet"

def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

def generate_image_hash(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    return hashlib.sha256(img_data).hexdigest()

def load_sd3_pipeline():
    id = "stabilityai/stable-diffusion-3-medium-diffusers"
    text_encoder = T5EncoderModel.from_pretrained(id, subfolder="text_encoder_3", load_in_8bit=True, device_map="auto")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        id, text_encoder_3=text_encoder, transformer=None, vae=None, device_map="balanced"
    )
    return pipeline

@torch.no_grad()
def compute_embeddings(pipeline, prompt, caption, max_sequence_length):
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(prompt=prompt, prompt_2=caption, prompt_3=None, max_sequence_length=max_sequence_length)

    print(
        f"{prompt_embeds.shape=}, {negative_prompt_embeds.shape=}, {pooled_prompt_embeds.shape=}, {negative_pooled_prompt_embeds.shape}"
    )

    max_memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())
    print(f"Max memory allocated: {max_memory:.3f} GB")
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def run(args):
    pipeline = load_sd3_pipeline()

    image_paths = [path for path in glob.glob(f"{args.local_data_dir}/*") if path.endswith(('.jpg', '.jpeg', '.png'))]
    data = []
    
    for image_path in image_paths:
        img_hash = generate_image_hash(image_path)
        
        caption_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(caption_path):
            with open(caption_path, "r") as f:
                caption = f.read().strip()
        else:
            caption = ""

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = compute_embeddings(
            pipeline, args.prompt, caption, args.max_sequence_length
        )

        data.append(
            (img_hash, caption, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        )

    embedding_cols = [
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
    ]
    df = pd.DataFrame(
        data,
        columns=["image_hash", "caption"] + embedding_cols,
    )

    for col in embedding_cols:
        df[col] = df[col].apply(lambda x: x.cpu().numpy().flatten().tolist())

    df.to_parquet(args.output_path)
    print(f"Data successfully serialized to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=PROMPT, help="The instance prompt.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Maximum sequence length to use for computing the embeddings. The more the higher computational costs.",
    )
    parser.add_argument(
        "--local_data_dir", type=str, default=LOCAL_DATA_DIR, help="Path to the directory containing instance images."
    )
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="Path to serialize the parquet file.")
    args = parser.parse_args()

    run(args)
