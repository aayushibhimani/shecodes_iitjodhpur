import os
import glob
import json
import gc
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Step 1: Install libraries and dependencies
os.system("pip install bitsandbytes transformers accelerate peft -q")
os.system("pip install git+https://github.com/huggingface/diffusers.git -q")
os.system("wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py")

# Step 2: Training data
local_dir = "./SDXL_train/"
os.makedirs(local_dir, exist_ok=True)

# Assuming the images are already uploaded to the SDXL_train directory

# Preview the images
def image_grid(imgs, rows, cols, resize=256):
    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

#img_paths = "./SDXL_train/*.jpg"
img_paths = "./SDXL_train/*.[jJpP][pPnN][gG]"
imgs = [Image.open(path) for path in glob.glob(img_paths)]
print("This is the len of images")
print(len(imgs))

num_imgs_to_preview = 5
image_grid(imgs[:num_imgs_to_preview], 1, num_imgs_to_preview)

# Generate custom captions with BLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to(device)

def caption_images(input_image):
    inputs = blip_processor(images=input_image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values
    generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)
    return generated_caption

#imgs_and_paths = [(path, Image.open(path)) for path in glob.glob(f"{local_dir}*.jpg")]
imgs_and_paths = [(path, Image.open(path)) for path in glob.glob(f"{local_dir}*.[jJpP][pPnN][gG]")]

caption_prefix = "a photo of TOK outfit, "
with open(f'{local_dir}metadata.jsonl', 'w') as outfile:
    for img in imgs_and_paths:
        caption = caption_prefix + caption_images(img[1]).split("\n")[0]
        entry = {"file_name": img[0].split("/")[-1], "prompt": caption}
        json.dump(entry, outfile)
        outfile.write('\n')

# Free up some memory
del blip_processor, blip_model
gc.collect()
torch.cuda.empty_cache()

# Step 3: Start training
import locale
locale.getpreferredencoding = lambda: "UTF-8"

os.system("accelerate config default")

# Setting the hyperparameters
os.system("pip install datasets -q")

training_script = """
accelerate launch train_dreambooth_lora_sdxl.py \\
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \\
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \\
  --dataset_name="SDXL_train" \\
  --output_dir="SDXL_LoRA_model" \\
  --caption_column="prompt"\\
  --mixed_precision="fp16" \\
  --instance_prompt="a photo of TOK outfit" \\
  --resolution=1024 \\
  --train_batch_size=1 \\
  --gradient_accumulation_steps=3 \\
  --gradient_checkpointing \\
  --learning_rate=1e-4 \\
  --snr_gamma=5.0 \\
  --lr_scheduler="constant" \\
  --lr_warmup_steps=0 \\
  --mixed_precision="fp16" \\
  --use_8bit_adam \\
  --max_train_steps=500 \\
  --checkpointing_steps=100 \\
  --seed="0"
"""
print("Hi second checkpoint")
os.system(training_script)

# Step 4: Inference model
from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.load_lora_weights('./SDXL_LoRA_model/pytorch_lora_weights.safetensors')
_ = pipe.to("cuda")

# Text to Image Generation
prompt = "A photo of TOK outfit, a mid length blue dress with long sleeves and floral print."
image = pipe(prompt=prompt, num_inference_steps=25).images[0]
image.save('./sdxl_output1.png')

# Image to Image Generation
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.load_lora_weights('./SDXL_LoRA_model/pytorch_lora_weights.safetensors')
_ = pipeline.to("cuda")

pipeline.enable_model_cpu_offload()

image2image = load_image('./sdxl_output1.png')
resized_image = image2image.resize((1024, 576))

#imgtoimg prompt
prompt = "A photo of TOK outfit, a mid length blue dress with long sleeves and floral print."
image_output = pipeline(prompt, image=resized_image, strength=0.5).images[0]
make_image_grid([image, image_output], rows=1, cols=2)
image_output.save('./sdxl_output2.png')

# Stable Video Diffusion (Image to Video)
os.system("pip install -q -U diffusers transformers accelerate")

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)

pipe.enable_model_cpu_offload()
pipe.unet.enable_forward_chunking()

image = load_image("./sdxl_output2.png")
resized_image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(resized_image, decode_chunk_size=2, generator=generator, num_frames=25).frames[0]

export_to_video(frames, "generated.mp4", fps=7)

# SVD with micro-conditioning
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

image1 = load_image("./sdxl_output2.png")
image1 = image1.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image1, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]
export_to_video(frames, "generated_micro_conditioning.mp4", fps=7)

# Step 5: Model Deployment on HF Spaces
""" from huggingface_hub import notebook_login, whoami, upload_folder, create_repo
from pathlib import Path
from train_dreambooth_lora_sdxl import save_model_card

notebook_login()

output_dir = "SDXL_LoRA_model"
username = whoami(token=Path("/root/.cache/huggingface/"))["name"]
repo_id = f"{username}/{output_dir}"

repo_id = create_repo(repo_id, exist_ok=True).repo_id

save_model_card(
    repo_id=repo_id,
    images=[],
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    train_text_encoder=False,
    instance_prompt="a photo of TOK home",
    validation_prompt=None,
    repo_folder=output_dir,
    vae_path="madebyollin/sdxl-vae-fp16-fix",
)

upload_folder(
    repo_id=repo_id,
    folder_path=output_dir,
    commit_message="End of training",
    ignore_patterns=["step_", "epoch_"],
)
 """