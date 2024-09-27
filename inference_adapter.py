import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess
import numpy as np


from pathlib import Path
from tqdm.auto import tqdm
# from einops import rearrange
from omegaconf import OmegaConf
# from safetensors import safe_open


import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T


from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers import T2IAdapter



from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel


from knobgen.utils import instantiate_from_config
from knobgen.utils import load_checkpoint
from knobgen.diff_pipeline.pipeline_stable_diffusion_adapter import StableDiffusionFixAdapterPipeline



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            rank = int(os.environ['RANK'])
            local_rank = rank % num_gpus
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend=backend, **kwargs)
        else:
            rank = int(os.environ['RANK'])
            dist.init_process_group(backend='gloo', **kwargs)
            return 0

    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank

def tanh_scheduler(epoch, num_epochs, min_value=0.20, max_value=1.0):
    if epoch >= num_epochs:
        return 1.0
    # Calculate progress as a fraction of the total epochs
    progress = epoch / num_epochs

    # Apply tanh to the progress (scaling it to the tanh range)
    tanh_progress = torch.tanh(torch.tensor(progress * 6 - 3))  # Scale to tanh range (-3, 3)

    # Scale tanh output from (-1, 1) to (0, 1)
    scaled_progress = (tanh_progress + 1) / 2

    # Scale to the range [min_value, max_value]
    result = min_value + scaled_progress * (max_value - min_value)

    return result.item()


def main(
    name: str,
    use_wandb: bool,
    launcher: str,
    config: dict
    ):
    
    is_debug = config.train.is_debug
    
    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher, port=29503)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0
    device = torch.device('cuda', local_rank)

    seed = config.train.global_seed + global_rank
    set_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(config.train.output_dir, folder_name)
    if is_debug and os.path.exists(output_dir) and is_main_process:
        os.system(f"rm -rf {output_dir}")
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="conffusion", name=folder_name, config=config)
        
    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
        
        
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(config.train.noise_scheduler_kwargs))
    
    vae          = AutoencoderKL.from_pretrained(config.train.pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(config.train.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.train.pretrained_model_path, subfolder="text_encoder")
    image_encoder = CLIPVisionModel.from_pretrained(config.train.pretrained_image_encoder)
    unet = UNet2DConditionModel.from_pretrained(config.train.pretrained_model_path, subfolder="unet")
    adapter = T2IAdapter.from_pretrained(config.train.pretrained_adapter_sketch)
    vision_condition = instantiate_from_config(config.model)
    
    # Get the validation dataset
    valid_dataset = instantiate_from_config(config.dataset.validation)
    
    valid_dataloader = DataLoader(valid_dataset,
                                  num_workers=config.train.num_workers,
                                  batch_size=config.train.valid_batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False)
    
    # Get the training iteration
    max_train_steps = config.train.max_train_steps
        
    checkpointing_steps = config.train.checkpointing_steps
    
    
    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)
    unet.to(local_rank)
    vision_condition.to(local_rank)
    image_encoder.to(local_rank)
    adapter.to(local_rank)

    # Load pretrained unet weights
    vision_condition, _, _, _, _, _ = load_checkpoint(vision_condition,
                                                      None,
                                                      None,
                                                      config.train.checkpoint_path,
                                                      logging,
                                                      is_main_process)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)
    adapter.requires_grad_(False)
    vision_condition.requires_grad_(False)

    # Enable xformers
    if config.train.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Instantaneous batch size per device = {config.train.train_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {config.optimize.gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")

    vae.eval()
    text_encoder.eval()
    image_encoder.eval()
    unet.eval()
    adapter.eval()
    vision_condition.train()

    logging.info("Validation is started")
    generator = torch.Generator(device=device)
    resolution = config.dataset.validation.params.resolution
    height = resolution[0] if not isinstance(resolution, int) else resolution
    width  = resolution[1] if not isinstance(resolution, int) else resolution
    
    # Validation pipeline
    validation_pipeline = StableDiffusionFixAdapterPipeline.from_pretrained(
        config.train.pretrained_model_path,
        adapter=adapter,
    ).to(device)
    validation_pipeline.enable_vae_slicing()
    validation_pipeline.vision_condition = vision_condition
    validation_pipeline.image_encoder = image_encoder
    
    for step_val, batch_val in enumerate(valid_dataloader):
        condition_images = batch_val['condition_images'].to(local_rank).squeeze(1)
        resize = T.Resize((224, 224))
        condition_images_resized = resize(condition_images)
        condition_images = condition_images[:, 0, :, :].unsqueeze(1)
        prompts = batch_val['prompt']
        for idx, prompt in enumerate(prompts):
            logging.info(prompt)
            for knob in range(10, config.dataset.validation.num_inference_steps + 2):
                combined_images = []
                for i in range(3):
                    generator.manual_seed(config.train.global_seed * i + 3)
                    sample = validation_pipeline(
                        prompt,
                        image               = condition_images,
                        vision_encoder_img  = condition_images_resized,
                        generator           = generator,
                        height              = height,
                        width               = width,
                        num_inference_steps_for_fine_graind = knob,
                        num_inference_steps = config.dataset.validation.num_inference_steps,
                        guidance_scale      = config.dataset.validation.guidance_scale,
                        rev = True
                    ).images[0]

                    sample = torchvision.transforms.functional.to_tensor(sample)
                    combined_images.append(sample.cpu())

                    
                condition_image_rgb = condition_images[idx].repeat(3, 1, 1).cpu()
                combined_images.append(condition_image_rgb)

                # Stack and save the combined images
                combined_images = torch.stack(combined_images)
                directory = f"{output_dir}/samples/sample_knob_{knob}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_path = directory + f"/prompt_{'-'.join(prompt.replace('/', '').split()[:10]) if not prompt == '' else f'{local_rank}-{step_val}'}.png"
                torchvision.utils.save_image(combined_images, save_path, nrow=len(combined_images))
                logging.info(f"Saved samples to {save_path}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/multigen20.yaml')
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, config=config)
