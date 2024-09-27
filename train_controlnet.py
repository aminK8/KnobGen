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
# from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as T


from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers import ControlNetModel



from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel


from knobgen.utils import instantiate_from_config
from knobgen.utils import load_checkpoint, save_checkpoint
from knobgen.diff_pipeline.pipeline_stable_diffusion_controlnet import StableDiffusionFixControlNetPipeline

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


def main(
    name: str,
    use_wandb: bool,
    launcher: str,
    config: dict
    ):
    
    is_debug = config.train.is_debug
    
    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher)
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
    controlnet = ControlNetModel.from_pretrained(config.train.pretrained_controlnet_sketch, use_safetensors=True)
    vision_condition = instantiate_from_config(config.model)
    
    # Get the training dataset
    train_dataset = instantiate_from_config(config.dataset.train)
    valid_dataset = instantiate_from_config(config.dataset.validation)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=local_rank,
        shuffle=True,
        seed=config.train.global_seed,
    )
    
    # DataLoaders creation:
    train_dataloader = DataLoader(train_dataset,
                                  sampler=distributed_sampler,
                                  num_workers=config.train.num_workers,
                                  batch_size=config.train.train_batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=True)
    
    valid_dataloader = DataLoader(valid_dataset,
                                  num_workers=config.train.num_workers,
                                  batch_size=config.train.valid_batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False)
    
    # Get the training iteration
    max_train_steps = config.train.max_train_steps
    max_train_epoch = config.train.max_train_epoch
    
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    checkpointing_steps = config.train.checkpointing_steps
    checkpointing_epochs = config.train.checkpointing_epochs
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)
    
    trainable_params = list(vision_condition.parameters())
    
    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)
    unet.to(local_rank)
    vision_condition.to(local_rank)
    image_encoder.to(local_rank)
    controlnet.to(local_rank)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.optimize.learning_rate,
        betas=(config.optimize.adam_beta1, config.optimize.adam_beta2),
        weight_decay=config.optimize.adam_weight_decay,
        eps=config.optimize.adam_epsilon,
    )
    
    # Scheduler
    lr_scheduler = get_scheduler(
        config.optimize.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optimize.lr_warmup_steps * config.optimize.gradient_accumulation_steps,
        num_training_steps=max_train_steps * config.optimize.gradient_accumulation_steps,
    )
    
    if is_main_process:
        logging.info(f"trainable params number: {len(trainable_params)}")
        logging.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")
    
    # Load pretrained unet weights
    vision_condition, optimizer, lr_scheduler, _, start_epoch, _ = load_checkpoint(vision_condition,
                                                                                   optimizer,
                                                                                   lr_scheduler,
                                                                                   config.train.checkpoint_path,
                                                                                   logging,
                                                                                   is_main_process)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    vision_condition.requires_grad_(True)

    # Enable xformers
    if config.train.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if config.train.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        
    
    learning_rate = config.optimize.learning_rate
    if config.train.scale_lr:
        learning_rate = (learning_rate * config.optimize.radient_accumulation_steps * config.train.train_batch_size * num_processes)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.optimize.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = config.train.train_batch_size * num_processes * config.optimize.gradient_accumulation_steps
    
    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {config.train.train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {config.optimize.gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = start_epoch * len(train_dataloader)
    first_epoch = start_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")
    
    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if config.train.mixed_precision_training else None
    
    vision_condition = DDP(vision_condition, device_ids=[local_rank], output_device=local_rank)
    
    for epoch in range(first_epoch, num_train_epochs):
        vae.eval()
        text_encoder.eval()
        image_encoder.eval()
        unet.eval()
        controlnet.eval()
        vision_condition.train()
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
            
        epoch_loss = 0.0 
        num_batches = len(train_dataloader)
        rand_temp = tanh_scheduler(epoch, num_train_epochs - 500)
        
        for step, batch in enumerate(train_dataloader):   
            # Data batch sanity check
            if epoch == first_epoch and step == 0 and is_main_process:
                target_imgs, texts, condition_images = batch['target_image'].cpu(), batch['prompt'], batch['condition_images']
                condition_images = condition_images.squeeze(1)
                condition_images = condition_images[:, 0, :, :]
                for idx, (target_img, text, cond_img) in enumerate(zip(target_imgs, texts, condition_images)):
                    target_img = target_img / 2. + 0.5
                    torchvision.utils.save_image(target_img, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{local_rank}-{idx}'}.jpg")
                    torchvision.utils.save_image(cond_img, f"{output_dir}/sanity_check/cond_{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{local_rank}-{idx}'}.jpg")
            
            ### >>>> Training >>>> ###
            
            # Convert videos to latent space            
            target_image = batch["target_image"].to(local_rank)
            with torch.no_grad():
                latents = vae.encode(target_image).latent_dist
                latents = latents.sample()
                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # 7.2 Create tensor stating which controlnets to keep
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < 0.0 or (i + 1) / len(timesteps) > 1.0)
                ]
                controlnet_keep.append(keeps[0])

            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['prompt'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]
                
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=config.train.mixed_precision_training):
                condition_images = batch['condition_images'].to(local_rank).squeeze(1)
                resize = T.Resize((224, 224))
                condition_images_resized = resize(condition_images)

                # encoded_condition_image = image_encoder(pixel_values=condition_images_resized).pooler_output.unsqueeze(1)
                encoded_condition_image = image_encoder(pixel_values=condition_images_resized).last_hidden_state[:, 1:, :]

                # condition_images = condition_images[:, 0, :, :].unsqueeze(1)
                vision_language_coarse_grained = vision_condition(encoded_condition_image=encoded_condition_image, 
                                                                  encoder_hidden_states=encoder_hidden_states)
                
                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(1, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = 1
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=vision_language_coarse_grained,
                    controlnet_cond=condition_images,
                    conditioning_scale=cond_scale,
                    guess_mode=False,
                    return_dict=False,
                )

                if config.train.random_contion:
                    for ind_, _ in enumerate(down_block_res_samples):
                        down_block_res_samples[ind_] = down_block_res_samples[ind_] * rand_temp
                    mid_block_res_sample = mid_block_res_sample * rand_temp

                model_pred = unet(sample=noisy_latents, 
                                  timestep=timesteps, 
                                  encoder_hidden_states=vision_language_coarse_grained,
                                  down_block_additional_residuals=down_block_res_samples,
                                  mid_block_additional_residual=mid_block_res_sample).sample
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            optimizer.zero_grad()
            
            # Backpropagate
            if config.train.mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vision_condition.parameters(), config.train.max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(vision_condition.parameters(), config.train.max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            if is_main_process:
                progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
            
            epoch_loss += loss.item()
            
            # if is_main_process:
            #     for name, param in vision_condition.named_parameters():
            #         if param.requires_grad:
            #             if param.grad is not None:
            #                 print(f"Gradient for {name}: {param.grad.norm().item()}")
            #             else:
            #                 print(f"No gradient for {name}")
                        
            
            # print("---------------------------------")
            # print("condition_models.0.conv_in.bias")
            # underlying_model = vision_condition.module
            # bias_value = underlying_model.condition_models[0].conv_in.bias
            # print(bias_value)
            # print("---------------------------------")
            
            # logging.info GPU memory usage
            if step % 1000 == 0 and is_main_process:  # Adjust the frequency as needed
                logging.info(f"Epoch: {epoch}, Step: {step}, Allocated GPU memory: {torch.cuda.memory_allocated(local_rank)/1024**2:.2f} MB, Reserved GPU memory: {torch.cuda.memory_reserved(local_rank)/1024**2:.2f} MB")
            
            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == num_train_epochs * len(train_dataloader) - 1):
                save_checkpoint(vision_condition, optimizer, lr_scheduler,
                                output_dir, epoch, global_step, step,
                                train_dataloader, logging)
            # Periodically validation
            if is_main_process and (global_step % config.train.validation_steps == 0 or global_step in config.train.validation_steps_tuple):
                logging.info("Validation is started")
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(config.train.global_seed)
                resolution = config.dataset.validation.params.resolution
                height = resolution[0] if not isinstance(resolution, int) else resolution
                width  = resolution[1] if not isinstance(resolution, int) else resolution
                
                # Validation pipeline
                validation_pipeline = StableDiffusionFixControlNetPipeline.from_pretrained(
                    config.train.pretrained_model_path,
                    controlnet=controlnet,
                ).to(device)
                validation_pipeline.enable_vae_slicing()
                validation_pipeline.vision_condition = vision_condition
                validation_pipeline.image_encoder = image_encoder
                logging.info(f"Now the rand_temp is {rand_temp}")
                for step_val, batch_val in enumerate(valid_dataloader):
                    condition_images = batch_val['condition_images'].to(local_rank).squeeze(1)
                    resize = T.Resize((224, 224))
                    condition_images_resized = resize(condition_images)
                    # condition_images = condition_images[:, 0, :, :].unsqueeze(1)
                    prompts = batch_val['prompt']
                    for idx, prompt in enumerate(prompts):
                        logging.info(prompt)
                        sample = validation_pipeline(
                            prompt,
                            image               = condition_images,
                            vision_encoder_img  = condition_images_resized,
                            generator           = generator,
                            height              = height,
                            width               = width,
                            num_inference_steps = config.dataset.validation.num_inference_steps,
                            guidance_scale      = config.dataset.validation.guidance_scale,
                            rand_temp = 1.0
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        combined_images = [sample.cpu()]
                        sample = validation_pipeline(
                            prompt,
                            image               = condition_images,
                            vision_encoder_img  = condition_images_resized,
                            generator           = generator,
                            height              = height,
                            width               = width,
                            num_inference_steps = config.dataset.validation.num_inference_steps,
                            guidance_scale      = config.dataset.validation.guidance_scale,
                            rand_temp = rand_temp
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        combined_images.append(sample.cpu())
                                
                        # condition_image_rgb = condition_images[idx].repeat(3, 1, 1).cpu()
                        condition_image_rgb = condition_images[idx].cpu()
                        combined_images.append(condition_image_rgb)

                        # Stack and save the combined images
                        combined_images = torch.stack(combined_images)
                        directory = f"{output_dir}/samples/sample-{global_step}"
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        save_path = directory + f"/prompt_{'-'.join(prompt.replace('/', '').split()[:10]) if not prompt == '' else f'{local_rank}-{step_val}'}.png"
                        torchvision.utils.save_image(combined_images, save_path, nrow=len(combined_images))
                        logging.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if is_main_process:
                progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
        
        if is_main_process:
            epoch_loss /= num_batches
            if (not is_debug) and use_wandb:
                wandb.log({"epoch_loss": epoch_loss}, step=epoch)
            logging.info(f"Epoch {epoch} loss: {epoch_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/multigen20.yaml')
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, config=config)
