# Copied and modified from Stable Diffusion
# https://github.com/runwayml/stable-diffusion/tree/main

import numpy as np
import torch
import random
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader, Dataset
# from functools import partial
# from ldm.data.base import Txt2ImgIterableBaseDataset
from typing import Tuple  # Optional, Dict, Any


def biased_random_int(a: int, b: int, bias: float = 2.0) -> int:
    """
    Generate a random integer within the range [a, b] with a bias towards larger values.
    
    Parameters:
        a (int): The lower bound of the range (inclusive).
        b (int): The upper bound of the range (inclusive).
        bias (float): The bias factor. Higher values make larger integers more likely.
    
    Returns:
        int: A random integer within the range [a, b] biased towards larger values.
    """
    if a == b:
        return a
    
    # Generate an unbiased random float between 0 and 1
    u = np.random.random()
    
    # Apply a power transformation to bias towards 1
    u_biased = u ** (1 / bias)
    
    # Scale to the desired range [a, b]
    random_int = int(a + (b - a) * u_biased)
    
    return random_int


def create_random_rectangular_mask(image_size: Tuple[int, int], 
                                   min_size_fraction: float = 0.1, 
                                   max_size_fraction: float = 0.5,
                                   bias: float = 2.0) -> torch.Tensor:
    """
    Create a random rectangular mask for an image of given size.
    
    Parameters:
        image_size (tuple): The size of the image (height, width).
        min_size_fraction (float): Minimum size fraction of the rectangle side length relative to image dimensions.
        max_size_fraction (float): Maximum size fraction of the rectangle side length relative to image dimensions.
        bias (float): A bias factor. Higher values make larger rectangle side length more likely.
    
    Returns:
        torch.Tensor: A binary mask with the same size as the image.
    """
    h, w = image_size

    # Randomly select the width and height of the rectangle
    rect_w = biased_random_int(int(min_size_fraction * w), int(max_size_fraction * w), bias=bias)
    rect_h = biased_random_int(int(min_size_fraction * h), int(max_size_fraction * h), bias=bias)
    
    # Randomly select the upper left corner of the rectangle
    x_start = random.randint(0, w - rect_w)
    y_start = random.randint(0, h - rect_h)
    
    # Initialize the mask with zeros
    mask = torch.zeros((h, w), dtype=torch.bool)
    
    # Set the rectangle area to 1
    mask[y_start:y_start + rect_h, x_start:x_start + rect_w] = 1
    
    return mask


def create_random_ellipsoid_mask(image_size: Tuple[int, int], 
                                 min_size_fraction: float = 0.1, 
                                 max_size_fraction: float = 0.5,
                                 bias: float = 2.0) -> torch.Tensor:
    """
    Create a random ellipsoid mask for an image of given size.
    
    Parameters:
        image_size (tuple): The size of the image (height, width).
        min_size_fraction (float): Minimum size fraction of the ellipsoid diameter relative to image dimensions.
        max_size_fraction (float): Maximum size fraction of the ellipsoid diameter relative to image dimensions.
        bias (float): A bias factor. Higher values make larger ellipsoid diameter more likely.
    
    Returns:
        torch.Tensor: A binary mask with the same size as the image.
    """
    h, w = image_size

    # Randomly select the width and height of the rectangle
    radius_x = biased_random_int(int(min_size_fraction * w), int(max_size_fraction * w), bias=bias) // 2
    radius_y = biased_random_int(int(min_size_fraction * h), int(max_size_fraction * h), bias=bias) // 2
    
    # Randomly select the upper left corner of the rectangle
    cx = random.randint(radius_x, w - radius_x)
    cy = random.randint(radius_y, h - radius_y)
    
    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    
    # Calculate the ellipsoid equation
    norm_x = ((x - cx) / radius_x) ** 2
    norm_y = ((y - cy) / radius_y) ** 2
    mask = norm_x + norm_y <= 1
    
    return mask


def create_random_mask(image_size: Tuple[int, int], 
                       min_size_fraction: float = 0.1, 
                       max_size_fraction: float = 0.5,
                       bias: float = 2.0,
                       p_rect: float = 0.5) -> torch.Tensor:
    """
    Create a random mask for an image of given size.
    The mask shape is randomly chosen from [rectangular, ellipsoid].
    
    Parameters:
        image_size (tuple): The size of the image (height, width).
        min_size_fraction (float): Minimum size fraction of the mask lengths relative to image dimensions.
        max_size_fraction (float): Maximum size fraction of the mask lengths relative to image dimensions.
        bias (float): A bias factor. Higher values make larger mask lengths more likely.
        p_rect (float): Probability of generating a rectangular mask. Probability for ellipsoid mask is then (1 - p_rect).

    Returns:
        torch.Tensor: A binary mask with the same size as the image.
    """

    # Generate a random float between 0 and 1
    u = random.random()

    # Determine the mask shape according to the probability
    if u <= p_rect:
        mask = create_random_rectangular_mask(image_size, min_size_fraction=min_size_fraction,
                                              max_size_fraction=max_size_fraction, bias=bias)
    else:
        mask = create_random_ellipsoid_mask(image_size, min_size_fraction=min_size_fraction,
                                            max_size_fraction=max_size_fraction, bias=bias)
        
    return mask


if __name__ == '__main__':
    import time
    import random
    from omegaconf import OmegaConf
    from transformers import CLIPTextModel, CLIPTokenizer
    from knobgen.utils import instantiate_from_config

    # start_time = time.time()
    # print(f"Start: {time.time()}")
    # cfg_path = '/fs/scratch/PAS0536/Mengxi/Control-Inpaint/configs/control-inpainting-ldm/cildm-1.5.yaml'
    # config = OmegaConf.load(cfg_path)
    # print(f'Time used config: {((time.time() - start_time) / 60.0):.3f} min.')
    # data = instantiate_from_config(config.data)
    # # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # # calling these ourselves should not be necessary but it is.
    # # lightning still takes care of proper multiprocessing though
    # # data.prepare_data()  # This is unnecessary if data has been downloaded
    # data.setup()
    # print(f'Time used init MultiGen20M: {((time.time() - start_time) / 60.0):.3f} min.')
    # print("#### Data #####")
    # for k in data.datasets:
    #     print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    # print(f'Time used: {((time.time() - start_time) / 60.0):.3f} min.')

    # prepare the MultiGen20M dataset
    dataset_config_name = '/fs/scratch/PAS0536/Mengxi/Control-Inpaint/configs/data/multigen20k.yaml'
    data_cfg = OmegaConf.load(dataset_config_name)
    train_dataset = instantiate_from_config(data_cfg.train)

    tokenizer = CLIPTokenizer.from_pretrained(
        'ddpm', subfolder="tokenizer", revision=None
    )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples['prompt']:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `prompt` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        examples['input_ids'] = inputs.input_ids
        return examples

    train_dataset.transform = tokenize_captions

    # train_dataset = train_dataset.shuffle(seed=42).select(range(1000))
    # # Set the training transforms
    # train_dataset = train_dataset.with_transform(tokenize_captions)

    def collate_fn(examples):
        # target images
        target_images = torch.stack([example["target_image"] for example in examples])
        target_images = target_images.to(memory_format=torch.contiguous_format).float()
        # textual prompts
        input_ids = torch.stack([example["input_ids"] for example in examples])
        # condition
        condition_images = torch.stack([example["condition_images"] for example in examples])
        # hint identifiers
        hints = torch.stack([example["hints"] for example in examples])
        # inpainting masks
        inpainting_masks = torch.stack([example["inpainting_mask"] for example in examples])
        return {"target_images": target_images, "input_ids": input_ids, "condition_images": condition_images,
                "hints": hints, "inpainting_masks": inpainting_masks}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=2,
        num_workers=10,
    )

    for idx, data in enumerate(train_dataloader):
        if idx > 2:
            break
        target_image = np.array(data['target_images'][0]).transpose(1, 2, 0)
        target_image = ((target_image + 1) * 127.5).astype(np.uint8)
