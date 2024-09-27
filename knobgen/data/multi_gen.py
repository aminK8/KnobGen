import random
import numpy as np
import json
import cv2
import torch
from torch.utils.data.dataset import Dataset
from typing import List, Dict
from PIL import Image, UnidentifiedImageError
import random
from knobgen.utils import get_condition_key_mappings
from knobgen.data.data_utils import create_random_mask


class MultiGen20M(Dataset):
    def __init__(self,
                 path_json: str,
                 path_meta: str,
                 conditions: List,
                 hints: List[Dict],
                 prompt_json: str = None,
                 resolution: int = 512,
                 none_loop: int = 0,
                 p_drop_text: float = 0.3,
                 p_drop_condition: float = 0.2,
                 resize_mode='random_crop'):
        """
            MultiGen20M Dataset/Dataloader.

            :param path_json: (formatted) path to MultiGen20M json files,
                e.g. "path/to/json_files/aesthetics_plus_all_group_{}_all.json"
            :param path_meta: path to the data root folder
            :param hints: conditions considered, and paths to the condition data folders,
                e.g. [{"task": "task_key", ..., "data":{"image_file": "condition_image_file"}}]
        """
        super().__init__()

        self.path_meta = path_meta
        self.resolution = resolution
        self.none_loop = none_loop
        self.p_drop_text = p_drop_text
        self.p_drop_condition = p_drop_condition
        self.conditions = conditions
        self.resize_mode = resize_mode

        self.data = dict()  # store all {"image_file": "textual_prompt"}
        self.hints = {}  # store all hints, {"hed": {"path": "path_to_conditions", "data":{"image_file": "condition_image_file", ...}}, ...}
        self.cond2key, self.key2cond = get_condition_key_mappings()
        self.prompts = None
        
        # load llava revised prompts
        if prompt_json is not None and prompt_json is not '':
            with open(prompt_json, 'rt') as f:
                prompts = json.loads(f.read())
            self.prompts = dict()
            for fn, prompt in prompts.items():
                if 'new_prompt' in prompt:
                    self.prompts[fn] = prompt['new_prompt']
                elif 'old_prompt' in prompt:
                    self.prompts[fn] = prompt['old_prompt']

        for hint in hints:
            task = hint['task']
            this_hint = dict(path=hint['path'])
            data = dict()
            if task in self.cond2key:
                key_prompt = self.cond2key[task]
            else:
                print(f'Task {task} is not in the dataset.')

            with open(path_json.format(task), 'rt') as f:
                for line in f:
                    info = json.loads(line)
                    if self.prompts is not None and info['source'] not in self.prompts:
                        self.prompts[info['source']] = info['prompt']
                    self.data[info['source']] = info['prompt']
                    data[info['source']] = info[key_prompt]  # {"image_file": "condition_image_file"}
            
            this_hint['data'] = data
            self.hints[task]= this_hint
            print(f"*** Hint {task}: {len(this_hint['data'])} conditional images ***")
            if self.prompts is not None:
                print(f"*** Total: {len(self.prompts)} images ***")
            else: 
                print(f"*** Total: {len(self.data)} images ***")
                
        if self.prompts is not None:
            self.list_data = list(self.prompts)  # get all image file paths
        else: 
            self.list_data = list(self.data) 

        self.transform = None  # transform function
    
    def imread(self, image_path):
        try:
            img = Image.open(image_path)
            if img.mode == 'PA' or img.mode == 'P':
                img = img.convert('RGBA')
            return np.asarray(img.convert('RGB'))
        except UnidentifiedImageError:
            print(f"Cannot identify image file {image_path}")
        except OSError as e:
            print(f"Error processing file {image_path}: {e}")

    def resize_image_random_cropping(self, control_image, resolution):
        H, W, C = control_image.shape
        if W >= H:
            crop_l = (W - H) // 2 if self.resize_mode == 'center_crop' else random.randint(0, W - H)
            crop_r = crop_l + H
            crop_t = 0
            crop_b = H
        else:
            crop_t = (H - W) // 2 if self.resize_mode == 'center_crop' else random.randint(0, H - W)
            crop_b = crop_t + W
            crop_l = 0
            crop_r = W
        control_image = control_image[crop_t: crop_b, crop_l: crop_r]
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        img = cv2.resize(control_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img, [crop_t/H, crop_b/H, crop_l/W, crop_r/W]

    def resize_image_fixed_cropping(self, target_image, resolution, sizes):
        H, W, C = target_image.shape
        crop_t_rate, crop_b_rate, crop_l_rate, crop_r_rate = sizes[0], sizes[1], sizes[2], sizes[3]
        crop_t, crop_b, crop_l, crop_r = int(crop_t_rate*H), int(crop_b_rate*H), int(crop_l_rate*W), int(crop_r_rate*W)
        target_image = target_image[crop_t: crop_b, crop_l:crop_r]
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        img = cv2.resize(target_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
            The __getitem__ method to iterate over the dataset.

            Parameters
            ----------
            idx : int
                index of the current file in the dataset.

            Returns
            -------
            dict
                "target_img" : image numpy array (C, H, W), the target real image
                "prompt" : str, the textual description of this image
                "condition_images" : dict, {"condition_type": conditional image numpy array (C, H, W)}
        """

        target_filename = self.list_data[idx]
        prompt = self.data[target_filename]
        condition_images = torch.zeros((len(self.conditions), 3, self.resolution, self.resolution))
        hints = torch.zeros(len(self.conditions))
        sizes = None

        # load all condition images
        for ci, cond in enumerate(self.conditions):
            if cond in self.hints and target_filename in self.hints[cond]['data']:
                # condition image exists for the current image
                condition_filename = self.hints[cond]['data'][target_filename].replace("aesthetics_6_25_plus_", "")
                condition_img = self.imread(self.hints[cond]["path"] + condition_filename)
                if sizes is None:
                    condition_img, sizes = self.resize_image_random_cropping(condition_img, self.resolution)
                else:
                    condition_img = self.resize_image_fixed_cropping(condition_img, self.resolution, sizes)
                    
                # Normalize source images to [0, 1].
                condition_img = np.where(condition_img > 50, 255, 0).astype(np.uint8)
                condition_img = condition_img.astype(np.float32) / 255.0
                condition_img = condition_img.transpose(2, 0, 1)
                condition_img = torch.from_numpy(condition_img)
                if random.uniform(0, 1) >= self.p_drop_condition:
                    condition_images[ci] = condition_img
                    hints[ci] = True
            else:
                # empty condition image, will be handled by the diffusion pipeline
                # condition_images[ci] = torch.zeros((3, self.resolution, self.resolution))
                # hints[ci] = False
                pass

        # load the target/real image
        if "./" == target_filename[0:2]:
            target_filename = target_filename[2:]
        target_image = self.imread(self.path_meta + "/images/" + target_filename)
        # Check if target_image is None
        if target_image is None:
            # If the target image is None, return a placeholder or continue without it
            target_image = torch.zeros((3, self.resolution, self.resolution))  # Or any other default value
            print(f"Error is from {target_filename}")
        else:
            target_image = self.resize_image_fixed_cropping(target_image, self.resolution, sizes)
            # Normalize target images to [-1, 1].
            target_image = (target_image.astype(np.float32) / 127.5) - 1.0
            target_image = target_image.transpose(2, 0, 1)
            target_image = torch.from_numpy(target_image)
        
        # Text prompt, drop with a probability
        prompt = prompt if random.uniform(0, 1) > self.p_drop_text else ''

        # return a mask
        inpainting_mask = create_random_mask((self.resolution, self.resolution))

        sample = dict(target_image=target_image, prompt=prompt, condition_images=condition_images,
                      hints=hints, inpainting_mask=inpainting_mask, filename=target_filename)
        # transform function, if specified
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
