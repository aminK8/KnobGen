import torch.distributed as dist

import torch
import importlib
import numpy as np
from conffusion import conditions
import os


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)

    

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_condition_key_mappings():
    cond2key = {cond: f'control_{cond}' if cond != 'segbase' else 'control_seg'
                for cond in conditions}
    key2cond = {v: k for k, v in cond2key.items()}
    return cond2key, key2cond


def load_checkpoint(model, optimizer, scheduler, checkpoint_path,
                    logging, is_main_process, loras=None, map_location="cpu"):
    if checkpoint_path is None or checkpoint_path == '' or not os.path.exists(checkpoint_path):
        logging.info(f"Checkpoint does not exist {checkpoint_path}")
        return model, optimizer, scheduler, loras, 0, 0

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)

    adjusted_state_dict = {}
    for key in checkpoint['state_dict']:
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        adjusted_state_dict[new_key] = checkpoint['state_dict'][key]

    m, u = model.load_state_dict(adjusted_state_dict, strict=False)

    if 'optimizer' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scheduler' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if 'loras' in checkpoint and loras is not None:
        lm, lu = loras.load_state_dict(checkpoint['loras'])
        if is_main_process:
            logging.info(f"Loras is loaded, Model loading - missing keys: {len(lm)}, unexpected keys: {len(lu)}")
    else:
        if is_main_process:
            logging.info(f"Loras is not loaded, not available")

    if is_main_process:
        logging.info(f"Checkpoint is loaded from {checkpoint_path} - Model loading - missing keys: {len(m)}, unexpected keys: {len(u)}")
    assert len(u) == 0
    if is_main_process:
        logging.info(f"Checkpoint is loaded from {checkpoint_path} with epoch: {epoch} and global step: {global_step}")
    return model, optimizer, scheduler, loras, epoch + 1, global_step
    
    
def print_model_keys(model, logging):
    model_keys = list(model.state_dict().keys())
    logging.info(f"Keys in model: {model_keys}")
    return model_keys

    
def save_checkpoint(model, optimizer, scheduler, output_dir, epoch,
                    global_step, step, train_dataloader, logging, loras = None):
    save_path = os.path.join(output_dir, "checkpoints")
    os.makedirs(save_path, exist_ok=True)

    state_dict = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if loras is not None:
        state_dict['loras'] = loras.state_dict()

    if step == len(train_dataloader) - 1:
        checkpoint_path = os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt")
    else:
        checkpoint_path = os.path.join(save_path, "checkpoint.ckpt")

    torch.save(state_dict, checkpoint_path)
    logging.info(f"Saved state to {checkpoint_path} (global_step: {global_step})")
