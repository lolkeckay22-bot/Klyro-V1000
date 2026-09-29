import os
import torch
import glob
from typing import Optional, Tuple

def save_checkpoint(model, optimizer, scheduler, step, config, checkpoint_dir="./checkpoints"):
    """
    Save model, optimizer, scheduler, and training step.
    Returns the path to the saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "config": config,
    }, checkpoint_path)
    # Also save as latest
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "config": config,
    }, latest_path)
    return checkpoint_path

def load_latest_checkpoint(checkpoint_dir="./checkpoints", model=None, optimizer=None, scheduler=None):
    """
    Load the latest checkpoint if exists.
    Returns (model, optimizer, scheduler, step) if successful, else None.
    """
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    if not os.path.exists(latest_path):
        return None
    checkpoint = torch.load(latest_path, map_location="cpu")
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    step = checkpoint.get("step", 0)
    config = checkpoint.get("config", {})
    if model is not None:
        return model, optimizer, scheduler, step, config
    else:
        return step, config

def get_all_checkpoints(checkpoint_dir="./checkpoints"):
    """
    Return list of checkpoint files sorted by step ascending.
    """
    pattern = os.path.join(checkpoint_dir, "checkpoint_step_*.pt")
    files = glob.glob(pattern)
    def extract_step(f):
        try:
            return int(os.path.basename(f).split("_")[-1].split(".")[0])
        except:
            return -1
    files.sort(key=extract_step)
    return files