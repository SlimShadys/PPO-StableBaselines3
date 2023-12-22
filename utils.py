from collections import deque
from typing import Union

import numpy as np
import torch
from torch import nn

from model import ActorCritic

def saveToFile(policy: ActorCritic, entropy_losses, pg_losses, value_losses, clip_fractions, approx_kl_divs, loss, num_timesteps, clip_range_loss, clip_range_vf, n_updates, game) -> None:
    with open(F"log_dir_{game}/output_log_{game}.txt", "a") as file:
        file.write(F"============= TIMESTEP N. {num_timesteps} =============\n")
        file.write(F"Mean episode reward (rollout): {compute_mean([ep_info['r'] for ep_info in policy.ep_info_buffer])}\n")
        file.write(F"Mean episode lenght (rollout): {compute_mean([ep_info['l'] for ep_info in policy.ep_info_buffer])}\n")
        file.write("============================================\n")
        file.write(f"Entropy loss: {np.mean(entropy_losses)}\n")
        file.write(f"Policy loss: {np.mean(pg_losses)}\n")
        file.write(f"Value loss: {np.mean(value_losses)}\n")
        file.write(f"Clip fraction: {np.mean(clip_fractions)}\n")
        file.write(f"Approximate KL Divergence: {np.mean(approx_kl_divs)}\n")
        file.write(f"Loss: {loss}\n")
        if hasattr(policy, "log_std"):
            file.write(f"Std: {torch.exp(policy.log_std).mean().item()}\n")
        file.write(f"Clip range: {clip_range_loss}\n")
        if clip_range_vf is not None:
            file.write(f"Clip range vf: {clip_range_vf}\n")
        file.write(F"Model updates: {n_updates}\n")
        file.write("======================================================\n")

# Compute mean. If array is empty, return NaN, otherwise return mean
def compute_mean(arr: Union[np.ndarray, list, deque]) -> float:
    return np.nan if len(arr) == 0 else float(np.mean(arr))

# Orthogonal initialization for weights and 0 for biases
def init_weights(module: nn.Module, gain: float = 1) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def retrieveOptimizer(model_path: str) -> str:
    # Split the string by underscores
    parts = model_path.split('_')
    # Modify the second part and remove the unnecessary part
    algorithm_info = parts[1].split('-')[0] + "-OPTIMIZER"
    # Update the parts list
    parts[1] = algorithm_info
    # Remove the unnecessary part
    parts.pop(3)
    # Join the parts back into a new string
    return '_'.join(parts) + ".pt"