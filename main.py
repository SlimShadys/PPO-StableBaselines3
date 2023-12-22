import json
import os
from functools import partial

import numpy as np
import torch
# Use stable-baselines3 for wrapping the Environment around SB3 (for logging)
from stable_baselines3.common.env_util import make_vec_env

from model import ActorCritic
from trainer import PPOTrainer
from utils import init_weights, retrieveOptimizer

if __name__ == '__main__':

    # Print device information
    print("===================================================")
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        print('Cuda available: {}'.format(torch.cuda.is_available()))
        print("GPU: " + torch.cuda.get_device_name(torch.cuda.current_device()))
        print("Total memory: {:.1f} GB".format((float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)))))
    else:
        device = torch.device("cpu")
        print('Cuda not available, so using CPU. Please consider switching to a GPU runtime before running torche notebook!')
    print("===================================================")

    # Game settings
    game = "LunarLander-v2" # "CartPole-v1" / "LunarLander-v2"

    # Load configurations automatically based on the game
    with open(F"configs/{game}.json", 'r') as j:
        configs = json.loads(j.read())

    # Save model directory
    saving_model_dir = F'models/{game}'
    if(os.path.exists(saving_model_dir) == False):
        os.makedirs(saving_model_dir)

    # Use make_vec_env since it already includes DummyVecEnv and VecMonitor
    env_args = configs["environmentConfigs"]
    env = make_vec_env(game, env_kwargs=env_args, n_envs=1, monitor_dir=F"log_dir_{game}/")
    configs["n_envs"] = 1

    # Adjust input_dim dynamically based on the observation space
    n_obs = env.observation_space.shape
    configs['input_dim'] = n_obs[0]

    # Adjust buffer size dynamically based on n_envs
    if("buffer_size" not in configs):
        buffer_size = configs["n_steps"] * configs["n_envs"]
        configs['buffer_size'] = buffer_size

    # Adjust output_dim dynamically based on the action space
    # Needed for the output of Actor Hidden Layer
    n_actions = env.action_space.n
    configs['output_dim'] = n_actions

    print(F"{game} environment created:")
    for key, value in env_args.items():
        print(F"- {key}: {value}")
    print("===================================================")
    print(F"Number of actions: {n_actions} | Class: {env.action_space.__class__}")
    print(F"Observation space: {n_obs} | Class: {env.observation_space.__class__}")
    print("===================================================")

    # Instantiate the ActorCritic model
    PPO = ActorCritic(env=env, configs=configs, device=device).to(device)
    
    # Orthogonal initialization
    module_gains = {
        PPO.mlp_extractor.policy_net: np.sqrt(2),
        PPO.mlp_extractor.value_net: np.sqrt(2),
        PPO.actor: 0.01,
        PPO.critic: 1,
    }
    for module, gain in module_gains.items():
        module.apply(partial(init_weights, gain=gain))

    # Print the model summary
    print(PPO)
    print("===================================================")

    # Instantiate the Trainer
    trainer = PPOTrainer(PPO, env, saving_model_dir, configs)

    # Load model directory
    loadModel = False
    if(loadModel):
        model_path = ...
        optimizer_path = retrieveOptimizer(model_path)
        trainer.load(model_path=model_path, optimizer_path=optimizer_path)
        print("Model & Optimizer loaded successfully!")
        print("===================================================")

    # Train the model
    trainer.train()