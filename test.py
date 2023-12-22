import json

import numpy as np
import rlgym
import torch
from stable_baselines3.common.env_util import make_vec_env

from model import ActorCritic

if __name__ == '__main__':

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

    # Load model
    if(game == 'CartPole-v1'):
        model_path = F"models/{game}/131072/{game}_PPO_ts-131072_rew-1000.000.pt"
    elif(game == 'LunarLander-v2'):
        model_path = F"models/{game}/417792/{game}_PPO_ts-417792_rew-255.762.pt"
    else:
        raise Exception("Game not supported!")

    # Load configurations automatically based on the game
    with open(F"configs/{game}.json", 'r') as j:
        configs = json.loads(j.read())

    # Create the environment
    env_test_args = configs["environmentConfigs"]
    env_test_args["render_mode"] = 'human'     
    env_test = make_vec_env(game, env_kwargs=env_test_args, n_envs=1)#, monitor_dir=F"log_dir_{game}-test/")

    n_obs = env_test.observation_space.shape
    configs['input_dim'] = n_obs[0]

    # Adjust buffer size dynamically based on n_envs
    if("buffer_size" not in configs):
        buffer_size = configs["n_steps"] * configs["n_envs"]
        configs['buffer_size'] = buffer_size

    n_actions = env_test.action_space.n
    configs['output_dim'] = n_actions

    for key, value in env_test_args.items():
        print(F"- {key}: {value}")
    print("===================================================")
    print(F"Number of actions: {n_actions} | Class: {env_test.action_space.__class__}")
    print(F"Observation space: {n_obs} | Class: {env_test.observation_space.__class__}")
    print("===================================================")

    PPO_test = ActorCritic(env=env_test, configs=configs, device=device).to(device)
    PPO_test.load_state_dict(torch.load(model_path))

    print(PPO_test)

    PPO_test.eval()

    # Reset the environment
    obs = env_test.reset()

    # Eval variables
    MAX_EPISODES = 3
    rewards = []

    # Evaluation loop
    for i in range(MAX_EPISODES):
        done = False
        while not done:
            with torch.no_grad():
                obs_tensor = PPO_test.tensor_from_obs(obs)
                actions, values, log_probs = PPO_test.forward(obs_tensor, mode='test')
            actions = actions.cpu().numpy()
            obs, reward, done, _ = env_test.step(actions)

            rewards.append(reward)

    # Close the environment
    env_test.close()
    
    # Print the results
    print("Total reward: {:.2f}".format(np.sum(rewards) / MAX_EPISODES))