from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3.common.env_util import VecEnv
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from rollout import RolloutBuffer

# MLP Extractor for extracting the latent policy and value from the state
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size: int = 64, game = 'LunarLander-v2', device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.device = device
        self.game = game

        # Define the policy and value network
        self.policy_net = nn.Sequential(*self.get_net()).to(device=device)
        self.value_net = nn.Sequential(*self.get_net()).to(device=device)

    # Based on the game, the proper network is defined
    def get_net(self):
        return [
            nn.Linear(self.input_dim, self.hidden_size, bias=True, device=self.device),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2, bias=True, device=self.device),
            nn.ReLU(),
            nn.Linear(self.hidden_size*2, self.hidden_size*2, bias=True, device=self.device),
            nn.ReLU(),
        ]
    
    # Forward pass of the MLP Extractor.
    # It returns the latent policy and the latent value taken from the MLP
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    # Forward pass of the MLP Extractor for the Actor
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    # Forward pass of the MLP Extractor for the Critic
    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

# Actual Actor-Critic model
class ActorCritic(nn.Module):
    def __init__(self, env: VecEnv, configs: Dict, device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
        super().__init__()

        # Misc variables
        self.device = device
        self.env = env
        self.game = configs['game']

        # Rollout buffer parameters
        self.buffer_size = configs['buffer_size']
        self.batch_size = configs['batch_size']
        self.gae_lambda = configs['gae_lambda']
        self.gamma = configs['gamma']
        self.n_envs = self.env.num_envs
        self.rollout_buffer = RolloutBuffer(self.env, device, self.buffer_size, self.batch_size, self.gae_lambda, self.gamma, self.n_envs, self.game)
        self._last_obs = None
        self._last_episode_starts = None
        window = 100
        self.ep_info_buffer = deque(maxlen=window)
        self.ep_success_buffer = deque(maxlen=window)

        # =========== NETWORK =========== #
        # First, let's define the network parameters
        self.input_dim = configs['input_dim']
        self.hidden_dim = configs['hidden_dim']

        # Based on the game, the proper output dimension is defined
        # CartPole = 2
        # LunarLander = 4
        self.output_dim = configs['output_dim']
            
        # Second, let's define the MLP extractor, which is a simple Multi-Layer Perceptron
        # separated for both Actor and Critic.
        self.mlp_extractor = MLP(self.input_dim, self.hidden_dim, self.game, device)

        # Finally, let's define the Actor and Critic final layers as simple Linear Layers
        # In order to define the input of the last layer, we retrieve the units from the MLP
        policy_last_layer = self.mlp_extractor.policy_net[-2] # [-1] is nn.Relu()

        # Check if policy_last_layer is an instance of nn.Linear
        if not isinstance(policy_last_layer, nn.Linear):
            raise ValueError("The last layer of policy_net is not an instance of nn.Linear!")

        # Extract the output size from the last layer and pass them to the Actor and Critic
        policy_last_layer_size = policy_last_layer.out_features
        self.actor = nn.Linear(policy_last_layer_size, self.output_dim, bias=True, device=self.device)
        self.critic = nn.Linear(policy_last_layer_size, 1, bias=True, device=self.device)
        # =============================== #

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=configs['learning_rate'], eps=1e-5)

    # Runs a forward pass on the Actor-Critic model
    # MLP -> Actor & Critic
    def forward(self, state: torch.Tensor, mode = 'train'):
        
        # Flatten the state and make sure it's on float values
        state = torch.flatten(state.float(), start_dim=1, end_dim=-1)

        # Get the latent embedding from MLP
        latent_pi = self.mlp_extractor.forward_actor(state)
        latent_vf = self.mlp_extractor.forward_critic(state)

        # Critic
        values = self.critic(latent_vf)

        # Actor
        mean_actions = self.actor(latent_pi)

        # Distribution
        distribution = Categorical(logits=mean_actions)

        # If in training mode, sample from the distribution
        # otherwise, take the argmax (test / eval mode)
        if(mode == 'train'):
            actions = distribution.sample()
        else: # mode == 'eval'
            mean_actions = torch.functional.F.softmax(mean_actions, dim=-1)
            actions = torch.argmax(mean_actions, dim = 1)

        log_prob = distribution.log_prob(actions)

        # Reshape the actions to match the action space of the environment
        actions = actions.reshape((-1, *self.env.action_space.shape))

        return actions, values, log_prob

    # Given the observations, evaluate the actions according to the current policy
    # and return the estimated value, log likelihood of taking those actions
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        # Flatten the state and make sure it's on float values
        obs = torch.flatten(obs.float(), start_dim=1, end_dim=-1)

        # Get the latent embedding from MLP
        latent_pi = self.mlp_extractor.forward_actor(obs)
        latent_vf = self.mlp_extractor.forward_critic(obs)

        # Critic
        values = self.critic(latent_vf)

        # Actor
        mean_actions = self.actor(latent_pi)

        # Distribution
        distribution = Categorical(logits=mean_actions)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    # Given the observations, retrieve the values according to the current policy
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        # Flatten the state and make sure it's on float values
        obs = torch.flatten(obs.float(), start_dim=1, end_dim=-1)
            
        # Get the latent embedding only from the critic
        features = self.mlp_extractor.forward_critic(obs)
        
        # Critic preds
        value_pred = self.critic(features)

        return value_pred

    # Convert the numpy observation to a PyTorch tensor
    def tensor_from_obs(self, observation) -> torch.Tensor:
        observation = np.array(observation)
        return torch.as_tensor(observation, device=self.device)

    # Needed because of Monitor wrapper from SB3
    def update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    # Collect experiences using the Rollout Buffer class instantiated in the constructor
    def collect_experiences(self, env: VecEnv, n_rollout_steps: int, num_timesteps: int, pbar: Optional[tqdm] = None) -> Tuple[bool, int]:
        # Switch to eval mode (this affects batch norm / dropout)
        self.eval()
        
        # Reset the environment if needed
        if(self._last_obs is None):
            self._last_obs = env.reset()
            self._last_episode_starts = np.ones((1,), dtype=bool)

        # Reset the buffer and n_steps
        self.rollout_buffer.reset()
        n_steps = 0

        # Main loop
        while n_steps < n_rollout_steps:
            with torch.no_grad():
                obs_tensor = self.tensor_from_obs(self._last_obs)
                actions, values, log_probs = self.forward(obs_tensor, mode = 'train') # Trainig mode during rollout

            # Perform action in the environment
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            # Update the info buffer and variables
            self.update_info_buffer(infos)
            n_steps += 1
            num_timesteps += 1 # This is needed for logging the timesteps into the print and for annealing LR / clipping, nothing else

            # Reshape the actions to match the action space of the environment
            actions = actions.reshape(-1, 1)

            # Handle resetting of the env (done/terminated) by approximating with value function
            # We take rewards/infos[0] because we only have one env
            if (dones
                and infos[0].get("terminal_observation") is not None
                and infos[0].get("TimeLimit.truncated", False)):
                terminal_obs = self.tensor_from_obs(infos[0]["terminal_observation"])
                with torch.no_grad():
                    terminal_obs = terminal_obs.unsqueeze(0)
                    terminal_value = self.predict_values(terminal_obs)
                rewards[0] += self.gamma * terminal_value 

            # Fill the buffer
            self.rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            
            # o = o_next
            # d = d_next
            self._last_obs = new_obs
            self._last_episode_starts = dones

            # Update the progress bar
            pbar.update(1)

        with torch.no_grad():
            # Compute value for the last step
            values_next = self.predict_values(self.tensor_from_obs(new_obs))

        # Compute GAE(λ)
        self.rollout_buffer.GAE(last_values=values_next, dones=dones)

        return True, num_timesteps