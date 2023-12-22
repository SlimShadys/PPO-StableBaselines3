from typing import Generator, NamedTuple, Optional

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.env_util import VecEnv

# Helper class for retrieving a batch of experiences from the Buffer
class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

class RolloutBuffer:
    def __init__(self, env: VecEnv, device, buffer_size: int = 2048, batch_size=64, gae_lambda: float = 0.95, gamma: float = 0.99, n_envs: int = 1, game = "LunarLander-v2"):

        # Variables for Rollout
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        self.env = env
        self.pos = 0
        self.full = False
        self.device = device

        # Buffers
        self.observations: np.ndarray
        self.actions: np.ndarray
        self.rewards: np.ndarray
        self.advantages: np.ndarray
        self.returns: np.ndarray
        self.episode_starts: np.ndarray
        self.log_probs: np.ndarray
        self.values: np.ndarray

        # Miscellanous
        self.game = game
        self.action_dim = self.get_action_dim(self.env.action_space)

    # Get size of the buffer
    # If not full, retrieve the current position
    def size(self) -> int:
        if self.full:
            return self.buffer_size
        return self.pos

    # Reset the buffer to Numpy arrays of zeros
    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.full = False
        self.pos = 0

    # Compute the returns and advantages based on the last values and dones
    # Actually, compute the advantages using GAE(λ)
    # Advantages based on GAE(λ)
    #   - If λ = 1, then it is Monte-Carlo estimate
    #   - If λ = 0, then it is 1-step estimate with bootstrapping
    # Returns based on TD(λ)
    def GAE(self, last_values: torch.Tensor, dones: np.ndarray):
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()
        
        # Advantages
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # Returns
        self.returns = self.advantages + self.values

    # Get the action dimension based on the action space
    # Needed for the output of Actor Hidden Layer
    def get_action_dim(action_space: spaces.Space) -> int:
        if isinstance(action_space, spaces.Box):
            return int(np.prod(action_space.shape))
        elif isinstance(action_space, spaces.Discrete):
            # Action is an int
            return 1
        elif isinstance(action_space, spaces.MultiDiscrete):
            # Number of discrete actions
            return int(len(action_space.nvec))
        elif isinstance(action_space, spaces.MultiBinary):
            # Number of binary actions
            assert isinstance(
                action_space.n, int
            ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
            return int(action_space.n)
        else:
            raise NotImplementedError(f"{action_space} action space is not supported")
    
    # Converts an array from:
    # [n_steps, n_envs, ...]
    # to:
    # [n_steps * n_envs, ...]
    def squeeze_array(self, arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    # Convert a numpy array to a PyTorch tensor
    def to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, device=self.device)

    # Add a new experience from the Buffer
    # This gets called from collect_experiences in model.py
    def add(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, episode_start: np.ndarray, value: torch.Tensor, log_prob: torch.Tensor) -> None:

        # If len of log_prob is 0, reshape it to avoid error
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        # Reshape both observation and action arrays,
        # handling correcly Discrete/MultiDiscrete obs/action space.
        if isinstance(self.env.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
        action = action.reshape((self.n_envs, self.action_dim))

        # Add the new experience to the buffer and increment the position counter
        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1

        # If the buffer is full, set the flag to True
        if self.pos == self.buffer_size:
            self.full = True

    # Get a generator that returns a batch of experiences (of shape (batch_size,)) from the Buffer
    # This gets called from collect_experiences in model.py
    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare torche data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.squeeze_array(self.__dict__[tensor])
            self.generator_ready = True

        # If batch_size is None, set it to buffer_size * n_envs
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))