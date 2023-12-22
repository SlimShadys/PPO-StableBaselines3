import os

import numpy as np
import torch
from gymnasium import spaces
from tqdm import tqdm

from stable_baselines3.common.env_util import VecEnv

from model import ActorCritic
from utils import compute_mean, saveToFile

# Instantiate the PPO Trainer
class PPOTrainer():
    def __init__(self, PPO: ActorCritic, env: VecEnv, saving_model_dir, configs) -> None:

        # Set the variables
        self.PPO = PPO
        self.env = env
        self.saving_model_dir = saving_model_dir

        # Get variables from JSON
        self.total_timesteps = configs["total_timesteps"]
        self.n_steps = configs["n_steps"]
        self.n_epochs = configs["n_epochs"]
        self.batch_size = configs["batch_size"]
        self.learning_rate = configs["learning_rate"]
        self.clip_range = configs["clip_range"]
        self.clip_range_vf = configs["clip_range_vf"]
        self.ent_coef = configs["ent_coef"]
        self.vf_coef = configs["vf_coef"]
        self.target_kl = configs["target_kl"]
        self.max_grad_norm = configs["max_grad_norm"]
        self.game = configs["game"]
        self.print_interval = configs["print_interval"]
        self.save_everything = configs["save_everything"]

    # Save function
    def save(self, num_timesteps, mean_ep_rew, save_everything = False) -> None:
        if(os.path.exists(f'{self.saving_model_dir}/{num_timesteps}') == False):
            os.makedirs(f'{self.saving_model_dir}/{num_timesteps}')
            
        torch.save(self.PPO.state_dict(), f'{self.saving_model_dir}/{num_timesteps}/{self.game}_PPO_ts-{num_timesteps}_rew-{mean_ep_rew:.3f}.pt')
        torch.save(self.PPO.optimizer.state_dict(), f'{self.saving_model_dir}/{num_timesteps}/{self.game}_PPO-OPTIMIZER_ts-{num_timesteps}.pt')

        if(save_everything):
            torch.save(self.PPO.actor.state_dict(), f'{self.saving_model_dir}/{num_timesteps}/{self.game}_ACTOR_ts-{num_timesteps}.pt')
            torch.save(self.PPO.critic.state_dict(), f'{self.saving_model_dir}/{num_timesteps}/{self.game}_CRITIC_ts-{num_timesteps}.pt')
            torch.save(self.PPO.mlp_extractor.policy_net.state_dict(), f'{self.saving_model_dir}/{num_timesteps}/{self.game}_POLICYEXT_ts-{num_timesteps}.pt')
            torch.save(self.PPO.mlp_extractor.value_net.state_dict(), f'{self.saving_model_dir}/{num_timesteps}/{self.game}_VALUEEXT_ts-{num_timesteps}.pt')
        return

    # Load function for model and optimizer
    def load(self, model_path = None, optimizer_path = None) -> None:
        self.PPO.load_state_dict(torch.load(model_path))

        if(optimizer_path is not None):
            self.PPO.optimizer.load_state_dict(torch.load(optimizer_path))
        return

    # Main training loop
    def train(self) -> None:
        # Tracking variables (must be set to 0 at the beginning)
        num_timesteps = 0
        n_updates = 0

        with tqdm(total=self.total_timesteps) as pbar:
            while num_timesteps < self.total_timesteps:

                # Logging variables
                entropy_losses = []
                pg_losses, value_losses = [], []
                clip_fractions = []

                # Collect experiences
                continue_training, num_timesteps = self.PPO.collect_experiences(self.env, n_rollout_steps=self.n_steps, num_timesteps=num_timesteps, pbar=pbar)

                # Set agent to train mode
                self.PPO.train()
                
                # Anneal learning rate linearly to 0 with num_timesteps
                current_progress_remaining = 1.0 - float(num_timesteps) / float(self.total_timesteps)
                self.PPO.optimizer.param_groups[0]['lr'] = self.learning_rate * current_progress_remaining # Learning rate
                clip_range_loss = self.clip_range # * current_progress_remaining                           # Clip range

                # train for n_epochs epochs
                for epoch in range(self.n_epochs):
                    
                    approx_kl_divs = []

                    # Do a complete pass on the rollout buffer
                    for rollout_data in self.PPO.rollout_buffer.get(self.batch_size):
                        actions = rollout_data.actions

                        # Convert discrete action from float to long
                        if isinstance(self.env.action_space, spaces.Discrete): 
                            actions = rollout_data.actions.long().flatten()

                        # Get values from last observation (vnext = v(o_next))
                        values, log_prob, entropy = self.PPO.evaluate_actions(rollout_data.observations, actions)
                        values = values.flatten()

                        # Normalize advantages only if length is > 1 (uselss if mini batchsize == 1)
                        advantages = rollout_data.advantages
                        if len(advantages) > 1:
                            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                        # Ratio between old and new policy
                        # Ratio is equal to 1 at the very first iteration (before the first update)
                        ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                        # Policy Loss
                        policy_loss_1 = ratio * advantages
                        policy_loss_2 = torch.clamp(ratio, 1 - clip_range_loss, 1 + clip_range_loss) * advantages
                        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                        # Clipped MSE
                        if self.clip_range_vf is None:
                            values_pred = values  # No clipping
                        else:
                            # Clip the difference between old and new value
                            values_pred = rollout_data.old_values + torch.clamp(values - rollout_data.old_values, -self.clip_range_vf, self.clip_range_vf)
                        
                        # Value loss using the returns from the rollout buffer
                        value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values_pred)

                        # Entropy loss
                        if entropy is None:
                            # Approximate entropy when no analytical form
                            entropy_loss = -torch.mean(-log_prob)
                        else:
                            entropy_loss = -torch.mean(entropy)

                        # Save the losses
                        pg_losses.append(policy_loss.item())
                        value_losses.append(value_loss.item())
                        entropy_losses.append(entropy_loss.item())
                        clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                        clip_fractions.append(clip_fraction)

                        # Compute final loss
                        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                        # # Calculate approximate form of reverse KL Divergence for early stopping
                        # # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                        # # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                        # # and Schulman blog: http://joschu.net/blog/kl-approx.html
                        with torch.no_grad():
                            log_ratio = log_prob - rollout_data.old_log_prob
                            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                            approx_kl_divs.append(approx_kl_div)

                        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                            continue_training = False
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                            break

                        self.PPO.optimizer.zero_grad()                                              # Clear the gradients
                        loss.backward()                                                             # Run a backward pass on the network
                        torch.nn.utils.clip_grad_norm_(self.PPO.parameters(), self.max_grad_norm)   # Clip gradients of Actor/Critic to 0.5
                        self.PPO.optimizer.step()                                                   # Run the optimization step for the parameters

                    n_updates += 1

                # Log the results every update     
                saveToFile(self.PPO, entropy_losses, pg_losses, value_losses, clip_fractions, approx_kl_divs, loss.item(), num_timesteps, clip_range_loss, self.clip_range_vf, n_updates, self.game)
                
                # We print the various stats every self.print_interval timesteps.
                if (num_timesteps % self.print_interval == 0):
                    # Print rollout metrics
                    print(F"============= TIMESTEP N. {num_timesteps} =============")
                    if len(self.PPO.ep_info_buffer) > 0 and len(self.PPO.ep_info_buffer[0]) > 0:
                        # Mean reward
                        mean_ep_rew = compute_mean([ep_info['r'] for ep_info in self.PPO.ep_info_buffer])
                        mean_ep_len = compute_mean([ep_info['l'] for ep_info in self.PPO.ep_info_buffer])
                        print(f"Mean episode reward (rollout): {mean_ep_rew:.3f}")
                        print(f"Mean episode lenght (rollout): {mean_ep_len:.3f}")
                        self.save(num_timesteps, mean_ep_rew, save_everything=self.save_everything)

                    print(f"Entropy loss: {np.mean(entropy_losses)}")
                    print(f"Policy loss: {np.mean(pg_losses)}")
                    print(f"Value loss: {np.mean(value_losses)}")
                    print(f"Clip fraction: {np.mean(clip_fractions)}")
                    print(f"Approximate KL Divergence: {np.mean(approx_kl_divs)}")
                    print(f"Loss: {loss.item()}")
                    
                    if hasattr(self.PPO, "log_std"):
                        print(f"Std: {torch.exp(self.PPO.log_std).mean().item()}")

                    print(f"Clip range: {self.clip_range}")            
                    if self.clip_range_vf is not None:
                        print(f"Clip range vf: {self.clip_range_vf}")

                    print(F"Model updates: {n_updates}")
                    print("===================================================")