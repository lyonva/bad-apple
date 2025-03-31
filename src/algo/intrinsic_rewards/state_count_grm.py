import gymnasium as gym
from typing import Dict, Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from src.algo.intrinsic_rewards.base_model import IntrinsicRewardBaseModel
from src.algo.common_models.mlps import *
from src.utils.enum_types import NormType

# Implements a simple state counting intrinsic reward mechanism
# Reward is 1 / sqrt(N); where N is the number of times the state has been encountered
# Calculation is done on the hash of the state after taking the action
# Plus GRM D-delayed discounting
class StateCountGRMModel(IntrinsicRewardBaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 0.5,
        model_learning_rate: float = 3e-4,
        model_cnn_features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        model_cnn_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        model_features_dim: int = 256,
        model_latents_dim: int = 256,
        model_mlp_norm: NormType = NormType.BatchNorm,
        model_cnn_norm: NormType = NormType.BatchNorm,
        model_gru_norm: NormType = NormType.NoNorm,
        use_model_rnn: int = 0,
        model_mlp_layers: int = 1,
        gru_layers: int = 1,
        use_status_predictor: int = 0,
        # GRM-specific params
        n_envs: int = 1,
        gamma: int = 0.99,
        grm_delay: int = 0,
    ):
        super().__init__(observation_space, action_space, activation_fn, normalize_images,
                         optimizer_class, optimizer_kwargs, max_grad_norm, model_learning_rate,
                         model_cnn_features_extractor_class, model_cnn_features_extractor_kwargs,
                         model_features_dim, model_latents_dim, model_mlp_norm,
                         model_cnn_norm, model_gru_norm, use_model_rnn, model_mlp_layers,
                         gru_layers, use_status_predictor)
        # self._build()
        # self._init_modules()
        # self._init_optimizers()
        self.counts = dict()
        self.gamma = gamma
        self.grm_delay = grm_delay
        self.grm_buffer = np.zeros((n_envs, grm_delay))

    def _build(self) -> None:
        # No building required
        pass


    def forward(self,
        curr_obs: Tensor, next_obs: Tensor, last_mems: Tensor,
        curr_act: Tensor, curr_dones: Tensor,
        curr_key_status: Optional[Tensor],
        curr_door_status: Optional[Tensor],
        curr_target_dists: Optional[Tensor],
    ):
        pass

    def _get_hash(self, next_obs):
        # hash tensor based on https://stackoverflow.com/questions/74805446/how-to-hash-a-pytorch-tensor
        if next_obs.dtype != th.int64:
            next_obs = next_obs.to(th.int64)
        batch_size = next_obs.shape[0]
        hashes = np.zeros(batch_size, dtype=np.int64)
        for env_id in range(batch_size):
            env_obs = next_obs[env_id]
            while env_obs.ndim > 0:
                env_obs = self._reduce_last_axis(env_obs)
            hashes[env_id] = env_obs
        return hashes
    
    @th.no_grad()
    def _reduce_last_axis(self, x: Tensor) -> Tensor:
        assert x.dtype == th.int64
        acc = th.zeros_like(x[..., 0])
        for i in range(x.shape[-1]):
            acc *= 6364136223846793005
            acc += 1
            acc += x[..., i]
            # acc %= MODULUS  # Not really necessary.
        return acc


    def get_intrinsic_rewards(self,
        curr_obs, next_obs, last_mems, curr_act, curr_dones, obs_history,
        stats_logger
    ):
        batch_size = curr_obs.shape[0]
        int_rews = np.zeros(batch_size, dtype=np.float32)
        hashes = self._get_hash(next_obs)
        for env_id in range(batch_size):
            # Update historical observation embeddings
            hash = hashes[env_id]
            if hash not in self.counts:
                self.counts[hash] = 0
            self.counts[hash] += 1

            # Generate intrinsic reward
            int_rews[env_id] += 1 / np.sqrt( self.counts[hash] )
        
        # GRM discount
        undiscounted_rewards = int_rews.copy()
        n_envs = self.grm_buffer.shape[0]
        int_rews = undiscounted_rewards - self.grm_buffer[:,-1]/(self.gamma**self.grm_delay)
        # Check for episode ends and discount all others if so
        for env in range(n_envs):
            if curr_dones[env] != 0:
                # Discount all and reset
                int_rews[env] -= np.sum(self.grm_buffer[env,0:-1]/(self.gamma** np.arange(self.grm_delay-1) ))
                

        # Log rewards on buffer, rewards on timestep D are removed
        self.grm_buffer = np.roll(self.grm_buffer, 1, axis=1)
        self.grm_buffer[:,0] = undiscounted_rewards

        # Logging
        # stats_logger.add(
        #     fwd_loss=fwd_loss,
        #     key_loss=key_loss,
        #     door_loss=door_loss,
        #     pos_loss=pos_loss,
        #     key_dist=key_dist,
        #     door_dist=door_dist,
        #     goal_dist=goal_dist,
        # )
        return int_rews, None


    def optimize(self, rollout_data, stats_logger):
        pass

        # stats_logger.add(
        #     fwd_loss=fwd_loss,
        #     key_loss=key_loss,
        #     door_loss=door_loss,
        #     pos_loss=pos_loss,
        #     key_dist=key_dist,
        #     door_dist=door_dist,
        #     goal_dist=goal_dist,
        # )