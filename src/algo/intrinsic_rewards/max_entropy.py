import gymnasium as gym
from typing import Dict, Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from src.algo.intrinsic_rewards.base_model import IntrinsicRewardBaseModel
from src.algo.common_models.mlps import *
from src.utils.enum_types import NormType

# Implements a simple intrinsic reward mechanism
# Reward is the Shanon entropy of the policy
class MaxEntropyModel(IntrinsicRewardBaseModel):
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

    def get_intrinsic_rewards(self,
        curr_obs, next_obs, last_mems, curr_act, curr_dones, entropy,
        obs_history, stats_logger
    ):
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
        return entropy.clone().cpu().detach().numpy(), None


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