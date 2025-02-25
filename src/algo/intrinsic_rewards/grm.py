import gymnasium as gym
from typing import Dict, Any
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from src.algo.intrinsic_rewards.rnd import RNDModel
from src.algo.common_models.mlps import *
from src.utils.common_func import normalize_rewards
from src.utils.enum_types import NormType
from src.utils.running_mean_std import RunningMeanStd
import numpy as np


class GRMModel(RNDModel):
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
        # RND-specific params
        rnd_err_norm: int = 0,
        rnd_err_momentum: float = -1.0,
        rnd_use_policy_emb: int = 1,
        policy_cnn: Type[nn.Module] = None,
        policy_rnns: Type[nn.Module] = None,
        # GRM-specific params
        grm_delay: int = 0,
    ):
        super().__init__(observation_space, action_space, activation_fn, normalize_images, optimizer_class,
                         optimizer_kwargs, max_grad_norm, model_learning_rate, model_cnn_features_extractor_class,
                         model_cnn_features_extractor_kwargs, model_features_dim, model_latents_dim, model_mlp_norm,
                         model_cnn_norm, model_gru_norm, use_model_rnn, model_mlp_layers, gru_layers,
                         use_status_predictor, rnd_err_norm, rnd_err_momentum, rnd_use_policy_emb, policy_cnn,
                         policy_rnns)

        # self._build()
        # self._init_modules()
        # self._init_optimizers()

        self.grm_delay = grm_delay
        self.grm_buffer = np.zeros((observation_space.shape[0], grm_delay))


    def get_intrinsic_rewards(self, curr_obs, last_mems, curr_dones, stats_logger):
        with th.no_grad():
            rnd_loss, rnd_losses, model_mems = \
                self.forward(curr_obs, last_mems, curr_dones)
        rnd_rewards = rnd_losses.clone().cpu().numpy()

        if self.rnd_err_norm > 0:
            # Normalize RND error per step
            self.rnd_err_running_stats.update(rnd_rewards)
            rnd_rewards = normalize_rewards(
                norm_type=self.rnd_err_norm,
                rewards=rnd_rewards,
                mean=self.rnd_err_running_stats.mean,
                std=self.rnd_err_running_stats.std,
            )

        # GRM discount
        undiscounted_rewards = rnd_rewards.clone()
        

        stats_logger.add(rnd_loss=rnd_loss)
        return rnd_rewards, model_mems


    def optimize(self, rollout_data, stats_logger):
        rnd_loss, _, _ = \
            self.forward(
                rollout_data.observations,
                rollout_data.last_model_mems,
                rollout_data.episode_dones,
            )

        stats_logger.add(rnd_loss=rnd_loss)

        # Optimization
        self.model_optimizer.zero_grad()
        rnd_loss.backward()
        th.nn.utils.clip_grad_norm_(self.model_params, self.max_grad_norm)
        self.model_optimizer.step()