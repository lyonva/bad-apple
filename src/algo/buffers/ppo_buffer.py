import numpy as np
import torch as th

from gymnasium import spaces
from gymnasium.spaces import Dict
from typing import Generator, Optional, Union

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize

from src.algo.buffers.type_aliases import RolloutBufferSamples
from src.utils.common_func import normalize_rewards
from src.utils.running_mean_std import RunningMeanStd


class PPORolloutBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        features_dim: int = 0,
        dim_policy_traj: int = 0,
        dim_model_traj: int = 0,
        int_rew_coef: float = 1.0,
        ext_rew_coef: float = 1.0,
        int_rew_norm: int = 0,
        int_rew_clip: float = 0.0,
        int_rew_eps: float = 1e-8,
        adv_momentum: float = 0.0,
        adv_norm: int = 0,
        adv_eps: float = 1e-8,
        gru_layers: int = 1,
        int_rew_momentum: Optional[float] = None,
        use_status_predictor: int = 0,
        adops_enabled: bool = False,
        adopes_epsilon : float = 1e-6,
        adopes_coef_max_rollout: int = 1,
        cost_limit: float = 0,
        saber_enabled: bool = False,
        saber_epsilon: float = 1e-6,
        saber_zeta_min_rollout: int = 1,
        saber_zeta_max_rollout: int = 1,
    ):
        if isinstance(observation_space, Dict):
            observation_space = list(observation_space.values())[0]
        super(PPORolloutBuffer, self)\
            .__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.int_rew_coef = int_rew_coef
        self.int_rew_norm = int_rew_norm
        self.int_rew_clip = int_rew_clip
        self.ext_rew_coef = ext_rew_coef
        self.features_dim = features_dim
        self.dim_policy_traj = dim_policy_traj
        self.dim_model_traj = dim_model_traj
        self.int_rew_eps = int_rew_eps
        self.adv_momentum = adv_momentum
        self.adv_mean = None
        self.int_rew_mean = None
        self.int_rew_std = None
        self.ir_mean_buffer = []
        self.ir_std_buffer = []
        self.use_status_predictor = use_status_predictor
        self.adops_enabled = adops_enabled
        self.adopes_coef_max_rollout = adopes_coef_max_rollout
        self.cost_limit = cost_limit
        self.saber_enabled = saber_enabled
        self.saber_epsilon = saber_epsilon
        self.saber_zeta_min_rollout = saber_zeta_min_rollout
        self.saber_zeta_max_rollout = saber_zeta_max_rollout
        self.adv_norm = adv_norm
        self.adv_eps = adv_eps
        self.gru_layers = gru_layers
        self.int_rew_momentum = int_rew_momentum
        self.int_rew_stats = RunningMeanStd(momentum=self.int_rew_momentum)
        self.ext_advantage_stats = RunningMeanStd(momentum=self.adv_momentum)
        self.int_advantage_stats = RunningMeanStd(momentum=self.adv_momentum)
        self.cost_advantage_stats = RunningMeanStd(momentum=self.adv_momentum)
        self.sb_advantage_stats = RunningMeanStd(momentum=self.adv_momentum)
        self.n_rollout = -1

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.new_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.last_policy_mems = np.zeros((self.buffer_size, self.n_envs, self.gru_layers, self.dim_policy_traj), dtype=np.float32)
        self.last_model_mems = np.zeros((self.buffer_size, self.n_envs, self.gru_layers, self.dim_model_traj), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.intrinsic_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.ext_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.int_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.ext_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.int_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.entropy = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.ext_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.int_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.sb_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        if self.use_status_predictor:
            self.curr_key_status = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
            self.curr_door_status = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
            self.curr_target_dists = np.zeros((self.buffer_size, self.n_envs, 3), dtype=np.float32)
        self.generator_ready = False
        self.n_rollout += 1
        super(PPORolloutBuffer, self).reset()

    def compute_intrinsic_rewards(self) -> None:
        # Normalize intrinsic rewards per rollout buffer
        self.int_rew_stats.update(self.intrinsic_rewards.reshape(-1))
        self.int_rew_mean = self.int_rew_stats.mean
        self.int_rew_std = self.int_rew_stats.std
        self.intrinsic_rewards = normalize_rewards(
            norm_type=self.int_rew_norm,
            rewards=self.intrinsic_rewards,
            mean=self.int_rew_mean,
            std=self.int_rew_std,
            eps=self.int_rew_eps,
        )

        # Rescale by IR coef
        self.intrinsic_rewards *= self.int_rew_coef

        # Clip after normalization
        if self.int_rew_clip > 0:
            self.intrinsic_rewards = np.clip(self.intrinsic_rewards, -self.int_rew_clip, self.int_rew_clip)
    
    def _adops_zeta(self):
        return np.clip(self.n_rollout / self.adopes_coef_max_rollout, 0, 1) 
    
    def _saber_zeta(self):
        return np.clip((self.n_rollout - self.saber_zeta_min_rollout + 1) / (self.saber_zeta_max_rollout - self.saber_zeta_min_rollout + 1), 0, 1)
    
    def compute_saber_r2(self, v_s, q_s, v_r, v_rplus1, r):
        return np.where(q_s < v_s,
            np.clip(v_s - q_s + v_r - self.gamma * v_rplus1 - r - self.saber_epsilon, None, 0),
            0,
        )
    
    def compute_adops_f2(self, v_e, q_e, v_i, v_iplus1, f):
        return np.where(q_e < v_e,
            np.clip(v_e - q_e + v_i - self.gamma * v_iplus1 - f - self.saber_epsilon, None, 0),
            np.clip(v_e - q_e + v_i - self.gamma * v_iplus1 - f, 0, None),
        )


    def compute_returns_and_advantage(self, last_ext_values: th.Tensor, last_int_values: th.Tensor, last_cost_values: th.Tensor, dones: np.ndarray) -> None:
        # Rescale extrinisc rewards
        self.rewards *= self.ext_rew_coef

        # Convert to numpy
        last_ext_values = last_ext_values.clone().cpu().numpy().flatten()
        last_int_values = last_int_values.clone().cpu().numpy().flatten()
        last_cost_values = last_cost_values.clone().cpu().numpy().flatten()

        last_ext_gae_lam = 0
        last_int_gae_lam = 0
        last_cost_gae_lam = 0
        last_sb_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_ext_values = last_ext_values
                next_int_values = last_int_values
                next_cost_values = last_cost_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_ext_values = self.ext_values[step + 1]
                next_int_values = self.int_values[step + 1]
                next_cost_values = self.cost_values[step + 1]

            q_ext = self.rewards[step] + self.gamma * next_ext_values * next_non_terminal
            q_int = self.intrinsic_rewards[step] + self.gamma * next_int_values  * next_non_terminal
            q_cost = self.costs[step]  + self.gamma * next_cost_values * next_non_terminal
            q_s = np.clip(self.cost_limit - q_cost, None, 0)
            v_s = np.clip(self.cost_limit - self.cost_values[step], None, 0)

            if self.saber_enabled:
                q_ext += self._saber_zeta() * self.compute_saber_r2(v_s, q_s, self.ext_values[step], next_ext_values, self.rewards[step])
            if self.adops_enabled:
                q_int += self._adops_zeta() * self.compute_adops_f2(self.ext_values[step], q_ext, self.int_values[step], next_int_values, self.intrinsic_rewards[step])

            delta_ext = q_ext - self.ext_values[step]
            delta_int = q_int - self.int_values[step]
            delta_cost = q_cost - self.cost_values[step]
            delta_sb = q_s - v_s
            last_ext_gae_lam = delta_ext + self.gamma * self.gae_lambda * next_non_terminal * last_ext_gae_lam
            last_int_gae_lam = delta_int + self.gamma * self.gae_lambda * next_non_terminal * last_int_gae_lam
            last_cost_gae_lam = delta_cost + self.gamma * self.gae_lambda * next_non_terminal * last_cost_gae_lam
            last_sb_gae_lam = delta_sb + self.gamma * self.gae_lambda * next_non_terminal * last_sb_gae_lam
            self.ext_advantages[step] = last_ext_gae_lam
            self.int_advantages[step] = last_int_gae_lam
            self.cost_advantages[step] = last_cost_gae_lam
            self.sb_advantages[step] = last_sb_gae_lam
            

        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.ext_returns = self.ext_advantages + self.ext_values 
        self.int_returns = self.int_advantages + self.int_values
        self.cost_returns = self.cost_advantages + self.cost_values

        # Normalize advantages per rollout buffer
        if self.adv_norm:
            self.ext_advantage_stats.update(self.ext_advantages)
            self.int_advantage_stats.update(self.int_advantages)
            self.cost_advantage_stats.update(self.cost_advantages)
            self.sb_advantage_stats.update(self.sb_advantages)
            self.ext_adv_mean = self.ext_advantage_stats.mean
            self.ext_adv_std = self.ext_advantage_stats.std
            self.int_adv_mean = self.int_advantage_stats.mean
            self.int_adv_std = self.int_advantage_stats.std
            self.cost_adv_mean = self.cost_advantage_stats.mean
            self.cost_adv_std = self.cost_advantage_stats.std
            self.sb_adv_mean = self.sb_advantage_stats.mean
            self.sb_adv_std = self.sb_advantage_stats.std

            # Standardization
            if self.adv_norm == 2:
                self.ext_advantages = (self.ext_advantages - self.ext_adv_mean) / (self.ext_adv_std + self.adv_eps)
                self.int_advantages = (self.int_advantages - self.int_adv_mean) / (self.int_adv_std + self.adv_eps)
                self.cost_advantages = (self.cost_advantages - self.cost_adv_mean) / (self.cost_adv_std + self.adv_eps)
                self.sb_advantages = self.sb_advantages / (self.sb_adv_std + self.adv_eps)

            # Standardization without subtracting the mean value
            if self.adv_norm == 3:
                self.ext_advantages = self.ext_advantages / (self.ext_adv_std + self.adv_eps)
                self.int_advantages = self.int_advantages / (self.int_adv_std + self.adv_eps)
                self.cost_advantages = self.cost_advantages / (self.cost_adv_std + self.adv_eps)
                self.sb_advantages = self.sb_advantages / (self.sb_adv_std + self.adv_eps)

    def add(
        self,
        obs: np.ndarray,
        new_obs: np.ndarray,
        last_policy_mem: th.Tensor,
        last_model_mem: th.Tensor,
        action: np.ndarray,
        reward: np.ndarray,
        intrinsic_reward: np.ndarray,
        cost: np.ndarray,
        episode_start: np.ndarray,
        episode_done: np.ndarray,
        ext_value: th.Tensor,
        int_value: th.Tensor,
        cost_value: th.Tensor,
        log_prob: Optional[th.Tensor],
        entropy: Optional[th.Tensor],
        curr_key_status: Optional[np.ndarray],
        curr_door_status: Optional[np.ndarray],
        curr_target_dist: Optional[np.ndarray],
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.new_observations[self.pos] = np.array(new_obs).copy()
        self.last_policy_mems[self.pos] = last_policy_mem.clone().cpu().numpy()
        self.last_model_mems[self.pos] = last_model_mem.clone().cpu().numpy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.intrinsic_rewards[self.pos] = np.array(intrinsic_reward).copy()
        self.costs[self.pos] = np.array(cost).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.episode_dones[self.pos] = np.array(episode_done).copy()
        self.ext_values[self.pos] = ext_value.clone().cpu().numpy().flatten()
        self.int_values[self.pos] = int_value.clone().cpu().numpy().flatten()
        self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.entropy[self.pos] = entropy.clone().cpu().numpy()
        if self.use_status_predictor:
            self.curr_key_status[self.pos] = np.array(curr_key_status).copy()
            self.curr_door_status[self.pos] = np.array(curr_door_status).copy()
            self.curr_target_dists[self.pos] = np.array(curr_target_dist).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def prepare_data(self):
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "new_observations",
                "last_policy_mems",
                "last_model_mems",
                "episode_starts",
                "episode_dones",
                "actions",
                "ext_values",
                "int_values",
                "cost_values",
                "log_probs",
                "entropy",
                "ext_advantages",
                "int_advantages",
                "cost_advantages",
                "sb_advantages",
                "ext_returns",
                "int_returns",
                "cost_returns",
                "costs",
            ]
            if self.use_status_predictor:
                _tensor_names += [
                    "curr_key_status",
                    "curr_door_status",
                    "curr_target_dists",
                ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        self.prepare_data()

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        indices = np.random.permutation(self.buffer_size * self.n_envs)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.new_observations[batch_inds],
            self.last_policy_mems[batch_inds],
            self.last_model_mems[batch_inds],
            self.episode_starts[batch_inds],
            self.episode_dones[batch_inds],
            self.actions[batch_inds],
            self.ext_values[batch_inds].flatten(),
            self.int_values[batch_inds].flatten(),
            self.cost_values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.entropy[batch_inds].flatten(),
            self.ext_advantages[batch_inds].flatten(),
            self.int_advantages[batch_inds].flatten(),
            self.cost_advantages[batch_inds].flatten(),
            self.sb_advantages[batch_inds].flatten(),
            self.ext_returns[batch_inds].flatten(),
            self.int_returns[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
            self.costs[batch_inds].flatten(),
        )
        if self.use_status_predictor:
            data += (
                self.curr_key_status[batch_inds].flatten(),
                self.curr_door_status[batch_inds].flatten(),
                self.curr_target_dists[batch_inds].flatten(),
            )

        samples = tuple(map(lambda x: self.to_torch(x, copy=False), data))
        if not self.use_status_predictor:
            samples += (None, None, None,)
        return RolloutBufferSamples(*samples)
