from typing import NamedTuple, Optional
import torch as th

class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    new_observations: th.Tensor
    last_policy_mems: th.Tensor
    last_model_mems: th.Tensor
    episode_starts: th.Tensor
    episode_dones: th.Tensor
    actions: th.Tensor
    old_ext_values: th.Tensor
    old_int_values: th.Tensor
    old_log_prob: th.Tensor
    old_entropy: th.Tensor
    ext_advantages: th.Tensor
    int_advantages: th.Tensor
    ext_returns: th.Tensor
    int_returns: th.Tensor
    curr_key_status: Optional[th.Tensor]
    curr_door_status: Optional[th.Tensor]
    curr_target_dists: Optional[th.Tensor]
