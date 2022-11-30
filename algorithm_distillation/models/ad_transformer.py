import abc

import torch


class ADTransformer(abc.ABC, torch.nn.Module):
    obs_dim: int
    act_dim: int

    @abc.abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def forward(
        self,
        obs,
        actions,
        rewards,
        current_obs,
        current_action=None,
        current_reward=None,
        attention_mask=None,
        step_ids=None,
        current_step_id=None,
    ):
        pass
