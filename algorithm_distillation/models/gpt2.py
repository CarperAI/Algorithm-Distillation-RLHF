"""
This implements an AD transformers as the subclass of a GPT-2 model.
The code is inspired by the implementation of Decision Transformer (DT):
  https://github.com/kzl/decision-transformer
"""
import logging
from typing import Union

import torch
import transformers

from .ad_transformer import ADTransformer
from .util import stack_seq

logger = logging.getLogger(__name__)


class ZeroDummy(torch.nn.Module):
    """
    Directly output zeros of given shape (replace the last axis of the input tensor).
    """

    def __init__(self, output_shape: tuple):
        super(ZeroDummy, self).__init__()
        self.output_shape = output_shape

    def forward(self, x):
        shape = x.shape[:-1] + self.output_shape
        return torch.zeros(shape, dtype=x.dtype, device=x.device)


class GPT2AD(ADTransformer):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_size,
        max_ep_len=4096,
        action_tanh=True,
        obs_emb_cls=torch.nn.Linear,
        act_emb_cls=torch.nn.Linear,
        rew_emb_cls=torch.nn.Linear,
        **kwargs
    ):
        """
        The AD model with GPT2 as the underlying model.

        :param obs_dim: observation dimension (as a flattened tensor)
        :param act_dim: action dimension (as a flattened tensor)
        :param hidden_size: the dimension of the embedding space
        :param max_ep_len: (Optional) maximal episode length
        :param action_tanh: (Optional) apply tanh activation function on the action
        :param obs_emb_cls: (Optional) the nn.Module class for observation embedding
        :param act_emb_cls: (Optional) the nn.Module class for action embedding
        :param rew_emb_cls: (Optional) the nn.Module class for reward embedding
        :param kwargs: (Optional) other param for the underlying GPT2 transformers
        """
        super(GPT2AD, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.action_tanh = action_tanh
        # Generate the most basic GPT2 config
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)
        self.transformers = transformers.GPT2Model(config)
        # Remove the position embedding by replacing it with a dummy.
        self.transformers.wpe = ZeroDummy((hidden_size,))
        # This is our position embedding based on steps.
        self.step_embedding = torch.nn.Embedding(max_ep_len, self.hidden_size)

        # The embedding layers
        self.obs_embedding = obs_emb_cls(self.obs_dim, self.hidden_size)
        self.act_embedding = act_emb_cls(self.act_dim, self.hidden_size)
        self.rew_embedding = rew_emb_cls(1, self.hidden_size)

        # The last layers mapping to obs/action/reward from the embedding space.
        self.obs_head = torch.nn.Linear(self.hidden_size, self.obs_dim)
        # Generate action head. Append a tanh activation if `action_tanh` is True
        act_head_list = [torch.nn.Linear(self.hidden_size, self.act_dim)]
        if self.action_tanh:
            act_head_list.append(torch.nn.Tanh())
        self.act_head = torch.nn.Sequential(*act_head_list)
        self.rew_head = torch.nn.Linear(self.hidden_size, 1)

        self.layer_norm_in_embedding = torch.nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        obs,
        actions,
        rewards,
        current_obs=None,
        current_action=None,
        current_reward=None,
        attention_mask=None,
        step_ids=None,
        current_step_id=None,
        action_only=False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Input represents a sequence of observations, actions, rewards, they line up as
        following:

        `obs_1, act_1, rew_1, ..., obs_t, act_t, rew_t, latest_obs, latest_action, latest_reward`

        The parameters `latest_action`, `latest_reward` can be None if they do not apply.
        The model returns the next action, reward or state. Common usage is to predict
        the next action, so the result should normally be `next_action`.
        But other cases are allowed, e.g.,
        `..., latest_obs, latest_act, None` -> `next_reward`

        :param obs: (b, t, obs_dim) or None if b==0
        :param actions: (b, t, act_dim) or None if b==0
        :param rewards: (b, t, 1) or None if b==0
        :param current_obs: (Optional) (b, obs_dim)
        :param current_action: (Optional) shape (b, act_dim)
        :param current_reward: (Optional) shape (b, 1)
        :param attention_mask: (Optional) shape (b, t)
        :param step_ids: (Optional) shape (b, t), similar to `position_ids` in GPT2
        :param current_step_id: (Optional) the latest step id applied to the latest obs/act/rew.
        :param action_only: (Optional) return predicted actions only.
        :return: predicted action logits (if action_only) or predicted action logits, rewards, and obs.
        """
        if obs is None:
            assert current_obs is not None, "Empty input."
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            batch_size, timestep = current_obs.shape[0], 0
        else:
            device = obs.device
            batch_size, timestep, _ = obs.shape

        if current_step_id is None:
            if step_ids is not None:
                logger.warning(
                    "'current_step_id' defaults to the number of steps. But it may conflict with the given 'step_ids'."
                )
            current_step_id = timestep

        if step_ids is None and timestep > 0:
            step_ids = torch.arange(0, timestep, dtype=torch.long, device=device).view(
                1, timestep
            ).repeat((batch_size, 1))

        if timestep == 0:
            embedded_obs, embedded_act, embedded_rew = None, None, None
        else:
            embedded_steps = self.step_embedding(step_ids).view(
                batch_size, timestep, self.hidden_size
            )
            embedded_obs = self.obs_embedding(obs) + embedded_steps
            embedded_act = self.act_embedding(actions) + embedded_steps
            embedded_rew = self.rew_embedding(rewards) + embedded_steps

        embedded_latest_step = self.step_embedding(
            torch.tensor([current_step_id], dtype=torch.long, device=device)
        )

        # Embed the current obs/action/reward (stop if None)
        latest = [current_obs, current_action, current_reward]
        apply_cls = [self.obs_embedding, self.act_embedding, self.rew_embedding]
        extra = []
        for inp, cls in zip(latest, apply_cls):
            if inp is not None:
                extra.append(cls(inp) + embedded_latest_step)  # (b, hidden_size)
            else:
                break
        num_extra = len(extra)  # number of non-empty current obs/action/reward
        extra = torch.stack(extra, dim=1) if extra else None

        # Stack the input into (obs, act, rew, obs, act, rew, ...) sequence.
        # Note: only affects axis 1. Axis 0 (batch) and axis 2 (embedding) are preserved.
        input_seq = stack_seq(embedded_obs, embedded_act, embedded_rew, extra)
        input_seq = self.layer_norm_in_embedding(input_seq)

        if timestep == 0:
            attention_mask = torch.ones((batch_size, num_extra), dtype=torch.float, device=device)
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, timestep), dtype=torch.float, device=device
                )
            attention_mask = (
                attention_mask.unsqueeze(-1).repeat((1, 1, 3)).view(batch_size, -1)
            )
            attention_mask = torch.concat(
                [
                    attention_mask,
                    torch.ones((batch_size, num_extra), dtype=torch.float, device=device),
                ],
                dim=1,
            )

        # Do inference using the underlying transformer.
        output = self.transformers(
            inputs_embeds=input_seq, attention_mask=attention_mask
        )  # (b, *, hidden_size)
        pred_actions = output["last_hidden_state"][:, ::3, :]
        pred_rewards = output["last_hidden_state"][:, 1::3, :]
        pred_obs = output["last_hidden_state"][:, 2::3, :]

        if action_only:
            return self.act_head(pred_actions)
        else:
            return (
                self.act_head(pred_actions),
                self.rew_head(pred_rewards),
                self.obs_head(pred_obs),
            )
