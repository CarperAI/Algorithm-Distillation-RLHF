import torch
import pytest
from algorithm_distillation.models import GPT2AD


@pytest.mark.parametrize("n", [1, 2, 3])
def test_GPT2AD(n):
    model = GPT2AD(2, n, 12, max_ep_len=16)
    model.eval()  # Disable dropout
    sample_obs = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float)
    # two samples, two steps in each trajectory:
    #   1. [1, 2] -> [3, 4]
    #   2. [5, 6] -> [7, 8]
    sample_act = torch.tensor([[[0] * n, [1] * n], [[1] * n, [0] * n]], dtype=torch.float)
    sample_rew = torch.tensor([[[-1], [1]], [[-1], [0]]], dtype=torch.float)
    # ... (current: -> [5, 6])
    # ... (current: -> [9, 10])
    current_obs = torch.tensor([[5, 6], [9, 10]], dtype=torch.float)

    # Predict the next actions
    out = model(sample_obs, sample_act, sample_rew, action_only=True)
    assert out.shape == (2, 2, n)  # 2 observations -> predicting 2 actions
    out = model(sample_obs, sample_act, sample_rew, current_obs, action_only=True)
    assert out.shape == (2, 3, n)  # 3 observations -> predicting 3 actions
    out_act, out_rew, out_obs = model(sample_obs, sample_act, sample_rew, current_obs)
    assert out_act.shape == (2, 3, n)
    assert out_rew.shape == (2, 2, 1)
    assert out_obs.shape == (2, 2, 2)

    # Infer a subsequence and make sure the relevant predictions remain the same.
    current_obs = sample_obs[:, 1]
    sample_obs = sample_obs[:, :1]
    sample_act = sample_act[:, :1]
    sample_rew = sample_rew[:, :1]
    new_act, new_rew, new_obs = model(sample_obs, sample_act, sample_rew, current_obs)
    assert torch.all(new_act.isclose(out_act[:, :-1]))
    assert torch.all(new_rew.isclose(out_rew[:, :-1]))
    assert torch.all(new_obs.isclose(out_obs[:, :-1]))

