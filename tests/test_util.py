import torch
from algorithm_distillation.models.util import stack_seq


def test_stack_seq():
    obs = torch.tensor([[[1, 1], [2, 3]], [[3, 1], [4, 2]]])
    act = obs + 3
    rew = obs + 6

    extra = torch.tensor([[[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]])

    # (obs, act, rew) on the second axis
    r = torch.tensor([[[1, 1],
                       [4, 4],
                       [7, 7],
                       [2, 3],
                       [5, 6],
                       [8, 9],
                       [-1, -2],
                       [-3, -4]],

                      [[3, 1],
                       [6, 4],
                       [9, 7],
                       [4, 2],
                       [7, 5],
                       [10, 8],
                       [-5, -6],
                       [-7, -8]]])

    result = stack_seq(obs, act, rew, extra)
    assert torch.all(result.isclose(r))
