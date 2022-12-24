import torch
from algorithm_distillation.models.util import stack_seq, get_sequence


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

    # when batch size is 0, all of obs, act, rew should be None. It should just return the extra.
    result = stack_seq(None, None, None, extra)
    assert torch.all(result.isclose(extra))


def test_get_sequence():
    arr = torch.tensor([7, 8, 9])
    n = 3
    end_idx = 4
    interval = 2
    assert all(get_sequence(arr, n, end_idx, interval) == torch.tensor([7, 9]))
    n = 1
    end_idx = 3
    interval = 2
    assert all(get_sequence(arr, n, end_idx, interval) == torch.tensor([8]))
    n = 1
    end_idx = 2
    interval = 2
    assert all(get_sequence(arr, n, end_idx, interval) == torch.tensor([7]))

    n = 3
    end_idx = 5
    interval = 2
    # can only fetch a max elem that the circular buffer allows (i.e., 2 in this case)
    assert all(get_sequence(arr, n, end_idx, interval) == torch.tensor([8, 7]))
