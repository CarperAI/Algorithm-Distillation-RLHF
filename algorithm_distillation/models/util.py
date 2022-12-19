import torch


def stack_seq(obs, act, rew, extra=None) -> torch.Tensor:
    """
    Stack up into a sequence (obs, act, rew, obs, act, rew, ...) in axis 1,
    and append extra in the end.
    :param obs: shape (b, t, hidden_size)
    :param act: shape (b, t, hidden_size)
    :param rew: shape (b, t, hidden_size)
    :param extra: (Optional) shape (b, i, hidden_size) where i can be 1, 2 or 3
    :return: shape (b, 3*t+i, hidden_size)
    """
    if obs is None:
        # batch size is 0. We return extra.
        return extra

    batch_size, timestep, _ = obs.shape
    stacked = torch.stack((obs, act, rew), dim=2).view(batch_size, 3 * timestep, -1)
    if extra is None:
        return stacked
    else:
        return torch.concat([stacked, extra], dim=1)
