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


def get_sequence(arr: torch.Tensor, num_items: int, end_idx: int, interval: int):
    """
    Get the subsequence of indices 'end_idx - num_items * interval, ..., end_idx - interval' from
    the PyTorch tensor `arr`.
    Note:
      - While `end_idx` is excluded, the intervals start backwards from it.
      - Negative indices are ignored (meaning the return could be shorter).
      - If the length of `arr` is less than `end_idx`, treat `arr` as a circular buffer.
    Example:
        arr = [7, 8, 9], length is 3
        num_items = 3, end_idx = 4, interval = 2
            -> [7, 9]
        num_items = 1, end_idx = 3, interval = 2
            -> [8]
        num_items = 1, end_idx = 2, interval = 2
            -> [7]

    :param arr: a tensor whose first dimension is the index we are concerned with.
    :param num_items: the max number of items.
    :param end_idx: the end index.
    :param interval: the interval length.
    :return: a subsequence of `arr` according to the description.
    """
    length = arr.size(0)
    # The size of the circular buffer determines max how many items it can return
    num_items = min(num_items, (length + interval - 1) // interval)
    # Get the actual start index (inclusive)
    start_idx = max(end_idx - num_items * interval, end_idx % interval)

    if end_idx >= length:
        # The subseq cuts in the middle. Update `end_idx` to the actual end index on the second half.
        end_idx = max(0, end_idx % length - interval + 1)
        return torch.concat(
            [arr[start_idx % length :: interval], arr[:end_idx:interval]], dim=0
        )
    else:
        return arr[start_idx:end_idx:interval]
