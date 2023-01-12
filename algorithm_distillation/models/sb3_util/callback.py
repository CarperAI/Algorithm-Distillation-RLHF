from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm


class RolloutCallback(BaseCallback):
    """
    This is a custom callback that collects rollouts from an on-policy algorithm.

    :param buffer: The external replay buffer to save rollouts.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages.
    """

    def __init__(
        self, agent: OnPolicyAlgorithm, buffer: ReplayBuffer, verbose: int = 0
    ):
        super().__init__(verbose)
        self.agent = agent
        self.buffer = buffer
        assert (
            self.buffer.buffer_size >= self.agent.rollout_buffer.buffer_size
        ), "External replay buffer must be larger than the agent rollout buffer."
        self._enforce_type()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        start = self.buffer.pos
        end = start + self.agent.rollout_buffer.buffer_size

        buffer = self.buffer
        tgt_buffer = self.agent.rollout_buffer

        ranges = [(start, min(buffer.buffer_size, end))]
        tgt_start = [0]
        if buffer.buffer_size < end:  # If overflown, wrap around to the beginning
            ranges.append((0, end - buffer.buffer_size))
            tgt_start.append(buffer.buffer_size - start)
            buffer.full = True

        for (st, ed), tst in zip(ranges, tgt_start):
            buffer.observations[st:ed] = tgt_buffer.observations[
                tst : tst + ed - st
            ].copy()
            buffer.actions[st:ed] = tgt_buffer.actions[tst : tst + ed - st].copy()
            buffer.rewards[st:ed] = tgt_buffer.rewards[tst : tst + ed - st].copy()
            self.buffer.pos = ed  # Update the pointer to the last ending range

    def _enforce_type(self) -> None:
        # Rollout buffer's observation and action are float and it can cause inconsistency.
        # So we force the types of buffer to be the same (should we emit a warning?).
        buffer = self.buffer
        tgt_buffer = self.agent.rollout_buffer
        if tgt_buffer.observations is None:
            raise RuntimeError("The rollout buffer is not initialized.")

        obs_type = tgt_buffer.observations.dtype
        action_type = tgt_buffer.actions.dtype
        reward_type = tgt_buffer.rewards.dtype

        buffer.observations = buffer.observations.astype(obs_type)
        buffer.actions = buffer.actions.astype(action_type)
        buffer.rewards = buffer.rewards.astype(reward_type)
