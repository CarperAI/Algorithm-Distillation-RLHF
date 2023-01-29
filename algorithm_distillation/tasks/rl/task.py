import abc
import math
import random
from typing import Optional

import gym
import numpy as np
import stable_baselines3
from stable_baselines3.common.buffers import ReplayBuffer


class Task(abc.ABC):
    """
    This class controls the training of one single task.
    Each object controls one single trainable model inside.
    """

    # `obs_dim` marks the total dimension of the observations
    obs_dim: int

    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def train(self, steps: int):
        """
        Train a number of steps.

        :param steps: the number of steps to train
        :return: None
        """
        pass

    @abc.abstractmethod
    def sample_history(self, length: int, skip: int = 0) -> tuple:
        """
        Sample a history of specific length.

        :param length: the length of the history.
        (note: length of training steps instead of length of sequence! Every step
        includes obs, act, rew. Thus, the final sequence is 3x as long.)
        :param skip: (Optional) skip certain amount of steps between two states.
        :return: a tuple of (observations, actions, rewards). Each is a tensor.
        """
        pass


class GymTask(Task):
    # TODO: Still need some work to set up the on-policy algorithms due to problems in their rollout buffers.
    _algorithms = {"DQN": stable_baselines3.DQN}
    _on_policy = ("PPO",)

    # If the environment has a discrete obs space of n classes, return an (n,)-shape array for each obs.
    # If the environment has a continuous obs space of certain shape, return a flattened array for each obs.
    obs_dim: int
    obs_cls: str

    def __init__(self, env, algorithm: str, config: Optional[dict] = None):
        """
        GymTask takes in a gym environment and set up a stable-baselines3 algorithm to train a policy network.

        :param env: the gym environment
        :param algorithm: the stable-baselines3 algorithm
        :param config: (Optional) the stable-baselines3 algorithm config (except for the gym environment)
        """
        self.env = env
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("Only support discrete action spaces.")

        self.algorithm = algorithm
        if algorithm not in self._algorithms:
            raise ValueError(
                f"Input must be one of {self._algorithms.keys()}. Got {algorithm} instead."
            )

        self.config = {} if config is None else config.copy()
        self.config["env"] = self.env
        # Default to MultiInputPolicy which is applicable most of the time.
        if "policy" not in self.config:
            self.config["policy"] = "MultiInputPolicy"

        self.agent = self._algorithms[algorithm](**self.config)
        self.obs_dim, self.obs_cls = self._get_obs_specs()

    def train(self, steps: int):
        self.agent.learn(total_timesteps=steps)

    def sample_history(
        self, length: int, skip: int = 0, most_recent: bool = False
    ) -> tuple:
        """
        This implementation will sample their most recent histories.

        :param length: the length of the history .
        (note: length of training steps instead of length of sequence! Every step
        includes obs, act, rew. Thus, the final sequence is 3x as long.)
        :param skip: (Optional) skip certain amount of steps between two states
        :param most_recent: (Optional) sample from the most recent histories. False to sample randomly.
        :return: a tuple of (observations, actions, rewards). Each is a tensor.
        """
        if self.algorithm in self._on_policy:
            raise NotImplementedError("Not supporting on-policy algorithms yet.")

        buffer = self.agent.replay_buffer

        if buffer.n_envs != 1:
            raise NotImplementedError("Not supporting parallel environments yet.")

        if most_recent:
            return self._get_most_recent_history(buffer, length, skip)
        else:
            return self._randomly_sample_buffer(buffer, length, skip)

    def _get_obs_specs(self) -> tuple:
        if not hasattr(self, "env"):
            raise ValueError('Need to assign "env" attribute first.')
        obs_space = self.env.observation_space

        if isinstance(obs_space, gym.spaces.Discrete):
            return obs_space.n, "discrete"
        elif isinstance(obs_space, gym.spaces.Box):
            return math.prod([n for n in obs_space.shape]), "box"
        else:
            raise NotImplementedError(
                f"The observation space does not support {type(obs_space)}."
            )

    def _obs_post_process(self, obs: np.ndarray) -> np.ndarray:
        """
        Post-process the observations according to its type and shape.

        :param obs: the batched observation array.
        :return: the processed observation array of shape (length, obs_dim).
        """
        length = obs.shape[0]
        if self.obs_cls == "discrete":
            obs = obs.reshape((length, 1))
            # Return arrays according to one-hot encoding
            return (obs == np.tile(np.arange(self.obs_dim), (length, 1))).astype(float)
        elif self.obs_cls == "box":
            # Flatten all the other
            return obs.reshape((length, -1))
        else:
            raise RuntimeError("Impossible code path.")

    @staticmethod
    def _act_post_process(act: np.ndarray) -> np.ndarray:
        """
        Post-process the actions. Assume actions are discrete with shape (length, 1) or (length).

        :param act: the batched action array.
        :return: the processed action array of shape (length, 1).
        """
        return act.reshape((-1, 1)).astype(int)

    @staticmethod
    def _rew_post_process(rew: np.ndarray) -> np.ndarray:
        """
        Post-process the rewards. Rewards are scalars with shape (length,).

        :param rew: the batched reward array.
        :return: the processed reward array of shape (length, 1).
        """
        return rew.reshape((-1, 1)).astype(float)

    def _get_most_recent_history(
        self, buffer: ReplayBuffer, length: int, skip: int
    ) -> tuple:
        """
        Get the most recent history from the buffer.

        :param buffer: ReplayBuffer object from stable-baselines3.
        :param length: the length of steps to sample.
        :param skip: the amount to skip between states.
        :return: an (observations, actions, rewards) tuple.
        """
        pos = buffer.pos
        total_length = length * (skip + 1)
        if pos >= total_length:
            return (
                self._obs_post_process(
                    buffer.observations[pos - total_length : pos : skip + 1]
                ),
                self._act_post_process(
                    buffer.actions[pos - total_length : pos : skip + 1]
                ),
                self._rew_post_process(
                    buffer.rewards[pos - total_length : pos : skip + 1]
                ),
            )
        else:
            latest = (
                self._obs_post_process(buffer.observations[: pos : skip + 1]),
                self._act_post_process(buffer.actions[: pos : skip + 1]),
                self._rew_post_process(buffer.rewards[: pos : skip + 1]),
            )
            if buffer.full:
                extra = (
                    self._obs_post_process(
                        buffer.observations[pos - length :: skip + 1]
                    ),
                    self._act_post_process(buffer.actions[pos - length :: skip + 1]),
                    self._rew_post_process(buffer.rewards[pos - length :: skip + 1]),
                )
                latest = tuple(
                    [np.concatenate([extra[i], latest[i]], axis=0) for i in range(3)]
                )

            return latest

    def _randomly_sample_buffer(
        self, buffer: ReplayBuffer, length: int, skip: int
    ) -> tuple:
        """
        Randomly sample a sequence from the buffer (requires that there is enough to sample from).

        :param buffer: ReplayBuffer object from stable-baselines3.
        :param length: the length of steps to sample.
        :param skip: the amount to skip between states.
        :return: an (observations, actions, rewards) tuple.
        """
        upper_bound = buffer.size if buffer.full else buffer.pos
        total_length = length * (skip + 1)

        if total_length > upper_bound:
            raise IndexError("Buffer contains fewer samples than necessary.")

        start = random.randint(0, upper_bound - total_length)
        return (
            self._obs_post_process(
                buffer.observations[start : start + total_length : skip + 1]
            ),
            self._act_post_process(
                buffer.actions[start : start + total_length : skip + 1]
            ),
            self._rew_post_process(
                buffer.rewards[start : start + total_length : skip + 1]
            ),
        )
