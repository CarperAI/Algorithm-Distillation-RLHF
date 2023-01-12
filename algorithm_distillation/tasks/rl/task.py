import abc
import math
import random
from typing import Optional

import gym
import numpy as np
import stable_baselines3
from stable_baselines3.common.buffers import ReplayBuffer

from algorithm_distillation.models.sb3_util.callback import RolloutCallback


class Task(abc.ABC):
    """
    This class controls the training of one single task.
    Each object controls one single trainable model inside.
    """

    # `obs_dim` marks the total dimension of the observations
    obs_dim: int
    act_dim: int
    env: object

    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def train(self, steps: int, **kwargs):
        """
        Train a number of steps.

        :param steps: the number of steps to train.
        :param kwargs: (Optional) extra parameters for SB3 `learn` method.
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
    # TODO: support TD3 for continuous action space?
    _algorithms = {
        "DQN": stable_baselines3.DQN,
        "PPO": stable_baselines3.PPO,
        "A2C": stable_baselines3.A2C,
    }
    _on_policy = ("PPO", "A2C")

    # If the environment has a discrete obs space of n classes, return an (n,)-shape array for each obs.
    # If the environment has a continuous obs space of certain shape, return a flattened array for each obs.
    obs_dim: int
    act_dim: int

    obs_cls: str
    act_cls: str

    def __init__(
        self,
        env,
        algorithm: str,
        buffer_size: int = 10_000,
        config: Optional[dict] = None,
    ):
        """
        GymTask takes in a gym environment and set up a stable-baselines3 algorithm to train a policy network.

        :param env: the gym environment
        :param algorithm: the stable-baselines3 algorithm
        :param buffer_size: (Optional) the buffer size. This refers to the stored history length where the
            transformer samples from, and applies to both on-policy and off-policy
        :param config: (Optional) the stable-baselines3 algorithm config (except for the gym environment)
        """
        self._env = env
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise NotImplementedError("Only supports discrete action spaces for now.")

        self.algorithm = algorithm
        if algorithm not in self._algorithms:
            raise ValueError(
                f"Input must be one of {self._algorithms.keys()}. Got {algorithm} instead."
            )
        self.buffer_size = buffer_size

        # Set up the agent
        self.config = {} if config is None else config.copy()
        self.config["env"] = self.env
        if "buffer_size" in self.config or algorithm not in self._on_policy:
            # Overwrite policy buffer size if exists/doing off-policy
            self.config["buffer_size"] = self.buffer_size

        # Set up the defaults
        defaults = {
            "policy": "MultiInputPolicy",  # a common policy applicable most of the time
        }
        # Set up the defaults specifically for either on-policy or off-policy algos
        if self.algorithm in self._on_policy:
            policy_level_defaults = {
                "n_steps": 2048,
            }
        else:
            policy_level_defaults = {}
        self.config = {**defaults, **policy_level_defaults, **self.config}
        # Set up the agent
        self.agent = self._algorithms[algorithm](**self.config)

        # If on-policy, set up the the callback to collect rollout
        if self.algorithm in self._on_policy:
            buffer_config = {
                "buffer_size": self.buffer_size,
                "observation_space": self._env.observation_space,
                "action_space": self._env.action_space,
            }
            self.buffer = ReplayBuffer(**buffer_config)
            self.callback = RolloutCallback(self.agent, self.buffer)
        else:
            self.buffer = self.agent.replay_buffer
            self.callback = None

        self.obs_dim, self.obs_cls = self._get_obs_specs()
        self.act_dim, self.act_cls = self._get_act_specs()

    def train(self, steps: int, **kwargs):
        self.agent.learn(total_timesteps=steps, callback=self.callback, **kwargs)

    def sample_history(
        self, length: int, skip: int = 0, most_recent: bool = False
    ) -> tuple:
        """
        This implementation will sample their most recent histories.

        :param length: the length of the history .
        (note: length of training steps instead of length of sequence! Every step
        includes obs, act, rew. Thus, the final sequence is 3x as long.)
        :param skip: (Optional) skip certain amount of steps between two states
        :param most_recent: (Optional) get the most recent histories. False to sample randomly.
        :return: a tuple of (observations, actions, rewards). Each is a tensor.
        """
        if self.buffer.n_envs != 1:
            raise NotImplementedError("Not supporting parallel environments yet.")

        if most_recent:
            return self._get_most_recent_history(self.buffer, length, skip)
        else:
            return self._randomly_sample_buffer(self.buffer, length, skip)

    def _get_obs_specs(self) -> tuple:
        obs_space = self.env.observation_space

        if isinstance(obs_space, gym.spaces.Discrete):
            return obs_space.n, "discrete"
        elif isinstance(obs_space, gym.spaces.Box):
            return math.prod([n for n in obs_space.shape]), "box"
        else:
            raise NotImplementedError(
                f"The observation space does not support {type(obs_space)}."
            )

    def _get_act_specs(self) -> tuple:
        act_space = self.env.action_space

        if isinstance(act_space, gym.spaces.Discrete):
            return act_space.n, "discrete"
        else:
            raise NotImplementedError(
                f"The observation space does not support {type(act_space)}."
            )

    @property
    def env(self) -> gym.Env:
        if not hasattr(self, "_env"):
            raise ValueError('Need to assign "_env" attribute first.')
        return self._env

    def obs_post_process(self, obs: np.ndarray) -> np.ndarray:
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
    def act_post_process(act: np.ndarray) -> np.ndarray:
        """
        Post-process the actions. Assume actions are discrete with shape (length, 1) or (length).

        :param act: the batched action array.
        :return: the processed action array of shape (length, 1).
        """
        return act.reshape((-1, 1)).astype(int)

    @staticmethod
    def rew_post_process(rew: np.ndarray) -> np.ndarray:
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
        total_length = (length - 1) * (skip + 1) + 1

        assert (
            buffer.buffer_size > total_length
        ), "Replay buffer size must be larger than the sequence length."

        start = (pos - total_length + buffer.buffer_size) % buffer.buffer_size
        end = pos

        return self._get_obs_act_rew(buffer, start, end, skip)

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
        total_length = (length - 1) * (skip + 1) + 1
        assert (
            buffer.buffer_size > total_length
        ), "Replay buffer size must be larger than the sequence length."

        if not buffer.full:
            start = random.randint(0, buffer.pos - total_length)
            end = start + total_length
        else:
            start = (
                random.randint(0, buffer.buffer_size - total_length) + buffer.pos
            ) % buffer.buffer_size
            end = (start + total_length) % buffer.buffer_size

        return self._get_obs_act_rew(buffer, start, end, skip)

    @staticmethod
    def _get_range(
        array: np.ndarray, start: int, end: int, interval: int
    ) -> np.ndarray:
        """
        A helper function to either slice array[start:end:interval] or combine array[start::interval] and
        array[:end:interval] depending on whether start < end.

        :param array: the sliced array.
        :param start: the starting index.
        :param end: the ending index (exclusive).
        :param interval: the interval.
        :return: the sliced sub-array.
        """
        if start < end:
            return array[start:end:interval]
        else:
            return np.concatenate(
                [array[start::interval], array[:end:interval]], axis=0
            )

    def _get_obs_act_rew(self, buffer: ReplayBuffer, start: int, end: int, skip: int):
        """
        Return a tuple (obs, act, rew) sampled according to the buffer and the parameters.

        :param buffer: the replay buffer.
        :param start: the starting index.
        :param end: the ending index.
        :param skip: the amount of states to skip.
        :return: the tuple (obs, act, rew)
        """
        return (
            self.obs_post_process(
                self._get_range(buffer.observations, start, end, skip + 1)
            ),
            self.act_post_process(
                self._get_range(buffer.actions, start, end, skip + 1)
            ),
            self.rew_post_process(
                self._get_range(buffer.rewards, start, end, skip + 1)
            ),
        )
