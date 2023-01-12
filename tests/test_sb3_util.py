import gym
import numpy as np
import stable_baselines3
from stable_baselines3.common.buffers import ReplayBuffer

from algorithm_distillation.models.sb3_util.callback import RolloutCallback


def test_callback():
    env = gym.make('FrozenLake-v1')
    config = {'batch_size': 10,
              'n_steps': 100,  # determines the rollout size
              'policy': 'MlpPolicy'
              }
    buffer_config = {'buffer_size': 1000,
                     'observation_space': env.observation_space,
                     'action_space': env.action_space}
    buffer = ReplayBuffer(**buffer_config)
    agent = stable_baselines3.PPO(env=env, **config)
    cb = RolloutCallback(agent, buffer)

    assert buffer.pos == 0
    agent.learn(200, callback=cb)
    assert buffer.pos == 200
    # Note: There is a subtlety in comparing shapes of two buffers after the learning is finished.
    # RolloutBuffer.get changes the shapes of everything. It will call swap_and_flatten which flattens
    # the first two dimensions: (buffer_size, n_env, ...) -> (buffer_size * n_env, ...)
    assert np.all(np.isclose(buffer.observations[100:200].flatten(),
                             agent.rollout_buffer.observations.flatten()))
    assert np.all(np.isclose(buffer.actions[100:200].flatten(),
                             agent.rollout_buffer.actions.flatten()))
    assert np.all(np.isclose(buffer.rewards[100:200].flatten(),
                             agent.rollout_buffer.rewards.flatten()))

