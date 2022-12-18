import gym
from algorithm_distillation import GymTask


def test_gym_task():
    env = gym.make('FrozenLake-v1')
    config = {'learning_starts': 10,
              'buffer_size': 1000,
              'policy': 'MlpPolicy'
              }
    task = GymTask(env, 'DQN', config)
    assert task.obs_cls == 'discrete'
    assert task.obs_dim == 16  # The default observation space of FrozenLake is discrete 16 labels

    task.train(100)

    for most_recent in [True, False]:
        sample = task.sample_history(10, most_recent=most_recent)
        assert sample[0].shape == (10, 16)  # Observations are discrete classes
        assert sample[1].shape == (10, 1)  # Actions are discrete classes
        assert sample[2].shape == (10, 1)  # Rewards.

        sample = task.sample_history(10, skip=2, most_recent=most_recent)
        assert sample[0].shape == (10, 16)  # Observations are discrete classes
        assert sample[1].shape == (10, 1)  # Actions are discrete classes
        assert sample[2].shape == (10, 1)  # Rewards.

    # More than buffer
    try:
        sample = task.sample_history(1000, skip=0, most_recent=True)
        assert False, "Error should have been raised."
    except AssertionError:
        assert True

    """# Please install gym[atari, accept-rom-license] manually if you want to run Atari.
    env = gym.make('Alien-v4')
    config = {'learning_starts': 10,
              'buffer_size': 1000,
              'policy': 'MlpPolicy'
              }
    task = GymTask(env, 'DQN', config)
    task.train(100)
    """

