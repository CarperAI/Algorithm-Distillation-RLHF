import gym
import pytest

from algorithm_distillation.rl_tasks import GymTask


@pytest.mark.parametrize("policy", ['DQN', 'PPO', 'A2C'])
def test_gym_task(policy: str):
    env = gym.make('FrozenLake-v1')
    if policy in ['DQN', 'TD3']:
        config = {'learning_starts': 10,
                  'policy': 'MlpPolicy'
                  }
    else:
        config = {'n_steps': 3,
                  'batch_size': 10,
                  'policy': 'MlpPolicy'
                  }
    if policy == 'A2C':
        config.pop('batch_size')
    task = GymTask(env, policy, buffer_size=100, config=config)
    assert task.obs_cls == 'discrete'
    assert task.obs_dim == 16  # The default observation space of FrozenLake is discrete 16 labels

    task.train(150)

    for most_recent in [True, False]:
        for _ in range(100):  # There is a bit of randomness. Try many times.
            sample = task.sample_history(10, most_recent=most_recent)
            assert sample[0].shape == (10, 16)  # Observations are discrete classes
            assert sample[1].shape == (10, 1)  # Actions are discrete classes
            assert sample[2].shape == (10, 1)  # Rewards.

            sample = task.sample_history(10, skip=3, most_recent=most_recent)
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
    _env = gym.make('Alien-v4')
    config = {'learning_starts': 10,
              'buffer_size': 1000,
              'policy': 'MlpPolicy'
              }
    task = GymTask(_env, 'DQN', config)
    task.train(100)
    """

