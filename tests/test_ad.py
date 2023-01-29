import gym

from algorithm_distillation import GymTask, TaskManager, GymAD
from algorithm_distillation.models import GPT2AD


def test_ad():
    env = gym.make('FrozenLake-v1')
    config = {'learning_starts': 10,
              'buffer_size': 1000,
              'policy': 'MlpPolicy'
              }
    model = GPT2AD(env.observation_space.n, env.action_space.n, 12, max_ep_len=16)

    task = GymTask(env, 'DQN', config)
    task_manager = TaskManager([task])
    task_manager.train(100)

    ad = GymAD(model)
    ad.train(task_manager, 100, 10, skip=0, batch_size=8)
