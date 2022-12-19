import gym

from algorithm_distillation import GymTask, TaskManager, GymAD
from algorithm_distillation.models import GPT2AD
from algorithm_distillation.sb3_util import CustomLogger, configure


def test_ad():
    env = gym.make('FrozenLake-v1')
    config = {'learning_starts': 10,
              'buffer_size': 1000,
              'policy': 'MlpPolicy'
              }
    model = GPT2AD(env.observation_space.n, env.action_space.n, 12, max_ep_len=16)

    task = GymTask(env, 'DQN', config)
    # Inject a customized logger
    logger = configure(None, None, CustomLogger)
    task.agent.set_logger(logger)

    task_manager = TaskManager([task])
    task_manager.train(100)

    assert 'history_value' in task.agent.logger.__dir__()
    # 100 total time-steps, but training only happens upon the finish of an episode. We don't know how many gradient
    # steps are trained, but we are sure it is nonzero.
    assert len(task.agent.logger.history_value['train/loss']) != 0
    # But we are sure that rollout happens 100 times.
    assert len(task.agent.logger.history_value['rollout/exploration_rate']) == 100

    ad = GymAD(model)
    ad.train(task_manager, 100, 10, skip=0, batch_size=8)

    obs, act, rew = ad.rollout(task, 16, 0)
    print(obs, act, rew)
