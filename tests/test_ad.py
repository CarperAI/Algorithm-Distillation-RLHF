import gym
import pytest

from algorithm_distillation import GymAD
from algorithm_distillation.tasks.rl import GymTask, TaskManager
from algorithm_distillation.models import GPT2AD
from algorithm_distillation.models.sb3_util import CustomLogger, configure


@pytest.mark.parametrize("policy", ['DQN', 'PPO', 'A2C'])
@pytest.mark.parametrize("env_name", ['FrozenLake-v1', 'CartPole-v1'])
def test_ad(policy: str, env_name: str):
    env = gym.make(env_name)

    if policy in ['DQN', 'TD3']:
        config = {'learning_starts': 10,
                  'policy': 'MlpPolicy'
                  }
    else:
        config = {'n_steps': 30,
                  'batch_size': 10,
                  'policy': 'MlpPolicy'
                  }
    if policy == 'A2C':
        config.pop('batch_size')

    task = GymTask(env, policy, buffer_size=100, config=config)
    model = GPT2AD(task.obs_dim, task.act_dim, 12, max_step_len=16)

    # Inject a customized logger
    logger = configure(None, None, CustomLogger)
    task.agent.set_logger(logger)

    task_manager = TaskManager([task])
    task_manager.train(100, log_interval=1)

    assert 'history_value' in task.agent.logger.__dir__()
    # 100 total time-steps, but training only happens upon the finish of an episode. We don't know how many gradient
    # steps are trained, but we are sure it is nonzero.
    loss_key = 'train/policy_loss' if policy == 'A2C' else 'train/loss'
    assert len(task.agent.logger.history_value[loss_key]) != 0
    # But we are sure that rollout happens 100 times.
    if policy in ['DQN']:
        assert len(task.agent.logger.history_value['rollout/exploration_rate']) == 100
    else:
        # Only logged once, because training step (100) equals the rollout length.
        # SB3 off-policy algorithms first collect rollouts accumulating `n_steps` until rollout buffer is full,
        # and then check if training continues.
        assert len(task.agent.logger.history_value['rollout/ep_rew_mean']) > 0  # A2C steps not deterministic??

    ad = GymAD(model)
    ad.train(task_manager, 100, 10, skip=0, batch_size=8, verbose=1)

    obs, act, rew, term = ad.rollout(task, 100, 0, verbose=1)
    assert obs.size(0) == 100
    obs, act, rew, term = ad.rollout(task, 100, 2, verbose=1)
    assert obs.size(0) == 100
