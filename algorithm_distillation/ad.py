import abc

import numpy as np
import torch

from algorithm_distillation.models.ad_transformer import ADTransformer

from .task import GymTask
from .task_manager import TaskManager


class AlgorithmDistillation(abc.ABC):
    """
    This is the base class of controllers of algorithm distillation training.
    """

    def __init__(self, model: ADTransformer):
        self.model = model

    @abc.abstractmethod
    def train(
        self,
        task_manager: TaskManager,
        steps: int,
        length: int,
        skip: int,
        batch_size: int,
        **config,
    ):
        pass

    @abc.abstractmethod
    def rollout(self, task, steps: int, skip: int) -> tuple:
        pass


class GymAD(AlgorithmDistillation):
    def train(
        self,
        task_manager: TaskManager,
        steps: int,
        length: int,
        skip: int,
        batch_size: int,
        **config,
    ):
        """
        Collect samples and train `steps` amount of gradient steps.

        :param task_manager: the controller that controls a collection of tasks.
        :param steps: the amount of gradient steps to train.
        :param length: the step-length of sampled sequences (not the sequence length which is 3x).
        :param skip: the amount of states to skip between two consecutive ones.
        :param batch_size: the batch size.
        :param config: the extra config that goes into transformer training.
        :return: None
        """
        # We implement a PyTorch training loop.
        # Use GPU if exists.
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        data_iter = self._get_data_iter(
            steps, batch_size, task_manager, length, skip, device=device
        )
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        self.model.train()  # Set to train mode so that dropout and batch norm would update.
        losses = []
        for step, sample in enumerate(data_iter):
            optimizer.zero_grad()
            obs, actions, rewards = sample
            one_hot_actions = torch.nn.functional.one_hot(
                actions.squeeze(-1), num_classes=self.model.act_dim
            ).type(torch.float)
            loss = self._compute_loss(
                self.model(obs, one_hot_actions, rewards, action_only=True), actions
            )
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        self.model.eval()  # By default, set to eval mode outside training.

    def rollout(self, task: GymTask, steps: int, skip: int) -> tuple:
        """
        Roll out for `steps` amount of steps (ignore the policy embedded in `task` and only uses its _env).

        :param task: the task to perform rollout on.
        :param steps: the amount of steps to roll out.
        :param skip: the amount of steps to skip (normally should be the same as `skip` during training).
        :return: the full sequences (observations, actions, rewards), each of length `steps`.
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(device)

        for st in ["obs", "act"]:
            if getattr(task, f"{st}_dim") != getattr(self.model, f"{st}_dim"):
                raise ValueError(
                    f"The task must have observation dimension {self.model.obs_dim}"
                )
        env = task.env
        observations = torch.zeros(
            (steps, task.obs_dim), device=device, dtype=torch.float
        )

        # Predicted action logits
        action_logits = torch.zeros(
            (steps, task.act_dim), device=device, dtype=torch.float
        )
        # The actual actions taken (argmax of action_logits)
        actions = torch.zeros((steps,), device=device, dtype=torch.long)
        # The actual one-hot encoded actions (nn.one_hot of actions)
        actions_one_hot = torch.zeros(
            (steps, task.act_dim), device=device, dtype=torch.float
        )

        rewards = torch.zeros((steps, 1), device=device, dtype=torch.float)

        obs, done = None, True
        for step in range(steps):
            if done:
                obs, done = (
                    torch.tensor(
                        task.obs_post_process(np.array([env.reset()])),
                        device=device,
                        dtype=torch.float,
                    ),
                    False,
                )

            # TODO: can be optimized using cache
            with torch.inference_mode():
                action_logits[step] = self.model(
                    None if step < skip + 1 else observations[None, : step : skip + 1],
                    None
                    if step < skip + 1
                    else actions_one_hot[None, : step : skip + 1],
                    None if step < skip + 1 else rewards[None, : step : skip + 1],
                    current_obs=obs[None, 0],
                    action_only=True,
                )[0, step]
            actions[step] = torch.argmax(action_logits[step]).type(torch.long)
            actions_one_hot[step] = torch.nn.functional.one_hot(
                actions[step], num_classes=task.act_dim
            ).type(torch.float)

            observations[step] = obs[None, 0]
            obs, rew, done, _ = env.step(actions[step].item())
            obs = torch.tensor(
                task.obs_post_process(np.array([obs])), device=device, dtype=torch.float
            )
            rew = torch.tensor(
                task.rew_post_process(np.array([rew])), device=device, dtype=torch.float
            )
            rewards[step] = rew[0]

        return observations, actions, rewards

    @staticmethod
    def _get_data_iter(
        steps: int, batch_size, task_manager, length, skip, device=torch.device("cpu")
    ):
        for _ in range(steps):
            samples = []
            for _ in range(batch_size):
                samples.append(task_manager.sample_history(length, skip))

            yield (
                torch.tensor(
                    np.array([sample[0] for sample in samples]),
                    dtype=torch.float,
                    device=device,
                ),  # observations
                torch.tensor(
                    np.array([sample[1] for sample in samples]),
                    dtype=torch.long,
                    device=device,
                ),  # actions
                torch.tensor(
                    np.array([sample[2] for sample in samples]),
                    dtype=torch.float,
                    device=device,
                ),  # rewards
            )

    @staticmethod
    def _compute_loss(x, y) -> torch.Tensor:
        """
        :param x: action logits.
        :param y: the target action.
        :return: the NLL loss.
        """
        assert y.dtype == torch.long
        assert x.shape[:-1] + (1,) == y.shape
        x = torch.nn.functional.log_softmax(x, dim=-1)  # (b, length, action_num)
        return -torch.take_along_dim(x, y, dim=len(y.shape) - 1).sum(-1).mean()
