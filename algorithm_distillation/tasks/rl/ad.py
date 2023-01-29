import abc

import torch

from algorithm_distillation.models.ad_transformer import ADTransformer

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
        **config
    ):
        pass


class GymAD(AlgorithmDistillation):
    def train(
        self,
        task_manager: TaskManager,
        steps: int,
        length: int,
        skip: int,
        batch_size: int,
        **config
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

        self.model.eval()  # By default, set to eval mode outside of training.

        print(losses)

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
                    [sample[0] for sample in samples], dtype=torch.float, device=device
                ),  # observations
                torch.tensor(
                    [sample[1] for sample in samples], dtype=torch.long, device=device
                ),  # actions
                torch.tensor(
                    [sample[2] for sample in samples], dtype=torch.float, device=device
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
        x = torch.nn.functional.log_softmax(x)  # (b, length, action_num)
        return -torch.take_along_dim(x, y, dim=len(y.shape) - 1).sum(-1).mean()
