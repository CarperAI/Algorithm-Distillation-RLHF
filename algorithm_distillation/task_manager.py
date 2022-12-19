import random
from typing import List

from .task import Task


class TaskManager:
    """
    This is the controller for a set of tasks (each inherit the `Task` class).
    Currently, all the methods has trivial behaviors (just pass on to one of the `Task` objects).
    But I expect more sophisticated behaviors when we move on (also don't forget to remove and update
    this docstring when we do so!).
    """

    def __init__(self, tasks: List[Task]):
        if not tasks:
            raise ValueError("The task list cannot be empty.")
        for task in tasks[1:]:
            for st in ["obs", "act"]:
                if getattr(task, f"{st}_dim") != getattr(tasks[0], f"{st}_dim"):
                    raise ValueError(f"All tasks must have the same {st}_dim.")

        self.tasks = tasks

    def train(self, steps: int):
        """
        Train `steps` amount of steps for all the tasks.

        :param steps: the amount of gradient steps to train
        :return: None
        """
        for task in self.tasks:
            task.train(steps)

    def sample_history(self, length: int, skip: int = 0) -> tuple:
        """
        Choose a random task and sample a given amount of history.
        TODO: in the AD paper, it samples sequences for each task and put together as a batch.
            Here I only sample randomly. Will create another version like the paper's description.

        :param length: length: the length of the history.
        :param skip: (Optional) skip certain amount of steps between two states.
        :return: a tuple of (observations, actions, rewards). Each is a tensor.
        """
        task = random.choice(self.tasks)
        return task.sample_history(length, skip=skip)
