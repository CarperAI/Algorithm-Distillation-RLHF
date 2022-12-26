import datetime
import os
import tempfile
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Union

from stable_baselines3.common.logger import Logger, make_output_format


class CustomLogger(Logger):
    """
    A logger object can be plugged into an SB3 agent to record the metrics. Here we customize it to save metric
    histories. One can further customize it and implement, for example, the connection with wandb.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_value = defaultdict(list)
        self.history_mean_value = defaultdict(list)

    def record(
        self,
        key: str,
        value: Any,
        exclude: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        super().record(key, value, exclude)
        self.history_value[key].append(value)

    def record_mean(
        self,
        key: str,
        value: Any,
        exclude: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        super().record_mean(key, value, exclude)
        self.history_mean_value[key].append(self.name_to_value[key])


def configure(
    folder: Optional[str] = None,
    format_strings: Optional[List[str]] = None,
    logger_class=Logger,
) -> Logger:
    """
    Configure the current logger.
    (This is almost the same as SB3's logger configuration helper function except one line in the parameter and
     another line towards the end to allow for customized logger classes.)

    :param folder: the save location
        (if None, $SB3_LOGDIR, if still None, tempdir/SB3-[date & time])
    :param format_strings: the output logging format
        (if None, $SB3_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    :param logger_class: (Optional) the custom logger class.
    :return: The logger object.
    """
    if folder is None:
        folder = os.getenv("SB3_LOGDIR")
    if folder is None:
        folder = os.path.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S-%f"),
        )
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    log_suffix = ""
    if format_strings is None:
        format_strings = os.getenv("SB3_LOG_FORMAT", "stdout,log,csv").split(",")

    format_strings = list(filter(None, format_strings))
    output_formats = [make_output_format(f, folder, log_suffix) for f in format_strings]

    logger = logger_class(folder=folder, output_formats=output_formats)
    # Only print when some files will be saved
    if len(format_strings) > 0 and format_strings != ["stdout"]:
        logger.log(f"Logging to {folder}")
    return logger
