import typing as tp
from catalyst.core.logger import ILogger
from clearml import Task
import pandas as pd

from src.config import Config


class ClearMLLogger(ILogger):
    def __init__(
        self,
        config: Config,
        experiment_name: str,
        log_batch_metrics: bool = False,
        log_epoch_metrics: bool = True,
    ):
        super().__init__(
            log_batch_metrics=log_batch_metrics,
            log_epoch_metrics=log_epoch_metrics,
        )
        self.metrics_to_log = config.log_metrics
        self.metrics_to_log.extend(["loss", "lr"])

        task = Task.init(
            project_name=config.project_name,
            task_name=experiment_name,
        )
        task.connect(config.to_dict())
        self.clearml_logger = task.get_logger()

    @property
    def logger(self):
        return self.clearml_logger

    def log_metrics(
        self,
        metrics: tp.Dict[str, float],
        scope: str,
        runner: "IRunner",  # noqa: F821
        infer: bool = False,
    ):
        step = runner.sample_step if self.log_batch_metrics else runner.epoch_step

        if scope == "loader" and self.log_epoch_metrics:
            if infer:
                self._log_infer_metrics(
                    metrics=metrics,
                )
            else:
                self._log_train_metrics(
                    metrics=metrics,
                    step=step,
                    loader_key=runner.loader_key,
                )

    def flush_log(self):
        self.logger.flush()

    def _report_scalar(self, title: str, mode: str, value: float, epoch: int):  # noqa: WPS110
        self.logger.report_scalar(
            title=title,
            series=mode,
            value=value,
            iteration=epoch,
        )

    def _log_train_metrics(self, metrics: tp.Dict[str, float], step: int, loader_key: str):
        log_keys = [k for log_m in self.metrics_to_log for k in metrics.keys() if log_m in k]
        for k in log_keys:
            self._report_scalar(k, loader_key, metrics[k], step)

    def _log_infer_metrics(self, metrics: tp.Dict[str, float]):
        test_results = pd.DataFrame.from_dict(metrics, orient="index", columns=[""])
        test_results = test_results.astype("float")
        self.logger.report_table(title="Test Results", series="Test Results", iteration=0, table_plot=test_results)
