import logging
import typing as tp
import torch
from catalyst import dl
from catalyst.core.callback import Callback
from catalyst.engines.torch import CPUEngine, GPUEngine

from src.config import config
from src.base_config import Config
from src.tools import set_global_seed, get_code
from src.dataset import get_loaders
from src.loggers import ClearMLLogger
from src.crnn import CRNN
from src.constants import BEST_GEN_MODEL
from src.runners import SupervisedOCRRunner


def get_base_callbacks() -> tp.List[Callback]:
    return [
        dl.CriterionCallback(
            input_key=dict(output="log_probs", output_size="input_lengths"),           
            target_key=dict(target="targets", target_len="target_lengths"),     
            metric_key="loss",
            criterion_key="ctc_loss_fn",
        ),
        dl.CriterionCallback(
            input_key="pred_codes",
            target_key="target",
            metric_key="accuracy",
            criterion_key="acc_fn",
        ),
    ]


def get_train_callbacks() -> tp.List[Callback]:
    callbacks = get_base_callbacks()
    callbacks.extend(
        [
            dl.CheckpointCallback(
                logdir=config.checkpoints_dir,
                loader_key="valid",
                metric_key=config.valid_metric,
                minimize=config.minimize_metric,
            ),
        ],
    )
    return callbacks


def train(
    config: Config, 
    clearml: bool = True, 
    pretrained: bool = False,
):
    loaders, infer_loader = get_loaders(config)  
    model = CRNN(**config.model_kwargs)
    if pretrained:
        state_dict = torch.load(BEST_GEN_MODEL)
        model.load_state_dict(state_dict)
    
    optimizer = config.optimizer(params=model.parameters(), **config.optimizer_kwargs)
    scheduler = config.scheduler(optimizer=optimizer, **config.scheduler_kwargs)
    if clearml:
        clearml_logger = ClearMLLogger(config, config.experiment_name)
        loggers={"_clearml": clearml_logger}
    else:
        loggers = None

    if torch.cuda.is_available():
        engine = GPUEngine()
    else:
        engine = CPUEngine()

    runner = SupervisedOCRRunner(
        input_key="image", 
        target_key="target", 
        output_key="output",
    )

    criterion = {
        "ctc_loss_fn": config.ctc_loss,
        "acc_fn": config.acc_fn,
    }

    runner.train(
        model=model,
        engine=engine,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=get_train_callbacks(),
        loggers=loggers,
        num_epochs=config.n_epochs,
        valid_loader="valid",
        valid_metric=config.valid_metric,
        minimize_valid_metric=config.minimize_metric,
        seed=config.seed,
        verbose=True,
        load_best_on_end=True,
    )
    
    runner = SupervisedOCRRunner(
        input_key="image", 
        target_key="target", 
        output_key="output",
    )
    runner.train(
        model=model,
        engine=engine,
        criterion=criterion,
        loaders=infer_loader,
        callbacks=get_base_callbacks(),
        loggers=None,
        valid_loader="infer",
        verbose=True,
    )
    metrics = dict(runner.loader_metrics)

    if clearml:
        clearml_logger.log_metrics(metrics, scope="loader", runner=runner, infer=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    set_global_seed(config.seed)
    train(config)
