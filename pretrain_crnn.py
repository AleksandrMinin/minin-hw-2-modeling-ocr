import logging
import typing as tp
import torch
from catalyst import dl
from catalyst.core.callback import Callback
from catalyst.engines.torch import CPUEngine, GPUEngine

from src.config import config
from src.base_config import Config
from src.tools import set_global_seed, get_code
from src.loggers import ClearMLLogger
from src.crnn import CRNN
from src.runners import SupervisedOCRRunner
from generator.dataset_generator import GenBarcodeDataset


train_callbacks = [
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
    dl.CheckpointCallback(
        logdir=config.gen_checkpoints_dir,
        loader_key="valid",
        metric_key=config.valid_metric,
        minimize=config.minimize_metric,
    ),
    dl.EarlyStoppingCallback(
        patience=config.early_stop_patience,
        loader_key="valid",
        metric_key=config.valid_metric,
        minimize=config.minimize_metric,
    ),
]


def train(
    config: Config, 
    clearml: bool = True,
    train_size: int = 300,
    valid_size: int = 100,
    num_epochs: int = 100
):
    train_loader = torch.utils.data.DataLoader(
        GenBarcodeDataset(
            epoch_size=train_size,
            backgrounds_dir=config.backgrounds_dir,
            transforms=config.train_augmentation,
        ), 
        batch_size=config.batch_size,
    )
    valid_loader = torch.utils.data.DataLoader(
        GenBarcodeDataset(
            epoch_size=valid_size,
            backgrounds_dir=config.backgrounds_dir,
            transforms=config.val_augmentation,
        ), 
        batch_size=config.batch_size,
    )
    loaders = {"train": train_loader, "valid": valid_loader}
    model = CRNN(**config.model_kwargs)
    optimizer = config.optimizer(params=model.parameters(), lr=0.0001)
    if clearml:
        clearml_logger = ClearMLLogger(config, config.pretrain_name)
        loggers={"_clearml": clearml_logger}
    else:
        loggers = None

    if torch.cuda.is_available():
        engine = GPUEngine()
    else:
        engine = CPUEngine()
        
    criterion = {
        "ctc_loss_fn": config.ctc_loss,
        "acc_fn": config.acc_fn,
    }
    runner = SupervisedOCRRunner(
        input_key="image", 
        target_key="target", 
        output_key="output",
    )

    runner.train(
        model=model,
        engine=engine,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        callbacks=train_callbacks,
        loggers=loggers,
        num_epochs=num_epochs,
        valid_loader="valid",
        valid_metric=config.valid_metric,
        minimize_valid_metric=config.minimize_metric,
        seed=config.seed,
        verbose=True,
        load_best_on_end=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    set_global_seed(config.seed)
    train(config)
