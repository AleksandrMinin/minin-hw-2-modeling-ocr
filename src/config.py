from datetime import datetime
import albumentations as albu
import torch
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import MultiStepLR

from src.base_config import Config
from src.constants import DF_PATH, BACKGROUNDS_DIR, TRAIN_IMAGES_PATH
from src.losses import my_accuracy
import cv2

EXP_NUM = "5"
NUM_CLASSES = 11
N_EPOCHS = 60
BATCH_SIZE = 8
TRAIN_SIZE = 0.8
OUTPUT_LEN = 20
IMG_HEIGHT = 256
IMG_WIDTH = 2048
PAD = 100


date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

train_augmentation = albu.Compose(
    [
        albu.ToGray(p=1),
        albu.RandomCropFromBorders(
            crop_left=0,
            crop_right=0,
            crop_top=0.7,
            crop_bottom=0,
        ),
        albu.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        albu.PadIfNeeded(
            min_height=IMG_HEIGHT + PAD,
            min_width=IMG_WIDTH + PAD,
            border_mode=cv2.BORDER_CONSTANT,
            p=1,
        ),
        albu.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        albu.MotionBlur(),
        albu.GaussianBlur(),
        albu.ImageCompression(10, 30),
    ])

val_augmentation = albu.Compose(
    [
        albu.ToGray(p=1),
        albu.CropAndPad(percent=(-0.5, 0, 0, 0)),
        albu.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        albu.PadIfNeeded(
            min_height=IMG_HEIGHT + PAD,
            min_width=IMG_WIDTH + PAD,
            border_mode=cv2.BORDER_CONSTANT,
            p=1,
        ),
        albu.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
    ])


config = Config(
    num_workers=6,
    seed=42,
    max_code_len=14,
    ctc_loss=CTCLoss(zero_infinity=True),
    acc_fn=my_accuracy,
    device="cuda",
    optimizer=torch.optim.AdamW,
    optimizer_kwargs={
        "lr": 1e-4,
        "weight_decay": 1e-3,
    },
    scheduler=MultiStepLR,
    scheduler_kwargs={
        "milestones": [25, 35, 45],
        "gamma": 0.1,
    },
    df_path=DF_PATH,
    backgrounds_dir=BACKGROUNDS_DIR,
    train_images_path=TRAIN_IMAGES_PATH,
    train_size=TRAIN_SIZE,
    img_width=IMG_HEIGHT,
    img_height=IMG_WIDTH,
    train_augmentation=train_augmentation,
    val_augmentation=val_augmentation,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    early_stop_patience=25,
    model_kwargs={
        "cnn_backbone_name": "resnet18d",
        "cnn_backbone_pretrained": False,
        "cnn_output_size": 4096,
        "rnn_features_num": 2048,
        "rnn_dropout": 0.1,
        "rnn_bidirectional": True,
        "rnn_num_layers": 2,
        "num_classes": NUM_CLASSES,
    },
    log_metrics=["loss", "accuracy"],
    valid_metric="loss",
    minimize_metric=True,
    project_name="BboxOCR",
    experiment_name=f"experiment_{EXP_NUM}_{date_time}",
    pretrain_name=f"pretrain_{EXP_NUM}_{date_time}",
)
