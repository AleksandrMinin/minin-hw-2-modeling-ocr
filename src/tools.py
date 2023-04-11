import os
import random
import cv2
import numpy as np
import torch

from src.config import config


def read_rgb_img(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image does not exist: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def set_global_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_code(output: torch.Tensor) -> torch.Tensor:  # noqa: WPS231
    preds = torch.argmax(output, dim=2).permute(1, 0)
    preds = preds.detach().cpu().numpy()
    pred_codes = []
    for pred in preds:
        pred_code = []
        for i in range(len(pred)):  # noqa: WPS518
            if pred[i] != 0:
                if i == 0 or (pred[i - 1] != pred[i]):
                    pred_code.append(pred[i])
        pred_code = pred_code[:config.max_code_len]
        pred_code = pred_code + [0] * (config.max_code_len - len(pred_code))
        pred_codes.append(pred_code)
    return torch.LongTensor(pred_codes)
