import random
from typing import Optional, Tuple

import cv2
import numpy as np

from albumentations.core.transforms_interface import DualTransform, ImageColorType
from albumentations.augmentations.geometric.functional import pad_with_params


class OCRTransform(DualTransform):
    def __init__(
        self,
        height: int,
        width: int,
        position: str = "left",
        border_mode: int = cv2.BORDER_CONSTANT,
        value: Optional[ImageColorType] = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super(OCRTransform, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.position = position
        self.border_mode = border_mode
        self.value = value

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img_height = self.height
        img_width = (img_height * img.shape[1]) // img.shape[0]
        img = cv2.resize(img, dsize=(img_width, img_height), interpolation=cv2.INTER_CUBIC)
        params.update({"rows": img_height, "cols": img_width})
        pad_params = self.__get_pad_params(params)
        return pad_with_params(
            img,
            h_pad_top=0,
            h_pad_bottom=0,
            border_mode=self.border_mode,
            value=self.value,
            **pad_params,
        )

    def get_transform_init_args_names(self):
        return (
            "height",
            "width",
            "position",
            "border_mode",
            "value",
        )

    def __get_pad_params(self, params):
        pad_params = {}
        cols = params["cols"]

        if cols < self.width:
            w_pad_left = int((self.width - cols) / 2.0)
            w_pad_right = self.width - cols - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0

        w_pad_left, w_pad_right = self.__update_position_params(w_left=w_pad_left, w_right=w_pad_right)

        pad_params.update({"w_pad_left": w_pad_left, "w_pad_right": w_pad_right})

        return pad_params

    def __update_position_params(self, w_left: int, w_right: int) -> Tuple[int, int]:
        if self.position == "left":
            w_right += w_left
            w_left = 0

        elif self.position == "random":
            w_pad = w_left + w_right
            w_left = random.randint(0, w_pad)
            w_right = w_pad - w_left
        elif self.position != "center":
            raise ValueError(f"Incorrect argument position={self.position}. Allowed position=left, center or random.")

        return w_left, w_right
