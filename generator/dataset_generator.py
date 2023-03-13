import os
import random
from typing import Optional, Tuple
import numpy as np
import torch
import albumentations as albu
import cv2
import barcode
from barcode.writer import ImageWriter

from generator.src.generator import overlay


def generate_barcode(
    value: str, 
    backgrounds_dir: str, 
    transforms: Optional[albu.BasicTransform] = None
) -> Tuple[np.ndarray, str]:

    random_background = random.choice(os.listdir(backgrounds_dir))
    background = cv2.imread(os.path.join(backgrounds_dir, random_background))

    ean = barcode.Code128(value, writer=ImageWriter())
    barcode_mask = ean.render()
    fullcode = ean.get_fullcode()
    
    barcode_mask = 255 - cv2.cvtColor(np.asarray(barcode_mask), cv2.COLOR_RGB2GRAY)
    h, w = barcode_mask.shape
    barcode_image = np.zeros((h, w, 3))
    image_res, mask = overlay(
        background, 
        barcode_image, 
        barcode_mask,
        transforms=transforms,
    )
    
    return image_res, fullcode


class GenBarcodeDataset(torch.utils.data.Dataset):
    def __init__(self, epoch_size, backgrounds_dir, transforms = None):
        self.epoch_size = epoch_size
        self.backgrounds_dir = backgrounds_dir
        self.transforms = transforms
    
    def __getitem__(self, i: int):
        # длина штрих-кода от 8 до 13 цифр
        n_digits = np.random.randint(7, 13)
        min_num = float('1e' + str(n_digits))
        max_num = float('1e' + str(n_digits + 1))
        value = str(np.random.randint(min_num, max_num))

        image, fullcode = generate_barcode(value, 
                                           self.backgrounds_dir, 
                                           self.transforms)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        fullcode = list(map(int, fullcode))
        # прибавляем 1 ко всем цифрам штрих-кода, чтобы при обучении разделителем был 0
        fullcode = [n + 1 for n in fullcode]
        
        return {"image": image, "target": (fullcode, len(fullcode))}
    
    def __len__(self):
        return self.epoch_size
