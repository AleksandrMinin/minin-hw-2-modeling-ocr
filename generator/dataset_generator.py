import os
import random
from typing import Optional, Tuple
import numpy as np
from numpy.random import randint
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
    image, mask = overlay(
        background, 
        barcode_image, 
        barcode_mask,
    )
    image = transforms(image=image)['image'] 
    
    return image, fullcode


class GenBarcodeDataset(torch.utils.data.Dataset):
    def __init__(self, epoch_size, backgrounds_dir, transforms = None):
        self.epoch_size = epoch_size
        self.backgrounds_dir = backgrounds_dir
        self.transforms = transforms
    
    def __getitem__(self, i: int):
        # длина штрих-кода от 8 до 13 цифр
        value = [str(randint(10)) for _ in range(randint(8, 14))]
        value = ''.join(value)
        image, fullcode = generate_barcode(value, 
                                           self.backgrounds_dir, 
                                           self.transforms)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        fullcode = list(map(int, fullcode))
        # прибавляем 1 ко всем цифрам штрих-кода, чтобы при обучении разделителем был 0
        target = np.array([n + 1 for n in fullcode] + [0] * (14 - len(fullcode)))
        
        return {"image": image, "target": target, "target_len": len(fullcode)}
    
    def __len__(self):
        return self.epoch_size
