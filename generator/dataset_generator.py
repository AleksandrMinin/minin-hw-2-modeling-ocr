import barcode # pip install git+https://github.com/WhyNotHugo/python-barcode
import numpy as np
import torch
import albumentations as albu

from barcode.writer import ImageWriter


class BarcodeDataset(torch.utils.data.Dataset):
    def __init__(self, epoch_size, vocab):
        self.epoch_size = epoch_size
    
    def __getitem__(self, i: int):
        value = str(np.random.randint(10e11, 10e12))
        #value = '12345678901' # учим на одной картинке, чтобы убедиться, что сеть переобучается

        ean = barcode.UPCA(value, writer=ImageWriter())
        bcode = ean.get_fullcode()
        bcode = list(map(int, bcode))
        
        image = np.asarray(ean.render()) / 255.0
        transforms = albu.Resize(height=280, width=2000)
        image = transforms(image=image)['image']
        image = torch.FloatTensor(image).permute(2, 0, 1)
        
        return image, torch.LongTensor(bcode), torch.LongTensor([len(bcode)])
    
    def __len__(self):
        return self.epoch_size
