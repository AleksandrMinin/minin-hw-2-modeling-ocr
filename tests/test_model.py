import torch

from src.config import OUTPUT_LEN, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH
from src.crnn import CRNN


def test_forward(config_test):
    model = CRNN(**config_test.model_kwargs)
    dummy_input = torch.ones(1, 3, IMG_HEIGHT, IMG_WIDTH)
    model_ouput = model(dummy_input)
    assert model_ouput.shape == (OUTPUT_LEN, 1, NUM_CLASSES)
