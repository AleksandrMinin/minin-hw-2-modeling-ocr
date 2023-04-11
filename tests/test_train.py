import os
import shutil

from train import train


def test_train(config_test):
    train(config_test, clearml=False, pretrained=False)
    checkpoints_dir = config_test.checkpoints_dir + '/'
    assert os.path.exists(checkpoints_dir)
    shutil.rmtree(checkpoints_dir)
    assert not os.path.exists(checkpoints_dir)
