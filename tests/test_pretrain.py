import os
import shutil

from pretrain_crnn import train


def test_train(config_test):
    train(
        config_test, 
        clearml=False,
        train_size=5,
        valid_size=2,
        num_epochs=1,
    )
    checkpoints_dir = config_test.gen_checkpoints_dir + '/'
    assert os.path.exists(checkpoints_dir)
    shutil.rmtree(checkpoints_dir)
    assert not os.path.exists(checkpoints_dir)
