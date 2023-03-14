import pytest

from src.config import config


@pytest.fixture()
def config_test():
    config.n_epochs = 1
    config.df_path = './tests/data_for_tests/num_under_barcodes.tsv'
    config.train_images_path = './tests/data_for_tests/test_barcodes'
    config.train_size = 0.5
    config.batch_size = 1
    return config
