import logging
import typing as tp
from os import path as osp
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as albu

from src.config import Config
from src.tools import read_rgb_img

logger = logging.getLogger(__name__)

class BarcodeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: albu.Compose, config: Config):
        self._df = df
        self._transforms = transforms
        self._config = config

    def __getitem__(self, idx: int):
        imp_path = osp.join(self._config.train_images_path, self._df.iloc[idx][0])  # noqa: WPS221
        image = read_rgb_img(imp_path)
        transformed = self._transforms(image=image)
        image = transformed["image"] / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        target = self._df.iloc[idx][1]
        target = target.replace(' ', '')
        target = list(map(int, target))
        target = [n + 1 for n in target]
        return {"image": image, "target": (target, len(target))}

    def __len__(self):
        return len(self._df)


def get_loaders(                                                           # noqa: WPS210
    config: Config,
) -> tp.Tuple[tp.OrderedDict[str, DataLoader], tp.Dict[str, DataLoader]]:  # noqa: WPS221
    train_dataset, valid_dataset, test_dataset = get_datasets(config)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return {"train": train_loader, "valid": valid_loader}, {"infer": test_loader}


def get_datasets(config: Config) -> tp.Tuple[Dataset, Dataset, Dataset]:  # noqa: WPS210
    train_df, valid_df, test_df = _get_dataframes(config)
    train_augs = config.train_augmentation
    test_augs = config.val_augmentation
    train_dataset = BarcodeDataset(train_df, transforms=train_augs, config=config)
    valid_dataset = BarcodeDataset(valid_df, transforms=test_augs, config=config)
    test_dataset = BarcodeDataset(test_df, transforms=test_augs, config=config)
    return train_dataset, valid_dataset, test_dataset


def _get_dataframes(config: Config) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: WPS210
    df = pd.read_csv(config.df_path, sep="\t")
    train_df, other_df = train_test_split(df, train_size=config.train_size, random_state=config.seed, shuffle=True)
    valid_df, test_df = train_test_split(other_df, train_size=0.5, shuffle=False)
    logger.info(f"Train dataset: {len(train_df)}")
    logger.info(f"Valid dataset: {len(valid_df)}")
    logger.info(f"Test dataset: {len(test_df)}")

    return train_df, valid_df, test_df
