"""Utilities for PyTorch Lightning DataModule.

Todo:
    * Add usage (Example Section).
    * Add unit-test.
"""
import copy
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

logger = getLogger(__name__)


class OpenPackBaseDataModule(pl.LightningDataModule):
    """Base class of PyTorch Lightning DataModule.
    A datamodule is a shareable, reusable class that encapsulates all the steps needed to process
    data:

    Attributes:
        dataset_class (torch.utils.data.Dataset): dataset class. this variable is call to create
            dataset instances.
        cfg (DictConfig): config object. The all parameters used to initialuze dataset class should
            be included in this object.
        batch_size (int): batch size.
        debug (bool): If True, enable debug mode.
    """
    dataset_class: torch.utils.data.Dataset

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.debug = cfg.debug
        if cfg.debug:
            self.batch_size = cfg.train.debug.batch_size
        else:
            self.batch_size = cfg.train.batch_size

    def get_kwargs_for_datasets(self, stage: Optional[str] = None) -> Dict:
        """Build a kwargs to initialize dataset class. This method is called in ``setup()``.

        Args:
            stage (str, optional): dataset type. {train, validate, test, submission}.

        Example:

            ::

                def get_kwargs_for_datasets(self) -> Dict:
                    kwargs = {
                        "window": self.cfg.train.window,
                        "debug": self.cfg.debug,
                    }
                    return kwargs

        Returns:
            Dict:
        """
        raise NotImplementedError()

    def _init_datasets(
        self,
        user_session: Tuple[int, int],
        kwargs: Dict,
    ) -> Dict[str, torch.utils.data.Dataset]:
        """Returns list of initialized dataset object.

        Args:
            rootdir (Path): _description_
            user_session (Tuple[int, int]): _description_
            kwargs (Dict): _description_

        Returns:
            Dict[str, torch.utils.data.Dataset]: dataset objects
        """
        datasets = dict()
        for user, session in user_session:
            key = f"{user}-{session}"
            datasets[key] = self.dataset_class(
                copy.deepcopy(self.cfg), [(user, session)], **kwargs)
        return datasets

    def setup(self, stage: Optional[str] = None) -> None:
        split = self.cfg.dataset.split

        if stage in (None, "fit"):
            kwargs = self.get_kwargs_for_datasets(stage="train")
            self.op_train = self.dataset_class(self.cfg, split.train, **kwargs)
        else:
            self.op_train = None

        if stage in (None, "fit", "validate"):
            kwargs = self.get_kwargs_for_datasets(stage="validate")
            self.op_val = self._init_datasets(split.val, kwargs)
        else:
            self.op_val = None

        if stage in (None, "test"):
            kwargs = self.get_kwargs_for_datasets(stage="test")
            self.op_test = self._init_datasets(split.test, kwargs)
        else:
            self.op_test = None

        if stage in (None, "submission"):
            kwargs = self.get_kwargs_for_datasets(stage="submission")
            kwargs.update({"submission": True})
            self.op_submission = self._init_datasets(split.submission, kwargs)
        elif stage == "test-on-submission":
            kwargs = self.get_kwargs_for_datasets(stage="submission")
            self.op_submission = self._init_datasets(split.submission, kwargs)
        else:
            self.op_submission = None

        logger.info(f"dataset[train]: {self.op_train}")
        logger.info(f"dataset[val]: {self.op_val}")
        logger.info(f"dataset[test]: {self.op_test}")
        logger.info(f"dataset[submission]: {self.op_submission}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.op_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers)

    def val_dataloader(self) -> List[DataLoader]:
        dataloaders = []
        for key, dataset in self.op_val.items():
            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.train.num_workers)
            )
        return dataloaders

    def test_dataloader(self) -> List[DataLoader]:
        dataloaders = []
        for key, dataset in self.op_test.items():
            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.train.num_workers)
            )
        return dataloaders

    def submission_dataloader(self) -> List[DataLoader]:
        dataloaders = []
        for key, dataset in self.op_submission.items():
            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.train.num_workers)
            )
        return dataloaders





class IKEADataModule(pl.LightningDataModule):
    """Base class of PyTorch Lightning DataModule.
    A datamodule is a shareable, reusable class that encapsulates all the steps needed to process
    data:

    Attributes:
        dataset_class (torch.utils.data.Dataset): dataset class. this variable is call to create
            dataset instances.
        cfg (DictConfig): config object. The all parameters used to initialuze dataset class should
            be included in this object.
        batch_size (int): batch size.
        debug (bool): If True, enable debug mode.
    """
    dataset_class: torch.utils.data.Dataset

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.debug = cfg.debug
        if cfg.debug:
            self.batch_size = cfg.train.debug.batch_size
        else:
            self.batch_size = cfg.train.batch_size

    def get_kwargs_for_datasets(self, stage: Optional[str] = None) -> Dict:
        """Build a kwargs to initialize dataset class. This method is called in ``setup()``.

        Args:
            stage (str, optional): dataset type. {train, validate, test, submission}.

        Example:

            ::

                def get_kwargs_for_datasets(self) -> Dict:
                    kwargs = {
                        "window": self.cfg.train.window,
                        "debug": self.cfg.debug,
                    }
                    return kwargs

        Returns:
            Dict:
        """
        raise NotImplementedError()

    def _init_datasets(
        self,
        user_session: Tuple[int, int],
        kwargs: Dict,
    ) -> Dict[str, torch.utils.data.Dataset]:
        """Returns list of initialized dataset object.

        Args:
            rootdir (Path): _description_
            user_session (Tuple[int, int]): _description_
            kwargs (Dict): _description_

        Returns:
            Dict[str, torch.utils.data.Dataset]: dataset objects
        """
        datasets = dict()
        for user, session in user_session:
            key = f"{user}-{session}"
            datasets[key] = self.dataset_class(
                copy.deepcopy(self.cfg), [(user, session)], **kwargs)
        return datasets

    def setup(self, stage: Optional[str] = None) -> None:
        split = self.cfg.dataset.split

        if stage in (None, "fit"):
            # kwargs = self.get_kwargs_for_datasets(stage="train")
            self.op_train = self.dataset_class(phase = 'train')
        else:
            self.op_train = None

        if stage in (None, "fit", "validate"):
            # kwargs = self.get_kwargs_for_datasets(stage="validate")
            self.op_val = self.dataset_class(phase = 'test')
        else:
            self.op_val = None

        if stage in (None, "test"):
            # kwargs = self.get_kwargs_for_datasets(stage="test")
            self.op_test = self.dataset_class(phase = 'test')
        else:
            self.op_test = None

        # if stage in (None, "submission"):
        #     kwargs = self.get_kwargs_for_datasets(stage="submission")
        #     kwargs.update({"submission": True})
        #     self.op_submission = self._init_datasets(split.submission, kwargs)
        # elif stage == "test-on-submission":
        #     kwargs = self.get_kwargs_for_datasets(stage="submission")
        #     self.op_submission = self._init_datasets(split.submission, kwargs)
        # else:
        #     self.op_submission = None

        logger.info(f"dataset[train]: {self.op_train}")
        logger.info(f"dataset[val]: {self.op_val}")
        logger.info(f"dataset[test]: {self.op_test}")
        # logger.info(f"dataset[submission]: {self.op_submission}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.op_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers)


    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.op_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers)


    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.op_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers)

    # def val_dataloader(self) -> List[DataLoader]:
    #     dataloaders = []
    #     for key, dataset in self.op_val.items():
    #         dataloaders.append(
    #             DataLoader(
    #                 dataset,
    #                 batch_size=self.batch_size,
    #                 shuffle=False,
    #                 num_workers=self.cfg.train.num_workers)
    #         )
    #     return dataloaders
    #
    # def test_dataloader(self) -> List[DataLoader]:
    #     dataloaders = []
    #     for key, dataset in self.op_test.items():
    #         dataloaders.append(
    #             DataLoader(
    #                 dataset,
    #                 batch_size=self.batch_size,
    #                 shuffle=False,
    #                 num_workers=self.cfg.train.num_workers)
    #         )
    #     return dataloaders

    # def submission_dataloader(self) -> List[DataLoader]:
    #     dataloaders = []
    #     for key, dataset in self.op_submission.items():
    #         dataloaders.append(
    #             DataLoader(
    #                 dataset,
    #                 batch_size=self.batch_size,
    #                 shuffle=False,
    #                 num_workers=self.cfg.train.num_workers)
    #         )
    #     return dataloaders
