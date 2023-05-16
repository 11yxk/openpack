from logging import getLogger
from pathlib import Path
from typing import Dict, Optional

import hydra
import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from openpack_toolkit import OPENPACK_OPERATIONS
from openpack_toolkit.codalab.operation_segmentation import (
    construct_submission_dict, eval_operation_segmentation_wrapper,
    make_submission_zipfile)

from my_model.C2F_TCN import C2F_TCN as TCN
import pickle
import loss_fn
import os
logger = getLogger(__name__)
optorch.configs.register_configs()
optorch.utils.reset_seed()

# ----------------------------------------------------------------------


def save_training_results(log: Dict, logdir: Path) -> None:
    # -- Save Model Outputs --
    df = pd.concat(
        [
            pd.DataFrame(log["train"]),
            pd.DataFrame(log["val"]),
        ],
        axis=1,
    )
    df.index.name = "epoch"

    path = Path(logdir, "training_log.csv")
    df.to_csv(path, index=True)
    logger.debug(f"Save training logs to {path}")
    print(df)


# ----------------------------------------------------------------------
class OpenPackImuDataModule(optorch.data.OpenPackBaseDataModule):
    dataset_class = optorch.data.datasets.OpenPackImu

    def get_kwargs_for_datasets(self, stage: Optional[str] = None) -> Dict:
        kwargs = {
            "window": self.cfg.train.window,
            "debug": self.cfg.debug,
        }
        return kwargs


class DeepConvLSTMLM(optorch.lightning.BaseLightningModule):

    def init_model(self, cfg: DictConfig) -> torch.nn.Module:

        in_ch = 12

        model = TCN(n_channels=in_ch, n_classes=len(OPENPACK_OPERATIONS))

        return model

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        x = batch["x"].to(device=self.device, dtype=torch.float)
        t = batch["t"].to(device=self.device, dtype=torch.long)
        b = batch["b"].to(device=self.device, dtype=torch.long)
        weight = torch.tensor([0.003, 1]).cuda()
        weight_ce = torch.nn.CrossEntropyLoss(weight=weight)
        tmse = loss_fn.TMSE()

        y_hat,boundary = self(x)

        loss = self.criterion(y_hat, t) + self.cfg.alpha * tmse(y_hat, t) + self.cfg.beta * weight_ce(boundary,b)

        acc = self.calc_accuracy(y_hat, t)

        return {"loss": loss, "acc": acc}

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        x = batch["x"].to(device=self.device, dtype=torch.float)
        t = batch["t"].to(device=self.device, dtype=torch.long)
        ts_unix = batch["ts"]

        y_hat,_ = self(x)

        outputs = dict(t=t, y=y_hat, unixtime=ts_unix)
        return outputs


# ----------------------------------------------------------------------


def train(cfg: DictConfig):
    device = torch.device("cuda")
    logdir = Path(cfg.path.logdir.rootdir)
    logger.debug(f"logdir = {logdir}")
    optk.utils.io.cleanup_dir(logdir, exclude="hydra")

    datamodule = OpenPackImuDataModule(cfg)
    plmodel = DeepConvLSTMLM(cfg)
    plmodel.to(dtype=torch.float, device=device)
    logger.info(plmodel)

    num_epoch = cfg.train.debug.epochs if cfg.debug else cfg.train.epochs

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        monitor=None,
    )

    # save all models
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     save_top_k=-1
    # )

    trainer = pl.Trainer(
        gpus=cfg.device,
        max_epochs=num_epoch,
        logger=False,  # disable logging module
        default_root_dir=logdir,
        enable_progress_bar=False,  # disable progress bar
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )
    logger.debug(f"logdir = {logdir}")

    logger.info(f"Start training for {num_epoch} epochs.")
    trainer.fit(plmodel, datamodule)
    logger.info("Finish training!")

    logger.debug(f"logdir = {logdir}")
    save_training_results(plmodel.log_dict, logdir)
    logger.debug(f"logdir = {logdir}")


def test(cfg: DictConfig, mode: str = "test"):
    assert mode in ("test", "submission", "test-on-submission")
    logger.debug(f"test() function is called with mode={mode}.")

    device = torch.device("cuda")
    logdir = Path(cfg.path.logdir.rootdir)

    datamodule = OpenPackImuDataModule(cfg)
    datamodule.setup(mode)

    ckpt_path = Path(logdir, "checkpoints", "last.ckpt")
    logger.info(f"load checkpoint from {ckpt_path}")
    plmodel = DeepConvLSTMLM.load_from_checkpoint(ckpt_path, cfg=cfg)
    plmodel.to(dtype=torch.float, device=device)

    trainer = pl.Trainer(
        gpus=cfg.device,
        logger=False,  # disable logging module
        default_root_dir=None,
        enable_progress_bar=False,  # disable progress bar
        enable_checkpointing=False,  # does not save model check points
    )

    if mode == "test":
        dataloaders = datamodule.test_dataloader()
        split = cfg.dataset.split.test
    elif mode in ("submission", "test-on-submission"):
        dataloaders = datamodule.submission_dataloader()
        split = cfg.dataset.split.submission
    outputs = dict()
    for i, dataloader in enumerate(dataloaders):
        user, session = split[i]
        logger.info(f"test on {user}-{session}")

        trainer.test(plmodel, dataloader)

        # save model outputs
        pred_dir = Path(
            cfg.path.logdir.predict.format(user=user, session=session)
        )
        pred_dir.mkdir(parents=True, exist_ok=True)

        for key, arr in plmodel.test_results.items():
            path = Path(pred_dir, f"{key}.npy")
            np.save(path, arr)
            logger.info(f"save {key}[shape={arr.shape}] to {path}")

        key = f"{user}-{session}"
        outputs[key] = {
            "y": plmodel.test_results.get("y"),
            "unixtime": plmodel.test_results.get("unixtime"),
        }
        if mode in ("test", "test-on-submission"):
            outputs[key].update({
                "t_idx": plmodel.test_results.get("t"),
            })

    if mode in ("test", "test-on-submission"):
        # save performance summary
        df_summary = eval_operation_segmentation_wrapper(
            cfg, outputs, OPENPACK_OPERATIONS,
        )
        if mode == "test":
            path = Path(cfg.path.logdir.summary.test)
        elif mode == "test-on-submission":
            path = Path(cfg.path.logdir.summary.submission)

        # NOTE: change pandas option to show tha all rows/cols.
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option("display.width", 200)
        df_summary.to_csv(path, index=False)
        logger.info(f"df_summary:\n{df_summary}")
    elif mode == "submission":
        # make submission file
        metadata = {
            "dataset.split.name": cfg.dataset.split.name,
            "mode": mode,
        }

        if not os.path.exists('./save_scores'):
            os.makedirs('./save_scores')
        with open('./save_scores/' + cfg.issue + '.pkl', 'wb') as f:
            pickle.dump(outputs, f)

        submission_dict = construct_submission_dict(
            outputs, OPENPACK_OPERATIONS)

        make_submission_zipfile(submission_dict, logdir, metadata=metadata)


@ hydra.main(version_base=None, config_path="config/TCN/configs/",
             config_name="TCN.yaml")

def main(cfg: DictConfig):
    cfg.dataset.split = optk.configs.datasets.splits.OPENPACK_CHALLENGE_2022_SPLIT
    # DEBUG
    if cfg.debug:
        cfg.dataset.split = optk.configs.datasets.splits.DEBUG_SPLIT
        cfg.path.logdir.rootdir += "/debug"

    # print("===== Params =====")
    # print(OmegaConf.to_yaml(cfg))
    # print("==================")

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode in ("test", "submission", "test-on-submission"):
        test(cfg, mode=cfg.mode)
    else:
        raise ValueError(f"unknown mode [cfg.mode={cfg.mode}]")


if __name__ == "__main__":
    main()