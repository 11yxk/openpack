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

from my_model import ctrgcn_base as CTRGCN


from openpack_torch.models.keypoint.graph_new1 import Graph
import pickle
import loss_fn
import os
import random
from metric import f_score, edit_score
logger = getLogger(__name__)
optorch.configs.register_configs()
optorch.utils.reset_seed(seed=0)

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ----------------------------------------------------------------------
MSCOCO_SKELETON_LAYOUT = ((0, 2),(1, 0),(2, 1),(3, 1),(4, 2),(5, 3),(6, 4),(7, 5)
                          ,(8, 6),(9, 7),(10, 8),(11, 5),(12, 6),(13, 11),(14, 12))


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
class OpenPackKeypointDataModule(optorch.data.OpenPackBaseDataModule):

    dataset_class = optorch.data.datasets.OpenPackKeypoint
    def get_kwargs_for_datasets(self, stage: Optional[str] = None) -> Dict:
        submission = True if self.cfg.mode == "submission" else False

        kwargs = {
            "debug": self.cfg.debug,
            "window": self.cfg.train.window,
            "submission": submission,
        }
        return kwargs


class STGCN4SegLM(optorch.lightning.BaseLightningModule):

    def init_model(self, cfg: DictConfig) -> torch.nn.Module:

        in_ch = 2
        self.cfg = cfg
        g = Graph(strategy='uniform')
        # g = Graph(strategy='spatial')


        model = CTRGCN.Model(
            num_class=len(OPENPACK_OPERATIONS), num_point=15, graph=g.A, in_channels=in_ch, adaptive=True,
        temporal_kernel = self.cfg.temporal_kernel,dilations = self.cfg.dilations)

        return model

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:

        x = batch["x"].to(device=self.device, dtype=torch.float)
        t = batch["t"].to(device=self.device, dtype=torch.long)
        # remove foot joint
        x = x[..., :15]

        tmse = loss_fn.TMSE()


        if self.cfg.bone:

            bone_x = torch.zeros_like(x)

            for v1, v2 in MSCOCO_SKELETON_LAYOUT:

                bone_x[..., v1] = x[..., v1] - x[..., v2]

            x = bone_x


        y_hat = self(x).squeeze(3)


        loss = self.criterion(y_hat, t) + self.cfg.alpha * tmse(y_hat, t)

        acc = self.calc_accuracy(y_hat, t)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        x = batch["x"].to(device=self.device, dtype=torch.float)
        t = batch["t"].to(device=self.device, dtype=torch.long)
        x = x[..., :15]

        ts_unix = batch["ts"]

        if self.cfg.bone:

            bone_x = torch.zeros_like(x)

            for v1, v2 in MSCOCO_SKELETON_LAYOUT:

                bone_x[..., v1] = x[..., v1] - x[..., v2]

            x = bone_x


        y_hat = self(x).squeeze(3)


        outputs = dict(t=t, y=y_hat, unixtime=ts_unix)

        return outputs




# ----------------------------------------------------------------------


def train(cfg: DictConfig):
    device = torch.device("cuda")
    logdir = Path(cfg.path.logdir.rootdir)
    logger.debug(f"logdir = {logdir}")
    optk.utils.io.cleanup_dir(logdir, exclude="hydra")
    datamodule = OpenPackKeypointDataModule(cfg)
    plmodel = STGCN4SegLM(cfg)
    plmodel.to(dtype=torch.float, device=device)
    # logger.info(plmodel)

    num_epoch = cfg.train.debug.epochs if cfg.debug else cfg.train.epochs

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     save_top_k=0,
    #     save_last=True,
    #     monitor=None,
    # )


    # save all models
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k = -1,
        filename='{epoch}'
    )

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

    logger.info(f"Train the model for {num_epoch} epochs.")
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

    datamodule = OpenPackKeypointDataModule(cfg)
    datamodule.setup(mode)

    ckpt_path = Path(logdir, "checkpoints", "best.ckpt")

    logger.info(f"load checkpoint from {ckpt_path}")
    plmodel = STGCN4SegLM.load_from_checkpoint(ckpt_path, cfg=cfg)
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
        df_summary, f1s , acc , edit = eval_operation_segmentation_wrapper(
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
        # logger.info(f"df_summary:\n{df_summary}")
        f1_score = df_summary[(df_summary['key']=='all') & (df_summary['name']=='avg/weighted')]['f1'].values[0]
        logger.info(f"weighted f1_score:{f1_score}")
        logger.info(f"f1_score@10:{f1s[0]}")
        logger.info(f"f1_score@25:{f1s[1]}")
        logger.info(f"f1_score@50:{f1s[2]}")
        logger.info(f"Accuracy:{acc}")
        logger.info(f"Edit score:{edit}")

    elif mode == "submission":
        # make submission file
        metadata = {
            "dataset.split.name": cfg.dataset.split.name,
        }

        if not os.path.exists('./save_scores'):
            os.makedirs('./save_scores')
        with open('./save_scores/' + cfg.issue + '.pkl', 'wb') as f:
            pickle.dump(outputs, f)

        submission_dict = construct_submission_dict(
            outputs, OPENPACK_OPERATIONS)

        make_submission_zipfile(submission_dict, logdir, metadata=metadata)


@ hydra.main(version_base=None, config_path="config/ctr-gcn/configs",
             config_name="ctr-gcn.yaml")

def main(cfg: DictConfig):

    init_seed(1234)

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