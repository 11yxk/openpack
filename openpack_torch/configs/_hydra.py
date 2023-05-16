import openpack_toolkit as optk
from hydra.core.config_store import ConfigStore

from .datasets import (OPENPACK_2D_KEYPOINT_DATASET_CONFIG,
                       OPENPACK_ACC_DATASET_CONFIG)


def register_configs() -> None:
    cs = ConfigStore.instance()

    data = {
        "user": [
            optk.configs.users.U0101,
            optk.configs.users.U0102,
            optk.configs.users.U0103,
            optk.configs.users.U0104,
            optk.configs.users.U0105,
            optk.configs.users.U0106,
            optk.configs.users.U0107,
            optk.configs.users.U0108,
            optk.configs.users.U0109,
            optk.configs.users.U0110,
            optk.configs.users.U0111,
            optk.configs.users.U0202,
            optk.configs.users.U0203,
            optk.configs.users.U0204,
            optk.configs.users.U0205,
            optk.configs.users.U0207,
            optk.configs.users.U0210,
        ],
        "dataset/stream": [
            optk.configs.datasets.streams.ATR_ACC_STREAM,
            optk.configs.datasets.streams.ATR_QAGS_STREAM,
            optk.configs.datasets.streams.E4_ACC_STREAM,
            optk.configs.datasets.streams.E4_BVP_STREAM,
            optk.configs.datasets.streams.E4_EDA_STREAM,
            optk.configs.datasets.streams.E4_TEMP_STREAM,
            optk.configs.datasets.streams.KINECT_2D_KPT_STREAM,
            optk.configs.datasets.streams.SYSTEM_HT_ORIGINAL_STREAM,
            optk.configs.datasets.streams.SYSTEM_ORDER_SHEET_STREAM,
        ],
        "dataset/split": [
            optk.configs.datasets.splits.DEBUG_SPLIT,
            optk.configs.datasets.splits.PILOT_CHALLENGE_SPLIT,
            optk.configs.datasets.splits.OPENPACK_CHALLENGE_2022_SPLIT,
        ],
        "dataset/annotation": [
            optk.configs.datasets.annotations.OPENPACK_ACTIONS_ANNOTATION,
            optk.configs.datasets.annotations.OPENPACK_OPERATIONS_ANNOTATION,
            optk.configs.datasets.annotations.ACTIVITY_1S_ANNOTATION,
        ],
        "dataset": [
            OPENPACK_ACC_DATASET_CONFIG, OPENPACK_2D_KEYPOINT_DATASET_CONFIG
        ],
    }
    for group, items in data.items():
        for item in items:
            cs.store(group=group, name=item.name, node=item)

    # Activity Set
    cs.store(group="dataset/classes", name="OPENPACK_OPERATIONS",
             node=optk.configs.datasets.annotations.OPENPACK_OPERATIONS)
    cs.store(group="dataset/classes", name="OPENPACK_ACTIONS",
             node=optk.configs.datasets.annotations.OPENPACK_ACTIONS)
