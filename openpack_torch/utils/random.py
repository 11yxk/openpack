import random
from logging import getLogger

import numpy as np
import torch

logger = getLogger(__name__)


def reset_seed(seed: int = 0) -> None:
    """Reset random seed (random, numpy, torch-cpu, torch-cuda) for reproducibility.

    Args:
        seed (int): random seed. (Default: 1)
    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logger.info(
        "Reset Seeds: python={}, numpy={}, Pytorch (cpu={}, cuda={})".format(
            seed, seed, seed, seed
        )
    )
