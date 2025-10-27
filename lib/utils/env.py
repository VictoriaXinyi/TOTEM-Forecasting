"""
中文说明：
环境工具：提供随机种子设置与统一输出目录管理。
- seed_all_rng：统一设置 torch/np/python 的随机种子；
- get_default_output_dir：优先读取环境变量 OUTPUT_DIR，否则使用当前工作目录下的 output；
- ensure_dir：若目录不存在则创建，用于稳健的路径管理。
"""

import numpy as np
import torch
import random
import os
from datetime import datetime


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.
    Args:
        seed (int): if None, will use a strong random seed.
    Use:
        seed_all_rng(None if seed < 0 else seed)
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        print("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_default_output_dir():
    """
    获取默认输出目录。
    优先使用环境变量 OUTPUT_DIR；否则在当前工作目录下创建 'output'。
    返回：字符串路径。
    """
    base = os.environ.get("OUTPUT_DIR", os.path.join(os.getcwd(), "output"))
    return base


def ensure_dir(path: str):
    """
    若目录不存在则创建，返回标准化路径。
    Args:
        path: 待创建的目录路径；若为空字符串/None则不操作。
    Returns:
        传入的路径字符串。
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path