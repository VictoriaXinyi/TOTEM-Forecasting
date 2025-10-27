# -*- coding: utf-8 -*-
"""
中文说明：
模型基类，统一优化器配置并声明训练/评估共享接口。
派生模型需实现 shared_eval(batch, optimizer, mode, comet_logger)。
"""

import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def shared_eval(self, batch, optimizer, mode, comet_logger='None'):
        pass

    def configure_optimizers(self, lr=1e-3):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=lr)  # adds weight decay
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        return optimizer
