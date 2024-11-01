# Copyright 2024 Huawei Technologies Co., Ltd

import gc

import torch
from torch.distributed._shard.sharded_tensor.api import ShardedTensor


class TorchPatcher:

    @staticmethod
    def new_get_preferred_device(self) -> torch.device:
        """
        Return the preferred device to be used when creating tensors for collectives.
        This method takes into account the asccociated process group
        This patch method makes the torch npu available for distribution
        """
        if dist.get_backend(self._process_group) == dist.Backend.NCCL:
            return torch.device(torch.cuda.current_device())
        try:
            import torch_npu

            return torch.device(torch_npu.npu.current_device())
        except Exception as e:
            return torch.device("cpu")

    @classmethod
    def apply_patch(cls):
        # Apply the patch for npu distribution
        ShardedTensor._get_preferred_device = cls.new_get_preferred_device


def config_gc():
    # set gc threshold
    gc.set_threshold(700, 50, 1000)
