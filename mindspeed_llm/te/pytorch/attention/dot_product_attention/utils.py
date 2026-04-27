# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import torch


def get_distributed_world_size(group):
    if group is None or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group)
