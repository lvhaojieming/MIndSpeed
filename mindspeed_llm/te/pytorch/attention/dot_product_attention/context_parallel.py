# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.


class KVAllGatherCPStrategy:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "KVAllGatherCPStrategy is not available in this MindSpeed-LLM/MindSpeed-Core "
            "combination. Please use a context-parallel attention strategy provided by the "
            "installed runtime."
        )
