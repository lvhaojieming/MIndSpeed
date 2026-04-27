# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from mindspeed_llm.core.context_parallel.kvallgather_context_parallel import (
    get_seq_chunk_ids_for_reordering_before_attn,
)
from mindspeed_llm.te.pytorch.attention.dot_product_attention.utils import get_distributed_world_size

