# coding=utf-8
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import dataclasses
from copy import deepcopy

import torch

from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel import set_defaults_if_not_set_tensor_model_parallel_attributes
from megatron.core.utils import get_model_config
from megatron.training import get_args, get_timers
from megatron.training.training import get_optimizer_param_scheduler
from megatron.training.utils import print_rank_0

FREEZE_STATE_KEY = "freeze_state"


def is_enabled(args=None):
    args = args if args is not None else get_args()
    return bool(getattr(args, "progressive_block_freeze", False))


def _get_total_layers(args):
    if getattr(args, "num_layer_list", None):
        return sum(int(item) for item in args.num_layer_list.split(","))
    return int(args.num_layers)


def _get_pp_size(args):
    return int(getattr(args, "pipeline_model_parallel_size", 1))


def _get_vp_size(args):
    vp_size = getattr(args, "virtual_pipeline_model_parallel_size", None)
    if vp_size is not None:
        return int(vp_size)

    pp_size = _get_pp_size(args)
    virtual_chunk_size = getattr(args, "num_layers_per_virtual_pipeline_stage", None)
    if pp_size <= 1 or virtual_chunk_size is None:
        return 1

    total_layers = _get_total_layers(args)
    layers_per_pp_rank = total_layers // pp_size
    return layers_per_pp_rank // int(virtual_chunk_size)


def _get_layers_per_virtual_chunk(args, total_layers=None):
    total_layers = _get_total_layers(args) if total_layers is None else total_layers
    pp_size = _get_pp_size(args)
    vp_size = _get_vp_size(args)
    layers_per_pp_rank = total_layers // pp_size
    return layers_per_pp_rank // vp_size


def _parse_stage_token(token):
    if "-" not in token:
        raise ValueError(
            "--progressive-block-freeze-stages only supports contiguous ranges like 0-8,8-16."
        )
    start, end = token.split("-", 1)
    start = int(start)
    end = int(end)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid progressive block freeze stage range: {token}")
    return start, end


def _stage_ranges(stage):
    if isinstance(stage, tuple) and len(stage) == 2 and all(isinstance(item, int) for item in stage):
        return [stage]
    return [tuple(item) for item in stage]


def _stage_bounds(stage):
    ranges = _stage_ranges(stage)
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def _stage_display(stage):
    return ",".join(f"[{start}, {end})" for start, end in _stage_ranges(stage))


def _uses_interleaved_pipeline(args):
    return (
        _get_pp_size(args) > 1
        and getattr(args, "virtual_pipeline_model_parallel_size", None) is not None
    )


def _get_interleaved_cycle_size(args, total_layers):
    return _get_pp_size(args) * _get_layers_per_virtual_chunk(args, total_layers)


def _parse_layer_mapping(mapping):
    parsed = {}
    if not mapping:
        return parsed
    for pp_entry in mapping.split(";"):
        pp_entry = pp_entry.strip()
        if not pp_entry:
            continue
        pp_name, layers = pp_entry.split(":", 1)
        pp_name = pp_name.strip().lower()
        if not pp_name.startswith("pp"):
            raise ValueError(f"Invalid layer mapping entry: {pp_entry}")
        pp_rank = int(pp_name[2:])
        parsed[pp_rank] = [int(item) for item in layers.split(",") if item.strip()]
    return parsed


def build_global_layer_to_owner(args):
    total_layers = _get_total_layers(args)
    pp_size = _get_pp_size(args)
    vp_size = _get_vp_size(args)
    if total_layers % pp_size != 0:
        raise ValueError("num_layers must be divisible by pipeline_model_parallel_size.")

    layers_per_pp_rank = total_layers // pp_size
    if layers_per_pp_rank % vp_size != 0:
        raise ValueError("layers per pipeline rank must be divisible by virtual pipeline size.")

    layers_per_virtual_chunk = layers_per_pp_rank // vp_size
    total_layers_per_vp_rank = total_layers // vp_size
    owner = {}
    for vp_rank in range(vp_size):
        for pp_rank in range(pp_size):
            offset = vp_rank * total_layers_per_vp_rank + pp_rank * layers_per_virtual_chunk
            for local_idx in range(layers_per_virtual_chunk):
                global_layer_id = offset + local_idx
                owner[global_layer_id] = (pp_rank, vp_rank, local_idx)

    if sorted(owner) != list(range(total_layers)):
        raise ValueError("native VP layer mapping has missing or duplicate layers.")

    explicit_mapping = _parse_layer_mapping(
        getattr(args, "progressive_block_freeze_layer_mapping", None)
    )
    if explicit_mapping:
        explicit_owner = {}
        for pp_rank, layers in explicit_mapping.items():
            if pp_rank < 0 or pp_rank >= pp_size:
                raise ValueError(f"Invalid PP rank in layer mapping: {pp_rank}")
            for global_layer_id in layers:
                if global_layer_id < 0 or global_layer_id >= total_layers:
                    raise ValueError(f"Invalid global layer id in layer mapping: {global_layer_id}")
                if global_layer_id in explicit_owner:
                    raise ValueError(f"Duplicate global layer id in layer mapping: {global_layer_id}")
                explicit_owner[global_layer_id] = pp_rank
        missing = sorted(set(range(total_layers)) - set(explicit_owner))
        if missing:
            raise ValueError(f"Layer mapping is missing global layers: {missing}")

        mismatches = [
            (global_layer_id, explicit_owner[global_layer_id], owner[global_layer_id][0])
            for global_layer_id in range(total_layers)
            if explicit_owner[global_layer_id] != owner[global_layer_id][0]
        ]
        if mismatches:
            raise ValueError(
                "progressive block freeze only supports Megatron native VP-compatible "
                f"layer mapping in the first implementation. Mismatches: {mismatches[:8]}"
            )

    return owner


def build_layer_to_pp_rank_mapping(args):
    owner = build_global_layer_to_owner(args)
    pp_size = _get_pp_size(args)
    mapping = {pp_rank: [] for pp_rank in range(pp_size)}
    for global_layer_id in sorted(owner):
        pp_rank, _, _ = owner[global_layer_id]
        mapping[pp_rank].append(global_layer_id)
    return mapping


def get_current_rank_global_layer_ids(args):
    owner = build_global_layer_to_owner(args)
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    if vp_rank is None:
        vp_rank = 0
    local_layers = [
        (local_idx, global_layer_id)
        for global_layer_id, (owner_pp_rank, owner_vp_rank, local_idx) in owner.items()
        if owner_pp_rank == pp_rank and owner_vp_rank == vp_rank
    ]
    return [global_layer_id for _, global_layer_id in sorted(local_layers)]


def _validate_interleaved_stage_alignment(args, total_layers, stages):
    if not _uses_interleaved_pipeline(args):
        return
    cycle_size = _get_interleaved_cycle_size(args, total_layers)
    for stage in stages:
        for start, end in _stage_ranges(stage):
            if start % cycle_size != 0 or end % cycle_size != 0:
                raise ValueError(
                    "progressive block freeze stage ranges must align to a full interleaved "
                    f"pipeline cycle of {cycle_size} blocks."
                )


def _build_global_stages(args, total_layers):
    window_size = int(args.progressive_block_freeze_window_size)
    stride = int(args.progressive_block_freeze_window_stride or window_size)
    start = int(args.progressive_block_freeze_start_block)
    stages = []
    while start < total_layers:
        stages.append((start, min(start + window_size, total_layers)))
        start += stride
    return stages


def build_stages(args):
    total_layers = _get_total_layers(args)
    if getattr(args, "progressive_block_freeze_stages", None):
        stages = [
            _parse_stage_token(token.strip())
            for token in args.progressive_block_freeze_stages.split(",")
            if token.strip()
        ]
    else:
        stages = _build_global_stages(args, total_layers)

    _validate_interleaved_stage_alignment(args, total_layers, stages)

    if not stages:
        raise ValueError("progressive block freeze generated no stages.")
    seen_ranges = []
    for stage in stages:
        stage_ranges = sorted(_stage_ranges(stage))
        prev_end = None
        for start, end in stage_ranges:
            if start < 0 or end <= start:
                raise ValueError(f"Invalid progressive block freeze stage range: [{start}, {end})")
            if end > total_layers:
                raise ValueError(
                    f"Progressive block freeze stage [{start}, {end}) exceeds num_layers={total_layers}."
                )
            if prev_end is not None and start < prev_end:
                raise ValueError("Progressive block freeze ranges in the same stage must not overlap.")
            prev_end = end
            seen_ranges.append((start, end))
    prev_end = None
    for start, end in sorted(seen_ranges):
        if prev_end is not None and start < prev_end:
            raise ValueError("Progressive block freeze stages must not overlap.")
        prev_end = end
    return stages


def _set_active_stage(state, stage):
    active_start, active_end = _stage_bounds(stage)
    state["active_start"] = active_start
    state["active_end"] = active_end
    state["active_ranges"] = _stage_ranges(stage)


# 初始化训练stage
def init_state(args, iteration=0):
    stages = build_stages(args)
    state = {
        "stage_idx": 0,
        "block_enter_iteration": int(iteration),
        "plateau_hits": 0,
        "recent_loss_history": [],
        "last_switch_reason": "init",
    }
    _set_active_stage(state, stages[0])
    return state

# 重点函数确定当前stage的状态（debug的重点位置）
"""
训练中args维持一个全局变量查看当前的状态,包含当前训练stage的完整信息
args.freeze_state

args.freeze_state = {
    "stage_idx": 2,
    "active_start": 16,
    "active_end": 24,
    "active_ranges": [(16, 24)],
    "block_enter_iteration": 5000,
    "plateau_hits": 1,
    "recent_loss_history": [2.31, 2.28, 2.27],
    "last_switch_reason": "plateau",
}

"""
def ensure_state(args=None, iteration=None):
    args = args if args is not None else get_args()
    if not is_enabled(args):
        return None
    if getattr(args, FREEZE_STATE_KEY, None) is None:
        setattr(args, FREEZE_STATE_KEY, init_state(args, args.iteration if iteration is None else iteration))
    return getattr(args, FREEZE_STATE_KEY)


def get_state_dict(args=None):
    args = args if args is not None else get_args()
    state = getattr(args, FREEZE_STATE_KEY, None)
    return deepcopy(state) if state is not None else None


def load_state_dict(state, args=None):
    args = args if args is not None else get_args()
    if state is None:
        return
    state = deepcopy(state)
    if state.get("active_ranges") is None:
        stages = build_stages(args)
        stage_idx = int(state.get("stage_idx", 0))
        if 0 <= stage_idx < len(stages):
            _set_active_stage(state, stages[stage_idx])
        elif "active_start" in state and "active_end" in state:
            state["active_ranges"] = [(int(state["active_start"]), int(state["active_end"]))]
    setattr(args, "progressive_block_freeze_loaded", True)
    setattr(args, FREEZE_STATE_KEY, state)


def _iter_transformer_blocks(model):
    for model_chunk in model if isinstance(model, list) else [model]:
        for module in model_chunk.modules():
            layers = getattr(module, "layers", None)
            if not isinstance(layers, torch.nn.ModuleList):
                continue
            for layer in layers:
                if not hasattr(layer, "layer_number"):
                    continue
                # Megatron-Core layer_number is 1-based and already includes pipeline offset.
                yield int(layer.layer_number) - 1, layer


def apply_freeze(model, args=None):
    args = args if args is not None else get_args()
    if not is_enabled(args):
        return

    state = ensure_state(args)
    active_ranges = state.get("active_ranges")
    if active_ranges is None:
        active_ranges = [(int(state["active_start"]), int(state["active_end"]))]
    active_blocks = 0
    active_params = 0

    for model_chunk in model if isinstance(model, list) else [model]:
        for param in model_chunk.parameters():
            param.requires_grad = False

    for global_idx, layer in _iter_transformer_blocks(model):
        trainable = any(start <= global_idx < end for start, end in active_ranges)
        if trainable:
            active_blocks += 1
        for param in layer.parameters(recurse=True):
            param.requires_grad = trainable
            if trainable:
                active_params += param.numel()

    if torch.distributed.is_initialized():
        count = torch.tensor([active_blocks], dtype=torch.int, device=torch.cuda.current_device())
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        if count.item() == 0:
            raise RuntimeError(
                f"Progressive block freeze active ranges {_stage_display(active_ranges)} "
                "did not match any local/global transformer blocks."
            )
    elif active_blocks == 0:
        raise RuntimeError(
            f"Progressive block freeze active ranges {_stage_display(active_ranges)} "
            "did not match any transformer blocks."
        )

    print_rank_0(
        f"progressive block freeze active blocks {_stage_display(active_ranges)}, "
        f"local active blocks {active_blocks}, local active params {active_params}"
    )


def reduced_train_loss(loss_dict, args=None):
    args = args if args is not None else get_args()
    if not loss_dict:
        return None
    loss_key = getattr(args, "progressive_block_freeze_loss_key", None)
    if loss_key is None:
        loss_key = next(iter(loss_dict))
    if loss_key not in loss_dict:
        raise KeyError(f"Loss key {loss_key} is not in reduced train loss dict: {list(loss_dict)}")
    loss = loss_dict[loss_key]
    if torch.is_tensor(loss):
        return float(loss.detach().float().mean().item())
    return float(loss)


def maybe_advance(loss_dict, iteration, skipped_iter, args=None):
    args = args if args is not None else get_args()
    if not is_enabled(args):
        return False, None
    state = ensure_state(args, iteration)
    stages = build_stages(args)
    if state["stage_idx"] >= len(stages) - 1:
        return False, None

    iters_in_block = iteration - int(state["block_enter_iteration"])
    max_block_iters = int(args.progressive_block_freeze_max_block_iters)
    if max_block_iters > 0 and iters_in_block >= max_block_iters:
        _switch_to_next_stage(state, stages, iteration, "max_block_iters")
        return True, "max_block_iters"

    if skipped_iter:
        return False, None

    loss = reduced_train_loss(loss_dict, args)
    if loss is None:
        return False, None
    history = state.setdefault("recent_loss_history", [])
    history.append(loss)
    window_size = int(args.progressive_block_freeze_plateau_window_size)
    del history[:max(0, len(history) - 2 * window_size)]

    if iters_in_block < int(args.progressive_block_freeze_min_block_iters):
        return False, None
    if len(history) < 2 * window_size:
        return False, None

    prev_mean = sum(history[:window_size]) / window_size
    curr_mean = sum(history[window_size:]) / window_size
    rel_improve = (prev_mean - curr_mean) / max(abs(prev_mean), 1.0e-12)
    if rel_improve < float(args.progressive_block_freeze_threshold):
        state["plateau_hits"] = int(state.get("plateau_hits", 0)) + 1
    else:
        state["plateau_hits"] = 0

    if state["plateau_hits"] >= int(args.progressive_block_freeze_patience):
        _switch_to_next_stage(state, stages, iteration, "plateau")
        return True, "plateau"
    return False, None


def _switch_to_next_stage(state, stages, iteration, reason):
    next_stage = int(state["stage_idx"]) + 1
    stage = stages[next_stage]
    state["stage_idx"] = next_stage
    _set_active_stage(state, stage)
    state["block_enter_iteration"] = int(iteration)
    state["plateau_hits"] = 0
    state["recent_loss_history"] = []
    state["last_switch_reason"] = reason
    print_rank_0(
        f"progressive block freeze switches to stage {next_stage} "
        f"{_stage_display(stage)} at iteration {iteration}, reason={reason}"
    )


def _model_chunks(model):
    return model if isinstance(model, list) else [model]


def _unwrap_data_parallel(model):
    chunks = []
    for model_chunk in _model_chunks(model):
        if isinstance(model_chunk, DDP):
            chunks.append(model_chunk.module)
        else:
            chunks.append(model_chunk)
    return chunks


def _build_ddp_config(args, num_parameters):
    kwargs = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(DistributedDataParallelConfig)
        if hasattr(args, field.name)
    }
    kwargs.update({
        'grad_reduce_in_fp32': args.accumulate_allreduce_grads_in_fp32,
        'check_for_nan_in_grad': args.check_for_nan_in_loss_and_grad,
        'check_for_large_grads': args.check_for_large_grads,
        'pad_buckets_for_high_nccl_busbw': args.ddp_pad_buckets_for_high_nccl_busbw,
        'average_in_collective': args.ddp_average_in_collective,
    })

    if args.ddp_num_buckets is not None:
        if args.ddp_bucket_size is not None:
            raise ValueError("Cannot specify both --ddp-num-buckets and --ddp-bucket-size")
        if args.ddp_num_buckets <= 0:
            raise ValueError("--ddp-num-buckets must be greater than 0")
        kwargs['bucket_size'] = num_parameters // args.ddp_num_buckets
    else:
        kwargs['bucket_size'] = args.ddp_bucket_size

    ddp_config = DistributedDataParallelConfig(**kwargs)
    if ddp_config.bucket_size is None:
        ddp_config.bucket_size = max(
            40000000, 1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    if not ddp_config.overlap_grad_reduce:
        ddp_config.bucket_size = None
    return ddp_config


def _build_optimizer_scheduler(model, timers, no_wd_decay_cond=None, scale_lr_cond=None, lr_mult=1.0):
    args = get_args()
    optim_kwargs = {}
    for field in dataclasses.fields(OptimizerConfig):
        if hasattr(args, field.name):
            optim_kwargs[field.name] = getattr(args, field.name)
    optim_config = OptimizerConfig(**optim_kwargs)
    optim_config.timers = timers
    optimizer = get_megatron_optimizer(
        optim_config,
        model,
        no_wd_decay_cond,
        scale_lr_cond,
        lr_mult,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
    if getattr(args, "consumed_train_samples", 0) > 0:
        opt_param_scheduler.step(increment=args.consumed_train_samples)
    return optimizer, opt_param_scheduler


def rebuild_optimizer_scheduler(model, no_wd_decay_cond=None, scale_lr_cond=None, lr_mult=1.0):
    args = get_args()
    timers = get_timers()

    if getattr(args, "use_custom_fsdp", False) or getattr(args, "use_torch_fsdp2", False):
        raise RuntimeError("progressive block freeze optimizer rebuild supports Megatron DDP only.")
    if not getattr(args, "use_distributed_optimizer", False):
        raise RuntimeError("progressive block freeze requires Megatron distributed optimizer.")

    model = _unwrap_data_parallel(model)
    apply_freeze(model, args)

    for model_module in model:
        for param in model_module.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    num_parameters = sum(sum(param.nelement() for param in model_module.parameters()) for model_module in model)
    config = get_model_config(model[0])
    ddp_config = _build_ddp_config(args, num_parameters)

    model = [
        DDP(
            config=config,
            ddp_config=ddp_config,
            module=model_chunk,
            disable_bucketing=(model_chunk_idx > 0) or args.overlap_param_gather_with_optimizer_step,
        )
        for model_chunk_idx, model_chunk in enumerate(model)
    ]
    if args.data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()

    optimizer, opt_param_scheduler = _build_optimizer_scheduler(
        model, timers, no_wd_decay_cond, scale_lr_cond, lr_mult
    )
    return model, optimizer, opt_param_scheduler
