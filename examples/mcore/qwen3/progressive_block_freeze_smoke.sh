#!/bin/bash
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export PYTHONPATH=/root/MindSpeed:${PYTHONPATH:-}
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_ASD_ENABLE=0
export TASK_QUEUE_ENABLE=2
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

NPUS_PER_NODE=${NPUS_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6030}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

TOKENIZER_PATH=${TOKENIZER_PATH:-/root/model_tokenizer}
DATA_PATH=${DATA_PATH:-/root/datasets/c4_en_6shards_megatron/c4_qwen3_text_document}
CKPT_SAVE_DIR=${CKPT_SAVE_DIR:-/root/MindSpeed/ckpt_progressive_freeze_smoke}
LOG_FILE=${LOG_FILE:-/root/MindSpeed/logs/progressive_block_freeze_smoke.log}

mkdir -p "$(dirname "${LOG_FILE}")" "${CKPT_SAVE_DIR}"

DISTRIBUTED_ARGS=(
  --nproc_per_node "${NPUS_PER_NODE}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

GPT_ARGS=(
  --use-mcore-models
  --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec
  --qk-layernorm
  --tokenizer-name-or-path "${TOKENIZER_PATH}"
  --tokenizer-type PretrainedFromHF
  --make-vocab-size-divisible-by 1
  --padded-vocab-size 151936
  --rotary-base 1000000
  --untie-embeddings-and-output-weights
  --disable-bias-linear
  --position-embedding-type rope
  --normalization RMSNorm
  --swiglu
  --attention-softmax-in-fp32
  --group-query-attention
  --num-query-groups 2
  --norm-epsilon 1e-6
  --ckpt-format torch
  --transformer-impl local
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --num-layers 4
  --hidden-size 128
  --ffn-hidden-size 256
  --num-attention-heads 4
  --seq-length 128
  --max-position-embeddings 128
)

TRAIN_ARGS=(
  --micro-batch-size 1
  --global-batch-size 1
  --lr 1.0e-5
  --lr-decay-style constant
  --min-lr 1.0e-6
  --weight-decay 0.0
  --clip-grad 1.0
  --adam-beta1 0.9
  --adam-beta2 0.95
  --initial-loss-scale 4096
  --seed 42
  --bf16
  --train-iters 2
)

DATA_ARGS=(
  --data-path "${DATA_PATH}"
  --split 100,0,0
)

OUTPUT_ARGS=(
  --log-interval 1
  --save-interval 1000
  --eval-interval 1000
  --eval-iters 0
  --save "${CKPT_SAVE_DIR}"
  --no-save-optim
  --no-save-rng
)

OPTIMIZE_ARGS=(
  --use-distributed-optimizer
  --reuse-fp32-param
  --no-gradient-accumulation-fusion
  --no-masked-softmax-fusion
  --distributed-backend nccl
)

FREEZE_ARGS=(
  --progressive-block-freeze
  --progressive-block-freeze-stages 0-2,2-4
  --progressive-block-freeze-max-block-iters 1
  --progressive-block-freeze-plateau-window-size 1
  --progressive-block-freeze-patience 1
)

/usr/local/python3.11.13/bin/python -m torch.distributed.run \
  "${DISTRIBUTED_ARGS[@]}" \
  /root/MindSpeed/pretrain_gpt.py \
  "${GPT_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${OUTPUT_ARGS[@]}" \
  "${OPTIMIZE_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  "${FREEZE_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
