#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 当前仅验证提供单卡推理,通过此配置选择使用的NPU卡
# export ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000

NPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

MM_MODEL="./examples/qwen2vl/inference_qwen2vl_7b.json"

TP=1
PP=1
CP=1
SEQ_LEN=1024
MBS=1
GRAD_ACC_STEP=96
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 28 \
    --num-layer-list 0,0,10,20 \
    --hidden-size 3584 \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28 \
    --tokenizer-type NullTokenizer \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 32768 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --rotary-base 1000000 \
    --lr 1.0e-5 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --weight-decay 0 \
    --train-iters 10000 \
    --lr-warmup-fraction 0.1 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --add-qkv-bias \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 42 \
    --group-query-attention \
    --num-query-groups 4 \
    --bf16 \
    --vocab-size 1 \
    --variable-seq-lengths \
    --enable-one-logger \
    --use-flash-attn
"

MM_ARGS="
    --mm-model $MM_MODEL
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 5000 \
"

torchrun $DISTRIBUTED_ARGS inference_vlm.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
