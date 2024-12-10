#!/bin/bash
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HOST_CACHE_CAPACITY=20
export ACLNN_CACHE_LIMIT=100000

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MBS=1
GRAD_ACC_STEP=64
TP=1
PP=4
CP=1
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

BASEPATH=$(cd `dirname $0`; cd ../../../; pwd)

MM_DATA="$BASEPATH/tests/st/run_configs/finetune_internvl2_8B/data_8B.json"
MM_MODEL="$BASEPATH/tests/st/run_configs/finetune_internvl2_8B/model_8B.json"
MM_TOOL="$BASEPATH/mindspeed_mm/tools/tools.json"
LOAD_PATH="/home/ci_resource/models/InternVL2-8B/pretrained/ckpt_pp4"


MM_ARGS="
    --mm-data ${MM_DATA} \
    --mm-model ${MM_MODEL} \
    --mm-tool ${MM_TOOL}
"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 16 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type NullTokenizer \
    --vocab-size 92553 \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --lr 4e-5 \
    --min-lr 0.0 \
    --train-iters 2 \
    --lr-decay-style cosine \
    --weight-decay 0.05 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --use-distributed-optimizer \
    --bf16 \
    --load $LOAD_PATH \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --variable-seq-lengths \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --eval-iters 5000 \
"


torchrun $DISTRIBUTED_ARGS \
    $BASEPATH/pretrain_internvl.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl