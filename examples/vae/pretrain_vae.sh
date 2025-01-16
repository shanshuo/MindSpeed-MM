#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29505
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


AE_DATA="./examples/vae/data.json"
AE_MODEL="./examples/vae/model.json"
AE_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="your_ckpt_path"
SAVE_PATH="your_ckpt_path_to_save"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TRAINING_ARGS="
    --epochs 10 \
    --micro-batch-size 1 \
    --num-workers 4 \
    --ae-lr 0.00001 \
    --ae-wd 0.0001 \
    --discriminator-lr 0.00001 \
    --discriminator-wd 0.01 \
    --mix-precision bf16
"

AE_ARGS="
    --data-config $AE_DATA \
    --model-config $AE_MODEL \
    --tool-config $AE_TOOL
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --save $SAVE_PATH
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS pretrain_ae.py \
    $TRAINING_ARGS \
    $AE_ARGS \
    $OUTPUT_ARGS  >> logs/train_${logfile}.log 2>&1