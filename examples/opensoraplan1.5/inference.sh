source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_LAUNCH_BLOCKING=1
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

TP=1
PP=1
CP=1
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

MM_MODEL="examples/opensoraplan1.5/inference_model.json"
LOAD_PATH="ckpt/opensoraplan1.5/"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
MM_ARGS="
    --mm-model $MM_MODEL
"

SORA_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --num-layers 32 \
    --hidden-size 3072 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type NullTokenizer \
    --vocab-size 0 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --sequence-parallel \
    --fp16 \
    --load $LOAD_PATH \
"

torchrun $DISTRIBUTED_ARGS  inference_sora.py  $MM_ARGS $SORA_ARGS 2>&1
