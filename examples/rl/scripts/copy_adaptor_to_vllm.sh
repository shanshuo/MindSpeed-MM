#!/bin/bash

# Copy all performance files from MindSpeed-MM to MindSpeed-RL
# This script should be run from MindSpeed-RL root directory

echo "Copying vllm adaptor files from MindSpeed-MM to VLLM/VLLM_ASCEND ..."

# vllm files
cp MindSpeed-MM/examples/rl/code/vllm/qwen2_vl.py vllm/vllm/model_executor/models/qwen2_vl.py
cp MindSpeed-MM/examples/rl/code/vllm/qwen2_5_vl.py vllm/vllm/model_executor/models/qwen2_5_vl.py

# vllm_ascend files
cp MindSpeed-MM/examples/rl/code/vllm_ascend/__init__.py vllm-ascend/vllm_ascend/models/__init__.py
cp MindSpeed-MM/examples/rl/code/vllm_ascend/model_runner.py vllm-ascend/vllm_ascend/worker/model_runner.py
cp MindSpeed-MM/examples/rl/code/vllm_ascend/utils.py vllm-ascend/vllm_ascend/utils.py

sleep 3
echo "All vllm files copied successfully!"
echo "Total files copied: 5"