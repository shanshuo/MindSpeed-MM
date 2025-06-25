#!/bin/bash

# Copy some files from MindSpeed MM to MindSpeed RL

echo "Copying some files from MindSpeed MM to MindSpeed RL..."

cp examples/rl/code/mindspeed_rl/vllm_parallel_state.py mindspeed_rl/models/rollout/vllm_adapter/vllm_parallel_state.py

echo "All performance files copied successfully!"
echo "Total files copied: 1"
