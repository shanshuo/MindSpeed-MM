#!/bin/bash

# Copy all performance files from MindSpeed-MM to MindSpeed-RL
# This script should be run from MindSpeed-RL root directory

echo "Copy vllm adaptor files from MindSpeed-MM to VLLM/VLLM_ASCEND"
echo "[Warning] Some files will be overwritten in VLLM/VLLM_ASCEND"

while true
do
    read -r -p "Are You Sure? [Y/n] " input

    case $input in
        [yY][eE][sS]|[yY])
            echo "Copying ..."
            break
            ;;

        [nN][oO]|[nN])
            echo "Exit"
            exit 1
            ;;

        *)
            echo "Invalid input..."
            ;;
    esac
done

# vllm files
cp MindSpeed-MM/examples/rl/code/vllm/qwen2_vl.py vllm/vllm/model_executor/models/qwen2_vl.py
cp MindSpeed-MM/examples/rl/code/vllm/qwen2_5_vl.py vllm/vllm/model_executor/models/qwen2_5_vl.py

# vllm_ascend files
cp MindSpeed-MM/examples/rl/code/vllm_ascend/__init__.py vllm-ascend/vllm_ascend/models/__init__.py
cp MindSpeed-MM/examples/rl/code/vllm_ascend/model_runner.py vllm-ascend/vllm_ascend/worker/model_runner.py
cp MindSpeed-MM/examples/rl/code/vllm_ascend/utils.py vllm-ascend/vllm_ascend/utils.py

echo "All vllm files copied successfully!"
echo "Total files copied: 5"