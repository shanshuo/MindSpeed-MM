import os
import stat
import copy
import argparse
from typing import Any, Dict, List

import torch
from safetensors.torch import load_file


DIT_CONVERT_MAPPING = {
    "time_in.in_layer.bias": "time_in.mlp.0.bias",
    "time_in.in_layer.weight": "time_in.mlp.0.weight",
    "time_in.out_layer.bias": "time_in.mlp.2.bias",
    "time_in.out_layer.weight": "time_in.mlp.2.weight",
    "vector_in.in_layer.bias": "vector_in.fc1.bias",
    "vector_in.in_layer.weight": "vector_in.fc1.weight",
    "vector_in.out_layer.bias": "vector_in.fc2.bias",
    "vector_in.out_layer.weight": "vector_in.fc2.weight",
}


def get_double_layer_mapping(i: int) -> Dict:
    layer_mapping = {}

    layer_mapping[f"double_blocks.{i}.img_mod.lin.bias"] = f"double_blocks.{i}.img_mod.linear.bias"
    layer_mapping[f"double_blocks.{i}.img_mod.lin.weight"] = f"double_blocks.{i}.img_mod.linear.weight"
    layer_mapping[f"double_blocks.{i}.img_attn.q_proj.bias"] = f"double_blocks.{i}.img_attn.proj_q.bias"
    layer_mapping[f"double_blocks.{i}.img_attn.q_proj.weight"] = f"double_blocks.{i}.img_attn.proj_q.weight"
    layer_mapping[f"double_blocks.{i}.img_attn.k_proj.bias"] = f"double_blocks.{i}.img_attn.proj_k.bias"
    layer_mapping[f"double_blocks.{i}.img_attn.k_proj.weight"] = f"double_blocks.{i}.img_attn.proj_k.weight"
    layer_mapping[f"double_blocks.{i}.img_attn.v_proj.bias"] = f"double_blocks.{i}.img_attn.proj_v.bias"
    layer_mapping[f"double_blocks.{i}.img_attn.v_proj.weight"] = f"double_blocks.{i}.img_attn.proj_v.weight"
    layer_mapping[f"double_blocks.{i}.img_attn.proj.bias"] = f"double_blocks.{i}.img_attn.proj_out.bias"
    layer_mapping[f"double_blocks.{i}.img_attn.proj.weight"] = f"double_blocks.{i}.img_attn.proj_out.weight"
    layer_mapping[f"double_blocks.{i}.img_attn.norm.query_norm.scale"] = f"double_blocks.{i}.img_attn.q_norm.weight"
    layer_mapping[f"double_blocks.{i}.img_attn.norm.key_norm.scale"] = f"double_blocks.{i}.img_attn.k_norm.weight"
    layer_mapping[f"double_blocks.{i}.img_mlp.0.bias"] = f"double_blocks.{i}.img_mlp.fc1.bias"
    layer_mapping[f"double_blocks.{i}.img_mlp.0.weight"] = f"double_blocks.{i}.img_mlp.fc1.weight"
    layer_mapping[f"double_blocks.{i}.img_mlp.2.bias"] = f"double_blocks.{i}.img_mlp.fc2.bias"
    layer_mapping[f"double_blocks.{i}.img_mlp.2.weight"] = f"double_blocks.{i}.img_mlp.fc2.weight"

    layer_mapping[f"double_blocks.{i}.txt_mod.lin.bias"] = f"double_blocks.{i}.txt_mod.linear.bias"
    layer_mapping[f"double_blocks.{i}.txt_mod.lin.weight"] = f"double_blocks.{i}.txt_mod.linear.weight"
    layer_mapping[f"double_blocks.{i}.txt_attn.q_proj.bias"] = f"double_blocks.{i}.txt_attn.proj_q.bias"
    layer_mapping[f"double_blocks.{i}.txt_attn.q_proj.weight"] = f"double_blocks.{i}.txt_attn.proj_q.weight"
    layer_mapping[f"double_blocks.{i}.txt_attn.k_proj.bias"] = f"double_blocks.{i}.txt_attn.proj_k.bias"
    layer_mapping[f"double_blocks.{i}.txt_attn.k_proj.weight"] = f"double_blocks.{i}.txt_attn.proj_k.weight"
    layer_mapping[f"double_blocks.{i}.txt_attn.v_proj.bias"] = f"double_blocks.{i}.txt_attn.proj_v.bias"
    layer_mapping[f"double_blocks.{i}.txt_attn.v_proj.weight"] = f"double_blocks.{i}.txt_attn.proj_v.weight"
    layer_mapping[f"double_blocks.{i}.txt_attn.proj.bias"] = f"double_blocks.{i}.txt_attn.proj_out.bias"
    layer_mapping[f"double_blocks.{i}.txt_attn.proj.weight"] = f"double_blocks.{i}.txt_attn.proj_out.weight"
    layer_mapping[f"double_blocks.{i}.txt_attn.norm.query_norm.scale"] = f"double_blocks.{i}.txt_attn.q_norm.weight"
    layer_mapping[f"double_blocks.{i}.txt_attn.norm.key_norm.scale"] = f"double_blocks.{i}.txt_attn.k_norm.weight"
    layer_mapping[f"double_blocks.{i}.txt_mlp.0.bias"] = f"double_blocks.{i}.txt_mlp.fc1.bias"
    layer_mapping[f"double_blocks.{i}.txt_mlp.0.weight"] = f"double_blocks.{i}.txt_mlp.fc1.weight"
    layer_mapping[f"double_blocks.{i}.txt_mlp.2.bias"] = f"double_blocks.{i}.txt_mlp.fc2.bias"
    layer_mapping[f"double_blocks.{i}.txt_mlp.2.weight"] = f"double_blocks.{i}.txt_mlp.fc2.weight"

    return layer_mapping


def get_single_layer_mapping(i: int) -> Dict:
    layer_mapping = {}
    layer_mapping[f"single_blocks.{i}.norm.query_norm.scale"] = f"single_blocks.{i}.q_norm.weight"
    layer_mapping[f"single_blocks.{i}.norm.key_norm.scale"] = f"single_blocks.{i}.k_norm.weight"
    layer_mapping[f"single_blocks.{i}.modulation.lin.bias"] = f"single_blocks.{i}.modulation.linear.bias"
    layer_mapping[f"single_blocks.{i}.modulation.lin.weight"] = f"single_blocks.{i}.modulation.linear.weight"

    return layer_mapping


def replace_state_dict(
    state_dict: Dict[str, Any],
    convert_mapping: Dict,
):
    for ori_key, mm_key in convert_mapping.items():
        state_dict[mm_key] = state_dict.pop(ori_key)
    return state_dict


def split_by_tp(
    state_dict: Dict[str, Any],
    tp_size: int = 2,
    double_stream_layers: int = 19,
    single_stream_layers: int = 38,
    num_heads: int = 24,
    head_dim: int = 128
) -> List[Dict]:
    if tp_size <= 1:
        return [state_dict]

    raise Exception("TP is not supported, please set the tp_size to 1")


def save(state_dicts: List[Dict], save_dir: str, latest_checkpointed_iteration="release"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    flags = os.O_WRONLY | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(save_dir, 'latest_checkpointed_iteration.txt'), flags, mode), 'w') as fout:
        fout.write(latest_checkpointed_iteration)
    if latest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(latest_checkpointed_iteration)

    for tp_rank, state_dict in enumerate(state_dicts):
        os.makedirs(os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}"))
        save_path = os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
        save_dict = {}
        save_dict['model'] = state_dict
        torch.save(save_dict, save_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="./Open-Sora-v2/Open_Sora_v2.safetensors", help="Source path of checkpoint")
    parser.add_argument("--target_path", type=str, default="./opensora2.0/convert_ckpt/", help="Save path of MM checkpoint")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor model parallel world size")
    parser.add_argument("--double_stream_layers", type=int, default=19)
    parser.add_argument("--single_stream_layers", type=int, default=38)
    parser.add_argument("--num_heads", type=int, default=24)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--mode", type=str, default="split", choices=["split", "merge"],
        help="Split mode is used to split the pretrained weights according to tp_size before training, \
        and Merge mode is used to merge weights based on tp_size after training is completed")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "split":
        source_state_dict = load_file(args.source_path)

        for i in range(args.double_stream_layers):
            DIT_CONVERT_MAPPING.update(get_double_layer_mapping(i))
        for i in range(args.single_stream_layers):
            DIT_CONVERT_MAPPING.update(get_single_layer_mapping(i))

        state_dict = replace_state_dict(source_state_dict, convert_mapping=DIT_CONVERT_MAPPING)

        state_dicts = split_by_tp(
            state_dict,
            tp_size=args.tp_size,
            double_stream_layers=args.double_stream_layers,
            single_stream_layers=args.single_stream_layers,
            num_heads=args.num_heads,
            head_dim=args.head_dim
        )
        save(state_dicts, args.target_path)
    else:
        print("coming soon")
