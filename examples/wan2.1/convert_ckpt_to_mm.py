import argparse
import os
import stat
from typing import Any, Dict, List

import torch
from safetensors.torch import load_file as safe_load

DIT_CONVERSION_MAPPING = {
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.linear_1.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.linear_1.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.linear_2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.linear_2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight"
}


def load_weight(weight_path):
    if weight_path.endswith(".safetensors"):
        return safe_load(weight_path)
    else:
        return torch.load(weight_path, map_location="cpu")


def load_from_hf(load_path):
    ckpt_dict = {}
    if os.path.isdir(load_path):
        for filename in os.listdir(load_path):
            file_path = os.path.join(load_path, filename)
            if os.path.isfile(file_path) and os.path.splitext(file_path)[1] in [
                ".pt",
                ".pth",
                ".ckpt",
                ".safetensors",
            ]:
                ckpt_dict_part = load_weight(file_path)
                ckpt_dict.update(ckpt_dict_part)
    elif os.path.isfile(load_path):
        ckpt_dict = load_weight(load_path)
    else:
        raise ValueError(f"Invalid path: {load_path}")
    return ckpt_dict


def replace_state_dict(
    state_dict: Dict[str, Any],
    conversion_mapping: Dict,
):
    for ori_key, mm_key in conversion_mapping.items():
        state_dict[mm_key] = state_dict.pop(ori_key)
    return state_dict


def convert_attn_to_mm(state_dict):
    new_checkpoint = {}
    state_dict = state_dict.get("blocks", state_dict)

    for key, value in state_dict.items():
        new_key = key.replace("attn1.norm_q", "self_attn.q_norm")
        new_key = new_key.replace("attn1.norm_k", "self_attn.k_norm")
        new_key = new_key.replace("attn2.norm_q", "cross_attn.q_norm")
        new_key = new_key.replace("attn2.norm_k", "cross_attn.k_norm")
        new_key = new_key.replace("attn1.to_q.", "self_attn.proj_q.")
        new_key = new_key.replace("attn1.to_k.", "self_attn.proj_k.")
        new_key = new_key.replace("attn1.to_v.", "self_attn.proj_v.")
        new_key = new_key.replace("attn1.to_out.0.", "self_attn.proj_out.")
        new_key = new_key.replace("attn2.to_q.", "cross_attn.proj_q.")
        new_key = new_key.replace("attn2.to_k.", "cross_attn.proj_k.")
        new_key = new_key.replace("attn2.to_v.", "cross_attn.proj_v.")
        new_key = new_key.replace("attn2.to_out.0.", "cross_attn.proj_out.")
        new_key = new_key.replace(".ffn.net.0.proj.", ".ffn.0.")
        new_key = new_key.replace(".ffn.net.2.", ".ffn.2.")
        new_key = new_key.replace("scale_shift_table", "modulation")
        new_key = new_key.replace(".norm2.", ".norm3.")

        new_checkpoint[new_key] = value

    return new_checkpoint


def save(
    state_dicts: List[Dict], save_dir: str, latest_checkpointed_iteration="release"
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    flags = os.O_WRONLY | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(
        os.open(
            os.path.join(save_dir, "latest_checkpointed_iteration.txt"), flags, mode
        ),
        "w",
    ) as fout:
        fout.write(latest_checkpointed_iteration)
    if latest_checkpointed_iteration == "release":
        directory = "release"
    else:
        directory = "iter_{:07d}".format(latest_checkpointed_iteration)

    os.makedirs(os.path.join(save_dir, directory, f"mp_rank_00"))
    save_path = os.path.join(save_dir, directory, f"mp_rank_00", "model_optim_rng.pt")
    save_dict = {}
    save_dict["model"] = state_dicts
    torch.save(save_dict, save_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path",
        type=str,
        default="./Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        help="Source path of checkpoint",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="./ckpt/wan2.1/",
        help="Save path of MM checkpoint",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    orig_dit_state_dict = load_from_hf(args.source_path)
    dit_state_dicts = replace_state_dict(
        orig_dit_state_dict, conversion_mapping=DIT_CONVERSION_MAPPING
    )
    dit_state_dicts = convert_attn_to_mm(dit_state_dicts)

    save(dit_state_dicts, args.target_path)
