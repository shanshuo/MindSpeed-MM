import os
import stat
import argparse
from typing import Any, Dict, List
import torch

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration
)


DIT_CONVERT_MAPPING = {
    "txt_in.t_embedder.mlp.0.weight": "txt_in.t_embedder.time_embed.0.weight",
    "txt_in.t_embedder.mlp.0.bias": "txt_in.t_embedder.time_embed.0.bias",
    "txt_in.t_embedder.mlp.2.weight": "txt_in.t_embedder.time_embed.2.weight",
    "txt_in.t_embedder.mlp.2.bias": "txt_in.t_embedder.time_embed.2.bias",
    "time_in.mlp.0.weight": "time_in.time_embed.0.weight",
    "time_in.mlp.0.bias": "time_in.time_embed.0.bias",
    "time_in.mlp.2.weight": "time_in.time_embed.2.weight",
    "time_in.mlp.2.bias": "time_in.time_embed.2.bias",
    "vector_in.in_layer.weight": "vector_in.fc1.weight",
    "vector_in.in_layer.bias": "vector_in.fc1.bias",
    "vector_in.out_layer.weight": "vector_in.fc2.weight",
    "vector_in.out_layer.bias": "vector_in.fc2.bias",
    "guidance_in.mlp.0.weight": "guidance_in.time_embed.0.weight",
    "guidance_in.mlp.0.bias": "guidance_in.time_embed.0.bias",
    "guidance_in.mlp.2.weight": "guidance_in.time_embed.2.weight",
    "guidance_in.mlp.2.bias": "guidance_in.time_embed.2.bias",
    "final_layer.linear.weight": "proj_out.weight",
    "final_layer.linear.bias": "proj_out.bias",
    "final_layer.adaLN_modulation.1.weight": "adaLN_modulation.1.weight",
    "final_layer.adaLN_modulation.1.bias": "adaLN_modulation.1.bias"
}


def preprocess_text_encoder_tokenizer(source_dir, save_dir):
    processor = AutoProcessor.from_pretrained(source_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        source_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.language_model.save_pretrained(save_dir)
    processor.tokenizer.save_pretrained(save_dir)


def replace_state_dict(
        state_dict: Dict[str, Any],
        convert_mapping: Dict,
):
    for ori_key, mm_key in convert_mapping.items():
        state_dict[mm_key] = state_dict.pop(ori_key)
    return state_dict


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
    parser.add_argument("--module", type=str, choices=["dit", "text_encoder"], default="dit", help="The module to convert")
    parser.add_argument("--source_path", type=str, default="./transformers/mp_rank_00/model_states.pt", help="Source path of checkpoint")
    parser.add_argument("--target_path", type=str, default="./ckpt/hunyuanvideo/", help="Save path of MM checkpoint")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if args.module == "text_encoder":
        preprocess_text_encoder_tokenizer(args.source_path, args.target_path)
    else:
        source_state_dict = torch.load(args.source_path, map_location='cpu')['module']
        state_dict = replace_state_dict(source_state_dict, convert_mapping=DIT_CONVERT_MAPPING)
        save([state_dict], args.target_path)