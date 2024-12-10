import os
import stat
import copy
import argparse
from typing import Any, Dict, List
import torch


CONVERT_MAPPING = {
    "time_embed.0.bias": "time_embed.time_embed.0.bias",
    "time_embed.0.weight": "time_embed.time_embed.0.weight",
    "time_embed.2.bias": "time_embed.time_embed.2.bias",
    "time_embed.2.weight": "time_embed.time_embed.2.weight",
    "mixins.patch_embed.proj.bias": "patch_embed.proj.bias",
    "mixins.patch_embed.proj.weight": "patch_embed.proj.weight",
    "mixins.patch_embed.text_proj.bias": "caption_projection.bias",
    "mixins.patch_embed.text_proj.weight": "caption_projection.weight",
    "mixins.pos_embed.freqs_cos": "pos_embed.freqs_cos",
    "mixins.pos_embed.freqs_sin": "pos_embed.freqs_sin",
    # "mixins.pos_embed.pos_embedding": "pos_embed.pos_embedding" # only for i2v task
    "transformer.final_layernorm.weight": "norm_final.weight",
    "transformer.final_layernorm.bias": "norm_final.bias",
    "mixins.final_layer.norm_final.weight": "norm_out.weight",
    "mixins.final_layer.norm_final.bias": "norm_out.bias",
    "mixins.final_layer.linear.weight": "proj_out.weight",
    "mixins.final_layer.linear.bias": "proj_out.bias",
    "mixins.final_layer.adaLN_modulation.1.weight": "adaLN_modulation.1.weight",
    "mixins.final_layer.adaLN_modulation.1.bias": "adaLN_modulation.1.bias"
}


def update_state_dict_inplace(
    state_dict: Dict[str, Any],
    convert_mapping: Dict,
    prefix: str = "model.diffusion_model."
):
    for old_key in convert_mapping:
        new_key = convert_mapping[old_key]
        state_dict[new_key] = state_dict.pop(prefix + old_key)


def get_layer_mapping(i: int) -> Dict:
    layer_mapping = {}
    layer_mapping[f"mixins.adaln_layer.adaLN_modulations.{i}.1.bias"] = f"videodit_blocks.{i}.scale_shift_table.1.bias"
    layer_mapping[f"mixins.adaln_layer.adaLN_modulations.{i}.1.weight"] = f"videodit_blocks.{i}.scale_shift_table.1.weight"
    layer_mapping[f"mixins.adaln_layer.query_layernorm_list.{i}.bias"] = f"videodit_blocks.{i}.self_atten.q_norm.bias"
    layer_mapping[f"mixins.adaln_layer.query_layernorm_list.{i}.weight"] = f"videodit_blocks.{i}.self_atten.q_norm.weight"
    layer_mapping[f"mixins.adaln_layer.key_layernorm_list.{i}.bias"] = f"videodit_blocks.{i}.self_atten.k_norm.bias"
    layer_mapping[f"mixins.adaln_layer.key_layernorm_list.{i}.weight"] = f"videodit_blocks.{i}.self_atten.k_norm.weight"
    layer_mapping[f"transformer.layers.{i}.input_layernorm.bias"] = f"videodit_blocks.{i}.norm1.bias"
    layer_mapping[f"transformer.layers.{i}.input_layernorm.weight"] = f"videodit_blocks.{i}.norm1.weight"
    layer_mapping[f"transformer.layers.{i}.attention.dense.bias"] = f"videodit_blocks.{i}.self_atten.proj_out.bias"
    layer_mapping[f"transformer.layers.{i}.attention.dense.weight"] = f"videodit_blocks.{i}.self_atten.proj_out.weight"
    layer_mapping[f"transformer.layers.{i}.post_attention_layernorm.bias"] = f"videodit_blocks.{i}.norm2.bias"
    layer_mapping[f"transformer.layers.{i}.post_attention_layernorm.weight"] = f"videodit_blocks.{i}.norm2.weight"
    layer_mapping[f"transformer.layers.{i}.mlp.dense_h_to_4h.bias"] = f"videodit_blocks.{i}.ff.net.0.proj.bias"
    layer_mapping[f"transformer.layers.{i}.mlp.dense_h_to_4h.weight"] = f"videodit_blocks.{i}.ff.net.0.proj.weight"
    layer_mapping[f"transformer.layers.{i}.mlp.dense_4h_to_h.bias"] = f"videodit_blocks.{i}.ff.net.2.bias"
    layer_mapping[f"transformer.layers.{i}.mlp.dense_4h_to_h.weight"] = f"videodit_blocks.{i}.ff.net.2.weight"
    layer_mapping[f"transformer.layers.{i}.attention.query_key_value.weight"] = f"videodit_blocks.{i}.self_atten.proj_qkv.weight"
    layer_mapping[f"transformer.layers.{i}.attention.query_key_value.bias"] = f"videodit_blocks.{i}.self_atten.proj_qkv.bias"
    return layer_mapping


def remove_layers(
    state_dict: Dict[str, Any],
    remove_keys: List
):
    for remove_key in remove_keys:
        state_dict.pop(remove_key)


def split_by_tp(state_dict: Dict[str, Any], tp_size: int = 2, num_layers: int = 42) -> List[Dict]:
    if tp_size <= 1:
        return [state_dict, ]
    
    new_state_dicts = []
    for tp_rank in range(tp_size):
        new_state_dict = copy.deepcopy(state_dict)

        for index in range(num_layers):
            # ColumnParallelLinear
            suffixed_0 = [
                f"videodit_blocks.{index}.ff.net.0.proj.weight",
                f"videodit_blocks.{index}.ff.net.0.proj.bias",
                f"videodit_blocks.{index}.scale_shift_table.1.weight",
                f"videodit_blocks.{index}.scale_shift_table.1.bias"
            ]
            # RowParallelLinear
            suffixed_1 = [
                f"videodit_blocks.{index}.self_atten.proj_out.weight",
                f"videodit_blocks.{index}.ff.net.2.weight",
            ]
            # self_atten.proj_qkv
            suffixed_special = [
                f"videodit_blocks.{index}.self_atten.proj_qkv.weight",
                f"videodit_blocks.{index}.self_atten.proj_qkv.bias"
            ]

            for split_name in suffixed_0:
                new_state_dict[split_name] = torch.chunk(state_dict[split_name], tp_size, dim=0)[tp_rank]
            for split_name in suffixed_1:
                new_state_dict[split_name] = torch.chunk(state_dict[split_name], tp_size, dim=1)[tp_rank]
            for split_name in suffixed_special:
                wq, wk, wv = torch.chunk(state_dict[split_name], 3, dim=0)
                wq = torch.chunk(wq, tp_size, dim=0)[tp_rank]
                wk = torch.chunk(wk, tp_size, dim=0)[tp_rank]
                wv = torch.chunk(wv, tp_size, dim=0)[tp_rank]
                weight = torch.cat([wq, wk, wv], dim=0)
                new_state_dict[split_name] = weight
        new_state_dicts.append(new_state_dict)
    
    return new_state_dicts


def save_by_tp(state_dicts: List[Dict], save_dir: str, latest_checkpointed_iteration='release'):
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


def merge_by_tp(train_save_dir: str, save_path: str, num_layers: int, tp_size: int):
    flags = os.O_RDONLY
    mode = stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(train_save_dir, "latest_checkpointed_iteration.txt"), flags, mode)) as f:
        latest_checkpointed_iteration = f.readline()

    if latest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(latest_checkpointed_iteration)    

    _state_dicts = []
    for tp_rank in range(tp_size):
        state_dict_path = os.path.join(train_save_dir, directory, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
        _state_dicts.append(torch.load(state_dict_path)['model'])
    
    if tp_size == 1:
        torch.save(_state_dicts[0], save_path)
        return 

    merged_state_dict = copy.deepcopy(_state_dicts[0])
    for index in num_layers:
        # ColumnParallelLinear
        suffixed_0 = [
            f"videodit_blocks.{index}.ff.net.0.proj.weight",
            f"videodit_blocks.{index}.ff.net.0.proj.bias",
            f"videodit_blocks.{index}.scale_shift_table.1.weight",
            f"videodit_blocks.{index}.scale_shift_table.1.bias"
        ]
        # RowParallelLinear
        suffixed_1 = [
            f"videodit_blocks.{index}.self_atten.proj_out.weight",
            f"videodit_blocks.{index}.ff.net.2.weight",
        ]
        # self_atten.proj_qkv
        suffixed_special = [
            f"videodit_blocks.{index}.self_atten.proj_qkv.weight",
            f"videodit_blocks.{index}.self_atten.proj_qkv.bias"
        ]
        for name in suffixed_0:
            parameters = [_state_dicts[tp_rank][name] for tp_rank in range(tp_size)]
            parameters = torch.cat(parameters, dim=0)
            merged_state_dict[name] = parameters
        for name in suffixed_1:
            parameters = [_state_dicts[tp_rank][name] for tp_rank in range(tp_size)]
            parameters = torch.cat(parameters, dim=1)
            merged_state_dict[name] = parameters
        for name in suffixed_special:
            wq = [torch.chunk(_state_dicts[tp_rank][name], 3, dim=0)[0] for tp_rank in range(tp_size)]
            wk = [torch.chunk(_state_dicts[tp_rank][name], 3, dim=0)[1] for tp_rank in range(tp_size)]
            wv = [torch.chunk(_state_dicts[tp_rank][name], 3, dim=0)[2] for tp_rank in range(tp_size)]
            wq = torch.cat(wq, dim=0)
            wk = torch.cat(wk, dim=0)
            wv = torch.cat(wv, dim=0)
            wqkv = torch.cat([wq, wk, wv], dim=0)
            merged_state_dict[name] = wqkv
    torch.save(merged_state_dict, save_path)
    return 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor model parallel world size")
    parser.add_argument("--num_layers", type=int, default=42, help="Layer numbers of video_dit")
    parser.add_argument("--source_path", type=str, default="./transformer/1/mp_rank_00_model_states.pt", help="Source path of checkpoint")
    parser.add_argument("--target_path", type=str, default="./ckpt/sat_dit/", help="Save path of MM checkpoint")
    parser.add_argument("--task", type=str, default="t2v", choices=["t2v", "i2v"], help="Task type")
    parser.add_argument("--remove_pos_emb", type=bool, default=False, help="remove_pos_emb")
    parser.add_argument("--mode", type=str, default="split", choices=["split", "merge"], 
        help="Split mode is used to split the pretrained weights according to tp_size before training, \
        and Merge mode is used to merge weights based on tp_size after training is completed")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == 'split':
        source_state_dict = torch.load(args.source_path, map_location='cpu')['module']

        if args.task == "i2v":
            CONVERT_MAPPING.update({"mixins.pos_embed.pos_embedding": "pos_embed.pos_embedding"})

        # inplace state dict
        for i in range(args.num_layers):
            CONVERT_MAPPING.update(get_layer_mapping(i))
        update_state_dict_inplace(source_state_dict, CONVERT_MAPPING)

        # remove dummy layers
        remove_keys = set(source_state_dict.keys()) - set(CONVERT_MAPPING.values())
        remove_layers(source_state_dict, remove_keys)

        if args.remove_pos_emb:
            remove_layers(source_state_dict, ["pos_embed.freq_cos", "pos_embed.freq_sin"])
            if args.task == "i2v":
                remove_layers(source_state_dict, ["pos_embed.pos_embedding"])

        state_dicts = split_by_tp(source_state_dict, tp_size=args.tp_size, num_layers=args.num_layers)
        save_by_tp(state_dicts, args.target_path)
    
    elif args.mode == 'merge':
        merge_by_tp(args.source_path, args.target_path, args.num_layers, args.tp_size)