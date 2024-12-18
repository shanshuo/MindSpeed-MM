import json
import os
import shutil
from pathlib import Path

import mindspeed.megatron_adaptor  # noqa
import torch
from safetensors.torch import save_file


def rename_pp_parameter(param_name: str, model_dir: Path, vit_pp_list: list[int], llm_pp_list: list[int]) -> str:
    index = int(model_dir.parent.stem.split('_')[-1])
    llm_pp_list = [sum(llm_pp_list[:i + 1]) for i in range(len(llm_pp_list))]
    vit_pp_list = [sum(vit_pp_list[:i + 1]) for i in range(len(vit_pp_list))]
    llm_pp_list = [0] + llm_pp_list[0:-1]
    vit_pp_list = [0] + vit_pp_list[0:-1]
    if param_name.startswith('image_encoder.encoder.blocks.layers'):
        # 比如pp_list = [0,0,10,20],当读取到最后一个.pt文件，此时index=3，该文件中存放的是第20到第27层
        index = vit_pp_list[index]
        name_li = param_name.split('.')
        name_li[4] = str(index + int(name_li[4]))
        param_name = '.'.join(name_li)
    elif param_name.startswith('text_decoder.decoder.layers'):
        # 比如pp_list = [0,0,10,20],当读取到最后一个.pt文件，此时index=3，该文件中存放的是第20到第27层
        index = llm_pp_list[index]
        name_li = param_name.split('.')
        name_li[3] = str(index + int(name_li[3]))
        param_name = '.'.join(name_li)
    return param_name


def load_from_mm(_load_dir, vit_pp_list, llm_pp_list):
    LATEST_TXT = "latest_checkpointed_iteration.txt"
    mm_save_dir = Path(_load_dir)
    save_iteration = mm_save_dir.joinpath(LATEST_TXT).read_text()
    save_iter_dir = mm_save_dir.joinpath(f"iter_{int(save_iteration):07}")
    state_dict = {}
    print(str(save_iter_dir).center(100, "="))
    for pt_path in save_iter_dir.glob("*/*.pt"):
        print(str(pt_path).center(100, '_'))
        state_dict.update(
            {rename_pp_parameter(param, pt_path, vit_pp_list, llm_pp_list): tensor
             for param, tensor in torch.load(pt_path, map_location='cpu')['model'].items()})
    for key, value in state_dict.items():
        print(key)

    return state_dict


def convert_mm_to_hf(_state_dict, _vit_hidden_size, _vit_attention_heads_num):
    hiddensize_per_head = _vit_hidden_size // _vit_attention_heads_num
    new_params = {}
    for key, value in _state_dict.items():
        new_key = None
        # image_encoder 权重转换部分
        if key.startswith('image_encoder'):
            new_key = key.replace('image_encoder.projector.encoder.linear_fc1', 'visual.merger.mlp.0')
            new_key = new_key.replace('image_encoder.projector.encoder.linear_fc2', 'visual.merger.mlp.2')
            new_key = new_key.replace('image_encoder.projector.layernorm', 'visual.merger.ln_q')
            new_key = new_key.replace('image_encoder.encoder.patch_embed.proj', 'visual.patch_embed.proj')

            new_key = new_key.replace('image_encoder.encoder.blocks.layers', 'visual.blocks')
            new_key = new_key.replace('self_attention.linear_proj', 'attn.proj')
            new_key = new_key.replace('self_attention.linear_qkv', 'attn.qkv')
            new_key = new_key.replace('mlp.linear_fc1', 'mlp.fc1')
            new_key = new_key.replace('mlp.linear_fc2', 'mlp.fc2')
            new_key = new_key.replace('input_layernorm', 'norm1')
            new_key = new_key.replace('pre_mlp_layernorm', 'norm2')
            if 'qkv.weight' in new_key:
                res = value * 0
                i = 0
                for j in range(_vit_attention_heads_num):
                    q_part = value[i * hiddensize_per_head: (i + 1) * hiddensize_per_head, :]
                    res[hiddensize_per_head * j: hiddensize_per_head * (j + 1), :] = q_part

                    k_part = value[(i + 1) * hiddensize_per_head: (i + 2) * hiddensize_per_head, :]
                    res[_vit_hidden_size + hiddensize_per_head * j: _vit_hidden_size + hiddensize_per_head * (j + 1),
                    :] = k_part

                    v_part = value[(i + 2) * hiddensize_per_head: (i + 3) * hiddensize_per_head, :]
                    res[_vit_hidden_size * 2 + hiddensize_per_head * j: _vit_hidden_size * 2 + hiddensize_per_head * (
                            j + 1), :] = v_part

                    i = i + 3
                new_params[new_key] = res
            elif 'qkv.bias' in new_key:
                res = value * 0
                i = 0
                for j in range(_vit_attention_heads_num):
                    q_part = value[i * hiddensize_per_head: (i + 1) * hiddensize_per_head]
                    res[hiddensize_per_head * j: hiddensize_per_head * (j + 1)] = q_part

                    k_part = value[(i + 1) * hiddensize_per_head: (i + 2) * hiddensize_per_head]
                    res[_vit_hidden_size + hiddensize_per_head * j: _vit_hidden_size + hiddensize_per_head * (
                            j + 1)] = k_part

                    v_part = value[(i + 2) * hiddensize_per_head: (i + 3) * hiddensize_per_head]
                    res[
                    _vit_hidden_size * 2 + hiddensize_per_head * j: _vit_hidden_size * 2 + hiddensize_per_head * (
                            j + 1)] = v_part

                    i = i + 3
                new_params[new_key] = res
            else:
                if value is not None:
                    new_params[new_key] = value

        else:
            if 'self_attention.linear_qkv.weight' in key:
                qkv_chunks = torch.chunk(value, 4, dim=0)
                # qkv的结构是[896(q)+128(k)+128(v)]*4
                indices = [896, 1024]
                indices = [0] + indices + [qkv_chunks[0].size(0)]
                q_chunks = []
                k_chunks = []
                v_chunks = []
                for j in range(4):
                    splits = [qkv_chunks[j][indices[i]:indices[i + 1]] for i in range(len(indices) - 1)]
                    q_chunks.append(splits[0])
                    k_chunks.append(splits[1])
                    v_chunks.append(splits[2])

                attention_q_weight = torch.cat(q_chunks, dim=0)
                attention_k_weight = torch.cat(k_chunks, dim=0)
                attention_v_weight = torch.cat(v_chunks, dim=0)

                layer = key.split('.')[3]
                attention_q = f'model.layers.{layer}.self_attn.q_proj.weight'
                attention_k = f'model.layers.{layer}.self_attn.k_proj.weight'
                attention_v = f'model.layers.{layer}.self_attn.v_proj.weight'

                new_params[attention_q] = attention_q_weight
                new_params[attention_k] = attention_k_weight
                new_params[attention_v] = attention_v_weight

            elif 'self_attention.linear_qkv.bias' in key:
                qkv_chunks = torch.chunk(value, 4, dim=0)
                # qkv的结构是[896(q)+128(k)+128(v)]*4
                indices = [896, 1024]
                indices = [0] + indices + [qkv_chunks[0].size(0)]
                q_chunks = []
                k_chunks = []
                v_chunks = []
                for j in range(4):
                    splits = [qkv_chunks[j][indices[i]:indices[i + 1]] for i in range(len(indices) - 1)]
                    q_chunks.append(splits[0])
                    k_chunks.append(splits[1])
                    v_chunks.append(splits[2])

                attention_q_weight = torch.cat(q_chunks, dim=0)
                attention_k_weight = torch.cat(k_chunks, dim=0)
                attention_v_weight = torch.cat(v_chunks, dim=0)

                layer = key.split('.')[3]
                attention_q = f'model.layers.{layer}.self_attn.q_proj.bias'
                attention_k = f'model.layers.{layer}.self_attn.k_proj.bias'
                attention_v = f'model.layers.{layer}.self_attn.v_proj.bias'

                new_params[attention_q] = attention_q_weight
                new_params[attention_k] = attention_k_weight
                new_params[attention_v] = attention_v_weight

            elif 'mlp.linear_fc1.weight' in key:
                gate_up_chunks = torch.chunk(value, 2, dim=0)
                layer = key.split('.')[3]
                attention_gate = f'model.layers.{layer}.mlp.gate_proj.weight'
                attention_up = f'model.layers.{layer}.mlp.up_proj.weight'

                new_params[attention_gate] = gate_up_chunks[0]
                new_params[attention_up] = gate_up_chunks[1]

            elif key.startswith('text_decoder.output_layer'):
                new_key = key.replace('text_decoder.output_layer', 'lm_head')
                if value is not None:
                    new_params[new_key] = value
            elif key == 'text_decoder.embedding.word_embeddings.weight':
                new_key = key.replace('text_decoder.embedding.word_embeddings.weight', 'model.embed_tokens.weight')
                if value is not None:
                    new_params[new_key] = value
            elif key == 'text_decoder.decoder.final_layernorm.weight':
                new_key = key.replace('text_decoder.decoder.final_layernorm.weight', 'model.norm.weight')
                if value is not None:
                    new_params[new_key] = value
            elif key.startswith('text_decoder.decoder.layers'):
                new_key = key.replace('text_decoder.decoder.layers', 'model.layers')
                new_key = new_key.replace('self_attention.linear_proj.weight', 'self_attn.o_proj.weight')
                new_key = new_key.replace('pre_mlp_layernorm.weight', 'post_attention_layernorm.weight')
                new_key = new_key.replace('mlp.linear_fc2.weight', 'mlp.down_proj.weight')

                new_params[new_key] = value

    return new_params


def copy_except_safetensors(src_dir: str, dst_dir: str) -> None:
    # 如果目标目录不存在，则创建它
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 遍历源目录及其子目录
    for root, dirs, files in os.walk(src_dir):
        # 构造目标路径
        dst_root = os.path.join(dst_dir, os.path.relpath(root, src_dir))

        # 复制文件，但排除 .safetensors 文件
        for file in files:
            if not file.endswith('.safetensors'):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_root, file)
                # 确保目标文件的父目录存在
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)


def check_pp_config(_pp_size, _vit_num_layers, _vit_pipeline_num_layers, _llm_num_layers, _llm_pipeline_num_layers):
    if len(_vit_pipeline_num_layers) != _pp_size:
        raise AssertionError(f'length of vit_pipeline_num_layers must be equal to pp_size, '
                             f'but got {len(_vit_pipeline_num_layers)} and {_pp_size}.')
    if sum(_vit_pipeline_num_layers) != _vit_num_layers:
        raise AssertionError(f'sum of vit_pipeline_num_layers must be equal to vit_num_layers, '
                             f'but got {sum(_vit_pipeline_num_layers)} and {_vit_num_layers}.')
    if len(_llm_pipeline_num_layers) != _pp_size:
        raise AssertionError(f'length of llm_pipeline_num_layers must be equal to pp_size, '
                             f'but got {len(_llm_pipeline_num_layers)} and {_pp_size}.')
    if sum(_llm_pipeline_num_layers) != _llm_num_layers:
        raise AssertionError(f'sum of llm_pipeline_num_layers must be equal to llm_num_layers, '
                             f'but got {sum(_llm_pipeline_num_layers)} and {_llm_num_layers}.')


def split_by_index_json(_state_dict, _model_path):
    index_json_path = os.path.join(_model_path, 'model.safetensors.index.json')
    return_dicts = []
    with open(index_json_path, 'r', encoding='utf-8') as file:
        weight_map = json.load(file)['weight_map']
    for key, value in weight_map.items():
        index = int(value.split('-')[1])
        while index > len(return_dicts):
            return_dicts.append({})
        return_dicts[index - 1][key] = _state_dict[key]
    return return_dicts


def save_by_index_json(_state_dicts, _save_dir):
    metadata = {
        'format': 'pt'
    }
    for index, state_dict in enumerate(_state_dicts, start=1):
        name = f'model-{index:05}-of-{len(_state_dicts):05}.safetensors'
        save_file(state_dict, Path(_save_dir).joinpath(name), metadata=metadata)


if __name__ == "__main__":
    mm_save_dir = "save_dir"                # 微调后保存的权重目录
    hg_save_dir = "Qwen2-VL-7B-Save"        # 希望保存的hf目录
    model_path = "Qwen2-VL-7B-Instruct"     # hf原仓目录

    pp_size = 4
    vit_num_layers = 32
    vit_pipeline_num_layers = [32, 0, 0, 0]
    llm_num_layers = 28
    llm_pipeline_num_layers = [1, 6, 11, 10]

    vit_hidden_size = 1280
    vit_attention_heads_num = 16

    check_pp_config(pp_size, vit_num_layers, vit_pipeline_num_layers, llm_num_layers, llm_pipeline_num_layers)
    state_dict = load_from_mm(mm_save_dir, vit_pipeline_num_layers, llm_pipeline_num_layers)
    state_dict = convert_mm_to_hf(state_dict, vit_hidden_size, vit_attention_heads_num)
    state_dicts = split_by_index_json(state_dict, model_path)
    copy_except_safetensors(model_path, hg_save_dir)
    save_by_index_json(state_dicts, hg_save_dir)
