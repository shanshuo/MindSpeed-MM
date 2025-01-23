#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright:   Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
@File    : qwen2vl_mm_to_hf.py
@Time    : 2025/01/14
@Desc    : qwen2vl mindspeed-mm模型转换成huggingface模型
"""
import json
import shutil
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import save_file
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
# 注意mindspeed-mm训练后保存的checkpoint中存储了patch相关信息，在load时需要加下面这行以支持反序列化
import mindspeed.megatron_adaptor # noqa

from checkpoint.utils import LATEST_TXT, ConvertHFConfig


def rename_pp_parameter(param_name: str,
                        vit_pp_list: list[int],
                        llm_pp_list: list[int],
                        pp_index: int = 0) -> str:
    index = pp_index
    llm_pp_list = [sum(llm_pp_list[:i + 1]) for i in range(len(llm_pp_list))]
    vit_pp_list = [sum(vit_pp_list[:i + 1]) for i in range(len(vit_pp_list))]
    llm_pp_list = [0] + llm_pp_list[0:-1]
    vit_pp_list = [0] + vit_pp_list[0:-1]
    if param_name.startswith('image_encoder.encoder.blocks.layers'):
        index = vit_pp_list[index]
        name_li = param_name.split('.')
        name_li[4] = str(index + int(name_li[4]))
        param_name = '.'.join(name_li)
    elif param_name.startswith('text_decoder.decoder.layers'):
        index = llm_pp_list[index]
        name_li = param_name.split('.')
        name_li[3] = str(index + int(name_li[3]))
        param_name = '.'.join(name_li)
    return param_name


def load_from_mm(load_dir: Path, vit_pp_list: list[int], llm_pp_list: list[int], tp_size: int = 1) -> list[dict]:
    save_iteration = load_dir.joinpath(LATEST_TXT).read_text()
    save_dir = load_dir.joinpath(f"iter_{int(save_iteration):07}" if save_iteration != "release" else save_iteration)
    state_dicts = []
    for tp_rank in range(tp_size):
        pp_state_dict = {}
        for pp_rank in range(len(vit_pp_list)):
            if len(vit_pp_list) > 1:
                current_path = save_dir.joinpath(f"mp_rank_{int(tp_rank):02}_{int(pp_rank):03}")
            else:
                current_path = save_dir.joinpath(f"mp_rank_{int(tp_rank):02}")
            pt_path = current_path.joinpath("model_optim_rng.pt")
            print(str(pt_path).center(100, '_'))
            # 注意output_layer存在_extra_state其值为None
            pp_state_dict.update(
                {rename_pp_parameter(param, vit_pp_list, llm_pp_list, pp_rank): tensor
                 for param, tensor in torch.load(pt_path, map_location='cpu')['model'].items() if tensor is not None})
        state_dicts.append(pp_state_dict)

    return state_dicts


def merge_by_tp(_state_dicts: list[dict[str, torch.Tensor]], _tp_size: int) -> dict:
    if len(_state_dicts) == 0:
        raise AssertionError(f'_state_dicts is empty.')
    if len(_state_dicts) == 1:
        return _state_dicts[0]
    return_state_dict = {}
    for key, value in _state_dicts[0].items():
        if key.startswith('text_decoder.decoder.layer') and 'linear_fc1.weight' in key:
            chunks_0 = [torch.chunk(_state_dicts[i][key], 2, dim=0) for i in range(_tp_size)]
            flattened_tensors = [pair[i] for i in range(2) for pair in chunks_0]
            return_state_dict[key] = torch.cat(flattened_tensors, dim=0)
        elif 'linear_qkv.weight' in key or 'linear_fc1.weight' in key:
            return_state_dict[key] = torch.cat([_state_dicts[i][key] for i in range(_tp_size)], dim=0)
        elif 'linear_qkv.bias' in key or 'linear_fc1.bias' in key:
            return_state_dict[key] = torch.cat([_state_dicts[i][key] for i in range(_tp_size)], dim=0)
        elif 'linear_proj.weight' in key or 'linear_fc2.weight' in key:
            return_state_dict[key] = torch.cat([_state_dicts[i][key] for i in range(_tp_size)], dim=1)
        elif 'output_layer' in key or 'word_embeddings' in key:
            return_state_dict[key] = torch.cat([_state_dicts[i][key] for i in range(_tp_size)], dim=0)
        else:
            return_state_dict[key] = value
    return return_state_dict


def convert_mm_to_hf(_state_dict: dict[str, torch.Tensor], cfg: Qwen2VLConfig) -> dict:
    vit_head_hidden_size = cfg.vision_config.embed_dim // cfg.vision_config.num_heads
    llm_head_hidden_size = cfg.hidden_size // cfg.num_attention_heads
    q_size = llm_head_hidden_size * cfg.num_attention_heads // cfg.num_key_value_heads
    k_size = llm_head_hidden_size * cfg.num_key_value_heads // cfg.num_key_value_heads
    v_size = llm_head_hidden_size * cfg.num_key_value_heads // cfg.num_key_value_heads

    new_params = {}
    for key, value in _state_dict.items():
        if value is None:
            continue
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
                for j in range(cfg.vision_config.num_heads):
                    q_part = value[i * vit_head_hidden_size: (i + 1) * vit_head_hidden_size, :]
                    res[vit_head_hidden_size * j: vit_head_hidden_size * (j + 1), :] = q_part

                    k_part = value[(i + 1) * vit_head_hidden_size: (i + 2) * vit_head_hidden_size, :]
                    res[
                    cfg.vision_config.embed_dim + vit_head_hidden_size * j: cfg.vision_config.embed_dim + vit_head_hidden_size * (
                            j + 1),
                    :] = k_part

                    v_part = value[(i + 2) * vit_head_hidden_size: (i + 3) * vit_head_hidden_size, :]
                    res[
                    cfg.vision_config.embed_dim * 2 + vit_head_hidden_size * j: cfg.vision_config.embed_dim * 2 + vit_head_hidden_size * (
                            j + 1), :] = v_part

                    i = i + 3
                new_params[new_key] = res
            elif 'qkv.bias' in new_key:
                res = value * 0
                i = 0
                for j in range(cfg.vision_config.num_heads):
                    q_part = value[i * vit_head_hidden_size: (i + 1) * vit_head_hidden_size]
                    res[vit_head_hidden_size * j: vit_head_hidden_size * (j + 1)] = q_part

                    k_part = value[(i + 1) * vit_head_hidden_size: (i + 2) * vit_head_hidden_size]
                    res[
                    cfg.vision_config.embed_dim + vit_head_hidden_size * j: cfg.vision_config.embed_dim + vit_head_hidden_size * (
                            j + 1)] = k_part

                    v_part = value[(i + 2) * vit_head_hidden_size: (i + 3) * vit_head_hidden_size]
                    res[
                    cfg.vision_config.embed_dim * 2 + vit_head_hidden_size * j: cfg.vision_config.embed_dim * 2 + vit_head_hidden_size * (
                            j + 1)] = v_part

                    i = i + 3
                new_params[new_key] = res
            else:
                if value is not None:
                    new_params[new_key] = value

        else:
            # self_attention.linear_qkv.weight 和 self_attention.linear_qkv.bias
            if 'self_attention.linear_qkv' in key:
                qkv_chunks = torch.chunk(value, cfg.num_key_value_heads, dim=0)
                q_chunks = []
                k_chunks = []
                v_chunks = []
                for chunk in qkv_chunks:
                    q_chunk, k_chunk, v_chunk = torch.split(chunk, [q_size, k_size, v_size], dim=0)
                    q_chunks.append(q_chunk)
                    k_chunks.append(k_chunk)
                    v_chunks.append(v_chunk)

                attention_q_weight = torch.cat(q_chunks, dim=0)
                attention_k_weight = torch.cat(k_chunks, dim=0)
                attention_v_weight = torch.cat(v_chunks, dim=0)

                layer = key.split('.')[3]
                name = key.split('.')[-1]  # weight或bias
                attention_q = f'model.layers.{layer}.self_attn.q_proj.{name}'
                attention_k = f'model.layers.{layer}.self_attn.k_proj.{name}'
                attention_v = f'model.layers.{layer}.self_attn.v_proj.{name}'

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


def copy_files_except_suffix(source_path: Path, target_path: Path, except_suffix: str = '.safetensors'):
    """拷贝源路径下除了以except_suffix为后缀的其他所有文件到目标路径，包含子目录"""
    target_path.mkdir(parents=True, exist_ok=True)
    for item in source_path.rglob('*'):
        if item.is_file() and item.suffix != except_suffix:
            relative_path = item.relative_to(source_path)
            destination = target_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
            print(f"Copied: {item} -> {destination}")


def split_by_index_json(state_dict: dict[str, torch.Tensor], hf_dir: Path) -> list[dict[str, torch.Tensor]]:
    index_json_path = hf_dir.joinpath('model.safetensors.index.json')
    return_dicts = []
    weight_map = json.loads(index_json_path.read_text()).get('weight_map', {})
    for key, value in weight_map.items():
        index = int(value.split('-')[1])
        while index > len(return_dicts):
            return_dicts.append({})
        return_dicts[index - 1][key] = state_dict[key]
    return return_dicts


def save_by_index_json(state_dicts: list[dict], save_dir: Path) -> None:
    metadata = {
        'format': 'pt'
    }
    for index, state_dict in enumerate(state_dicts, start=1):
        name = f'model-{index:05}-of-{len(state_dicts):05}.safetensors'
        save_file(state_dict, Path(save_dir).joinpath(name), metadata=metadata)


def main(convert_config: ConvertHFConfig):
    # qwen2vl获取到的config类型是Qn2VLConfig
    config = cast(Qwen2VLConfig, convert_config.hf_config.config)
    parallel_config = convert_config.parallel_config
    state_dicts = load_from_mm(convert_config.mm_dir, parallel_config.vit_pp_layers, parallel_config.llm_pp_layers,
                               parallel_config.tp_size)
    state_dict = merge_by_tp(state_dicts, parallel_config.tp_size)
    state_dict = convert_mm_to_hf(state_dict, config)
    state_dicts = split_by_index_json(state_dict, convert_config.hf_config.hf_dir)
    copy_files_except_suffix(convert_config.hf_config.hf_dir, convert_config.save_hf_dir)
    save_by_index_json(state_dicts, convert_config.save_hf_dir)
