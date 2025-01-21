#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright:   Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
@File    : qwen2vl_hf_to_mm.py
@Time    : 2025/01/14
@Desc    : qwen2vl huggingface模型转换成mindspeed-mm模型

huggingface模型目录：
Qwen2-VL-7B-Instruct/
├── chat_template.json
├── config.json
├── configuration.json
├── generation_config.json
├── LICENSE
├── merges.txt
├── model-00001-of-00005.safetensors
├── model-00002-of-00005.safetensors
├── model-00003-of-00005.safetensors
├── model-00004-of-00005.safetensors
├── model-00005-of-00005.safetensors
├── model.safetensors.index.json
├── preprocessor_config.json
├── README.md
├── tokenizer_config.json
├── tokenizer.json
└── vocab.json

mindspeed-mm模型目录：
Qwen2-VL-7B-Instruct/
├── latest_checkpointed_iteration.txt
└── release
    ├── mp_rank_00_000
    │    └── model_optim_rng.pt
    ├── mp_rank_00_001
    │    └── model_optim_rng.pt
    ├── mp_rank_00_002
    │    └── model_optim_rng.pt
    └── mp_rank_00_003
        └── model_optim_rng.pt

"""
from copy import deepcopy
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig

from checkpoint.utils import LATEST_TXT, ConvertMMConfig


def load_from_hf(hf_dir: Path) -> dict[str, torch.Tensor]:
    # 注意AutoModel.from_pretrained转换成模型对象时，存在torch_dtype问题需确认，因此这里直接读取safetensors确保dtype一致
    files = list(hf_dir.glob("*.safetensors"))
    state_dict = {}
    for safe_path in files:
        state_dict.update(load_file(str(safe_path), device='cpu'))
    return state_dict


def convert_hf_to_mm(_state_dict: dict[str, torch.Tensor],
                     _num_layers: int,
                     _vit_hidden_size: int,
                     _vit_attention_heads_num: int,
                     _llm_num_query_groups: int) -> dict[str, torch.Tensor]:
    vit_head_dim = _vit_hidden_size // _vit_attention_heads_num
    new_params = {}
    for key, value in tqdm(_state_dict.items(), desc="convert weights"):
        new_key = None
        # visual 权重转换部分
        if key.startswith('visual'):
            if 'merger' in key:
                if 'visual.merger.mlp.0' in key:
                    new_key = key.replace('visual.merger.mlp.0', 'image_encoder.projector.encoder.linear_fc1')
                if 'visual.merger.mlp.2' in key:
                    new_key = key.replace('visual.merger.mlp.2', 'image_encoder.projector.encoder.linear_fc2')
                if 'ln_q' in key:
                    new_key = key.replace('visual.merger.ln_q', 'image_encoder.projector.layernorm')
            if 'patch_embed' in key:
                new_key = key.replace('visual.patch_embed.proj', 'image_encoder.encoder.patch_embed.proj')
            if 'blocks' in key:
                new_key = key.replace('visual.blocks', 'image_encoder.encoder.blocks.layers')
                new_key = new_key.replace('attn.proj', 'self_attention.linear_proj')
                new_key = new_key.replace('attn.qkv', 'self_attention.linear_qkv')
                new_key = new_key.replace('mlp.fc1', 'mlp.linear_fc1')
                new_key = new_key.replace('mlp.fc2', 'mlp.linear_fc2')
                new_key = new_key.replace('norm1', 'input_layernorm')
                new_key = new_key.replace('norm2', 'pre_mlp_layernorm')

            if 'qkv.weight' in key:
                res = value * 0
                q_ = value[:_vit_hidden_size, :]
                k_ = value[_vit_hidden_size:_vit_hidden_size * 2, :]
                v_ = value[_vit_hidden_size * 2:_vit_hidden_size * 3, :]
                i = 0
                for j in range(_vit_attention_heads_num):
                    res[i * vit_head_dim:(i + 1) * vit_head_dim, :] = q_[j * vit_head_dim:(
                                                                                                  j + 1) * vit_head_dim,
                                                                      :]
                    res[(i + 1) * vit_head_dim:(i + 2) * vit_head_dim, :] = k_[j * vit_head_dim:(
                                                                                                        j + 1) * vit_head_dim,
                                                                            :]
                    res[(i + 2) * vit_head_dim:(i + 3) * vit_head_dim, :] = v_[j * vit_head_dim:(
                                                                                                        j + 1) * vit_head_dim,
                                                                            :]
                    i = i + 3
                new_params[new_key] = res

            elif 'qkv.bias' in key:
                res = value * 0
                q_ = value[:_vit_hidden_size]
                k_ = value[_vit_hidden_size:_vit_hidden_size * 2]
                v_ = value[_vit_hidden_size * 2:_vit_hidden_size * 3]

                i = 0
                for j in range(_vit_attention_heads_num):
                    res[i * vit_head_dim:(i + 1) * vit_head_dim] = q_[j * vit_head_dim:(
                                                                                               j + 1) * vit_head_dim]
                    res[(i + 1) * vit_head_dim:(i + 2) * vit_head_dim] = k_[j * vit_head_dim:(
                                                                                                     j + 1) * vit_head_dim]
                    res[(i + 2) * vit_head_dim:(i + 3) * vit_head_dim] = v_[j * vit_head_dim:(
                                                                                                     j + 1) * vit_head_dim]
                    i = i + 3
                new_params[new_key] = res
            else:
                new_params[new_key] = value
        else:
            if key.startswith('lm_head'):
                new_key = key.replace('lm_head', 'text_decoder.output_layer')
            elif key.startswith('model'):
                new_key = key.replace('model.layers', 'text_decoder.decoder.layers')
                new_key = new_key.replace('self_attn.o_proj.weight', 'self_attention.linear_proj.weight')
                new_key = new_key.replace('self_attn.q_proj.weight', 'self_attention.linear_q.weight')
                new_key = new_key.replace('self_attn.k_proj.weight', 'self_attention.linear_k.weight')
                new_key = new_key.replace('self_attn.v_proj.weight', 'self_attention.linear_v.weight')
                new_key = new_key.replace('self_attn.q_proj.bias', 'self_attention.linear_q.bias')
                new_key = new_key.replace('self_attn.k_proj.bias', 'self_attention.linear_k.bias')
                new_key = new_key.replace('self_attn.v_proj.bias', 'self_attention.linear_v.bias')
                new_key = new_key.replace('post_attention_layernorm.weight', 'pre_mlp_layernorm.weight')
                new_key = new_key.replace('mlp.gate_proj.weight', 'mlp.linear_fc1_gate.weight')
                new_key = new_key.replace('mlp.up_proj.weight', 'mlp.linear_fc1_up.weight')
                new_key = new_key.replace('mlp.down_proj.weight', 'mlp.linear_fc2.weight')
                new_key = new_key.replace('model.norm.weight', 'text_decoder.decoder.final_layernorm.weight')
                new_key = new_key.replace('model.embed_tokens.weight', 'text_decoder.embedding.word_embeddings.weight')

            new_params[new_key] = value
    for i in range(_num_layers):
        # 合并gate up
        gate_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1_gate.weight'
        up_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1_up.weight'
        fc1_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1.weight'
        # 如果权重名字在新字典中，则获取对应权重值
        # 合并 w1 和 w3
        if gate_name in new_params.keys():
            gate_proj_weight = new_params[gate_name]
        if up_name in new_params.keys():
            up_proj_weight = new_params[up_name]
        # 将 w1 和 w3 沿着第0维度进行拼接
        linear_fc1 = torch.cat([gate_proj_weight, up_proj_weight], dim=0)

        new_params[fc1_name] = linear_fc1
        # 移除合并前的权重
        if gate_name in new_params:
            new_params.pop(gate_name)
        if up_name in new_params:
            new_params.pop(up_name)

    for i in range(_num_layers):
        # 合并q k v weight
        attention_q = f'text_decoder.decoder.layers.{i}.self_attention.linear_q.weight'
        attention_k = f'text_decoder.decoder.layers.{i}.self_attention.linear_k.weight'
        attention_v = f'text_decoder.decoder.layers.{i}.self_attention.linear_v.weight'
        attention_qkv = f'text_decoder.decoder.layers.{i}.self_attention.linear_qkv.weight'
        if attention_q in new_params.keys():
            attention_q_weight = new_params[attention_q]
        if attention_k in new_params.keys():
            attention_k_weight = new_params[attention_k]
        if attention_v in new_params.keys():
            attention_v_weight = new_params[attention_v]

        q_chunks = torch.chunk(attention_q_weight, _llm_num_query_groups, dim=0)
        k_chunks = torch.chunk(attention_k_weight, _llm_num_query_groups, dim=0)
        v_chunks = torch.chunk(attention_v_weight, _llm_num_query_groups, dim=0)
        all_chunks = []
        for j in range(_llm_num_query_groups):
            all_chunks.append(q_chunks[j])
            all_chunks.append(k_chunks[j])
            all_chunks.append(v_chunks[j])
        concatenated_tensor = torch.cat(all_chunks, dim=0)
        new_params[attention_qkv] = concatenated_tensor
        if attention_q in new_params:
            new_params.pop(attention_q)
        if attention_k in new_params:
            new_params.pop(attention_k)
        if attention_v in new_params:
            new_params.pop(attention_v)

    for i in range(_num_layers):
        # 合并q k v bias
        attention_q1 = f'text_decoder.decoder.layers.{i}.self_attention.linear_q.bias'
        attention_k1 = f'text_decoder.decoder.layers.{i}.self_attention.linear_k.bias'
        attention_v1 = f'text_decoder.decoder.layers.{i}.self_attention.linear_v.bias'
        attention_qkv1 = f'text_decoder.decoder.layers.{i}.self_attention.linear_qkv.bias'
        if attention_q1 in new_params.keys():
            attention_q_bias = new_params[attention_q1]
        else:
            continue
        if attention_k1 in new_params.keys():
            attention_k_bias = new_params[attention_k1]
        else:
            continue
        if attention_v1 in new_params.keys():
            attention_v_bias = new_params[attention_v1]
        else:
            continue

        q_chunks1 = torch.chunk(attention_q_bias, _llm_num_query_groups, dim=0)
        k_chunks1 = torch.chunk(attention_k_bias, _llm_num_query_groups, dim=0)
        v_chunks1 = torch.chunk(attention_v_bias, _llm_num_query_groups, dim=0)
        all_chunks1 = []
        for j in range(_llm_num_query_groups):
            all_chunks1.append(q_chunks1[j])
            all_chunks1.append(k_chunks1[j])
            all_chunks1.append(v_chunks1[j])
        concatenated_tensor1 = torch.cat(all_chunks1, dim=0)
        new_params[attention_qkv1] = concatenated_tensor1
        if attention_q1 in new_params:
            new_params.pop(attention_q1)
        if attention_k1 in new_params:
            new_params.pop(attention_k1)
        if attention_v1 in new_params:
            new_params.pop(attention_v1)

    return new_params


def merge_pp_index(vit_pipeline_num_layers: list[int], llm_pipeline_num_layers: list[int]) -> list[tuple[int, int]]:
    """返回每张卡上vit和llm各自的层数"""
    split_method = []
    for vit_num, llm_num in zip(vit_pipeline_num_layers, llm_pipeline_num_layers):
        split_method.append((vit_num, llm_num))
    return split_method


def split_model_by_pipeline(state_dict: dict[str, torch.Tensor],
                            pp_split: list[tuple[int, int]]) -> list[dict[str, torch.Tensor]]:
    if len(pp_split) <= 1:
        return [state_dict]

    pp_size = len(pp_split)
    vit_range = [0, 0]
    llm_range = [pp_size - 1, pp_size - 1]
    for pp_rank, (vit_num, llm_num) in enumerate(pp_split):
        if vit_num > 0 and pp_rank > vit_range[1]:
            vit_range[1] = pp_rank
        if llm_num > 0 and pp_rank < llm_range[0]:
            llm_range[0] = pp_rank
    vit_start_idx = 0
    llm_start_idx = 0
    return_dicts = []
    copy_dict = deepcopy(state_dict)
    for pp_rank, (vit_num, llm_num) in enumerate(pp_split):
        vit_end_idx = vit_start_idx + vit_num
        llm_end_idx = llm_start_idx + llm_num
        new_dict = {}
        for key, value in state_dict.items():
            if key.startswith('image_encoder.encoder.patch_embed.'):
                if pp_rank == vit_range[0]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('image_encoder.encoder.blocks.layers.'):
                layer_idx = int(key.split('.')[4])
                if vit_start_idx <= layer_idx < vit_end_idx and vit_range[0] <= pp_rank <= vit_range[1]:
                    new_idx = layer_idx - vit_start_idx
                    new_key = key.replace(f'{layer_idx}', f'{new_idx}', 1)
                    new_dict[new_key] = value
                    copy_dict.pop(key)
            elif key.startswith('image_encoder.projector.'):
                if pp_rank == vit_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.embedding.'):
                if pp_rank == llm_range[0]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.decoder.layers.'):
                layer_idx = int(key.split('.')[3])
                if llm_start_idx <= layer_idx < llm_end_idx and llm_range[0] <= pp_rank <= llm_range[1]:
                    new_idx = layer_idx - llm_start_idx
                    new_key = key.replace(f'{layer_idx}', f'{new_idx}', 1)
                    new_dict[new_key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.decoder.final_layernorm.'):
                if pp_rank == llm_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.output_layer.'):
                if pp_rank == llm_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
        vit_start_idx = vit_end_idx
        llm_start_idx = llm_end_idx
        return_dicts.append(new_dict)
    return return_dicts


def save_by_pp(state_dicts: list[dict[str, torch.Tensor]],
               save_root_dir: Path,
               iteration: str | int = 'release',
               tp_rank: int = 0):
    for pp_rank, state_dict in enumerate(tqdm(state_dicts, desc="pp step")):
        name_parts = ["mp", "rank", f"{tp_rank:02d}"]
        if len(state_dicts) > 1:
            name_parts.append(f"{pp_rank:03d}")
        iter_name = iteration if isinstance(iteration, str) else f"iter_{iteration:07d}"
        save_path = save_root_dir.joinpath(iter_name, "_".join(name_parts))
        save_path.mkdir(exist_ok=True, parents=True)
        torch.save({'model': state_dict}, save_path.joinpath('model_optim_rng.pt'))
    save_root_dir.joinpath(LATEST_TXT).write_text(str(iteration))


def split_by_tp(_state_dict: dict[str, torch.Tensor], _tp_num: int = 1) -> list[dict[str, torch.Tensor]]:
    if _tp_num == 1:
        return [_state_dict]
    return_dicts = []
    copy_dict = deepcopy(_state_dict)
    for tp_rank in range(_tp_num):
        new_state_dict = {}
        for key, value in _state_dict.items():
            if key.startswith('text_decoder.decoder.layer') and 'linear_fc1.weight' in key:
                value_shape = value.shape
                size_per_tp = value_shape[0] // _tp_num // 2
                values = torch.chunk(value, 2, dim=0)
                gate_tp = values[0][tp_rank * size_per_tp:(tp_rank + 1) * size_per_tp, :]
                up_tp = values[1][tp_rank * size_per_tp:(tp_rank + 1) * size_per_tp, :]
                new_state_dict[key] = torch.cat((gate_tp, up_tp), dim=0)
            elif 'linear_qkv.weight' in key or 'linear_fc1.weight' in key:
                value_shape = value.shape
                size_per_tp = value_shape[0] // _tp_num
                new_state_dict[key] = value[tp_rank * size_per_tp:(tp_rank + 1) * size_per_tp, :]
            elif 'linear_qkv.bias' in key or 'linear_fc1.bias' in key:
                value_shape = value.shape
                size_per_tp = value_shape[0] // _tp_num
                new_state_dict[key] = value[tp_rank * size_per_tp:(tp_rank + 1) * size_per_tp]
            elif 'linear_proj.weight' in key or 'linear_fc2.weight' in key:
                value_shape = value.shape
                size_per_tp = value_shape[1] // _tp_num
                new_state_dict[key] = value[:, tp_rank * size_per_tp:(tp_rank + 1) * size_per_tp]
            elif 'output_layer' in key or 'word_embeddings' in key:
                value_shape = value.shape
                size_per_tp = value_shape[0] // _tp_num
                new_state_dict[key] = value[tp_rank * size_per_tp:(tp_rank + 1) * size_per_tp, :]
            else:
                new_state_dict[key] = value
        return_dicts.append(new_state_dict)
    return return_dicts


def main(convert_config: ConvertMMConfig):
    # qwen2vl获取到的config类型是Qwen2VLConfig
    config = cast(Qwen2VLConfig, convert_config.hf_config.config)
    parallel_config = convert_config.parallel_config
    # 加载权重字典
    state_dict = load_from_hf(convert_config.hf_config.hf_dir)
    # hf转换成mm格式，包含重命名、qkv合并、mlp合并等操作
    state_dict = convert_hf_to_mm(state_dict, config.num_hidden_layers, config.vision_config.embed_dim,
                                  config.vision_config.num_heads, config.num_key_value_heads)
    # 权重字典按tp域切分
    tp_state_dicts = split_by_tp(state_dict, parallel_config.tp_size)
    # pp索引生成
    pp_split = merge_pp_index(parallel_config.vit_pp_layers, parallel_config.llm_pp_layers)
    # 处理每个tp域
    for tp_rank, tp_state_dict in enumerate(tqdm(tp_state_dicts, desc="tp step")):
        # 每个tp域对应的pp域拆分
        pp_state_dicts = split_model_by_pipeline(tp_state_dict, pp_split)
        save_by_pp(pp_state_dicts, convert_config.mm_dir, tp_rank=tp_rank)
