#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : deepseekvl_hf_to_mm.py
@Time    : 2025/04/17
@Desc    : deepseekvl huggingface模型转换成mindspeed-mm模型

"""
from copy import deepcopy
from typing import cast

import torch
from tqdm import tqdm

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from checkpoint.utils import ConvertMMConfig, load_from_hf, merge_pp_index, split_model_by_pipeline, \
    save_by_pp


def load_from_hf(load_dir, trust_remote_code):
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(
        load_dir,
        device_map="cpu",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code
    )

    return hf_model.state_dict()


def convert_hf_to_mm(_state_dict: dict[str, torch.Tensor],
                     _num_layers: int,
                     _num_experts: int) -> dict[str, torch.Tensor]:
    new_params = {}
    for key, value in tqdm(_state_dict.items(), desc="convert weights"):
        new_key = None
        # visual 权重转换部分
        if key.startswith("vision"):
            new_key = key.replace("vision", "image_encoder.encoder")
        # projector 部分
        elif key.startswith("projector"):
            new_key = key.replace("projector", "image_encoder.projector")
        
        elif key.startswith("language"):
            new_key = key.replace('language', 'text_decoder')
            new_key = new_key.replace('model.embed_tokens', 'embedding.word_embeddings')
            new_key = new_key.replace('model.layers', 'decoder.layers')

            new_key = new_key.replace('self_attn.q_proj', 'self_attention.q_proj')
            new_key = new_key.replace('self_attn.kv_a_proj_with_mqa', 'self_attention.kv_a_proj_with_mqa')
            new_key = new_key.replace('self_attn.kv_a_layernorm', 'self_attention.k_layernorm')
            new_key = new_key.replace('self_attn.kv_b_proj', 'self_attention.linear_kvb')
            new_key = new_key.replace('self_attn.o_proj', 'self_attention.linear_proj')

            new_key = new_key.replace('gate_proj', 'linear_fc1_gate')
            new_key = new_key.replace('up_proj', 'linear_fc1_up')
            new_key = new_key.replace('down_proj', 'linear_fc2')
            new_key = new_key.replace('post_attention_layernorm', 'pre_mlp_layernorm')

            new_key = new_key.replace('mlp.experts', 'mlp.experts.local_experts')
            new_key = new_key.replace('mlp.gate', 'mlp.router')

            new_key = new_key.replace('model.norm', 'decoder.final_layernorm')
            new_key = new_key.replace('lm_head', 'output_layer')
        else:
            new_key = key

        if new_key is not None:
            print(f"mapping {key} to {new_key}")
            new_params[new_key] = value
    
    # 合并gate up 权重
    for i in range(_num_layers):
        # 合并self attn的mlp
        gate_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1_gate.weight'
        up_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1_up.weight'
        fc1_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1.weight'

        # 如果权重名字在新字典中，则获取对应权重值
        if gate_name in new_params.keys() and up_name in new_params.keys():
            gate_proj_weight, up_proj_weight = new_params[gate_name], new_params[up_name]
            new_params.pop(gate_name)
            new_params.pop(up_name)

            # 将 gate 和 up 沿着第0维度进行拼接
            linear_fc1 = torch.cat([gate_proj_weight, up_proj_weight], dim=0)
            new_params[fc1_name] = linear_fc1

        # 合并moe的shared experts的mlp
        gate_name = f'text_decoder.decoder.layers.{i}.mlp.shared_experts.linear_fc1_gate.weight'
        up_name = f'text_decoder.decoder.layers.{i}.mlp.shared_experts.linear_fc1_up.weight'
        fc1_name = f'text_decoder.decoder.layers.{i}.mlp.shared_experts.linear_fc1.weight'

        # 如果权重名字在新字典中，则获取对应权重值
        if gate_name in new_params.keys() and up_name in new_params.keys():
            gate_proj_weight, up_proj_weight = new_params[gate_name], new_params[up_name]
            new_params.pop(gate_name)
            new_params.pop(up_name)

            # 将 gate 和 up 沿着第0维度进行拼接
            linear_fc1 = torch.cat([gate_proj_weight, up_proj_weight], dim=0)
            new_params[fc1_name] = linear_fc1

            # 合并moe的experts的mlp
            for j in range(_num_experts):
                gate_name = f'text_decoder.decoder.layers.{i}.mlp.experts.local_experts.{j}.linear_fc1_gate.weight'
                up_name = f'text_decoder.decoder.layers.{i}.mlp.experts.local_experts.{j}.linear_fc1_up.weight'
                fc1_name = f'text_decoder.decoder.layers.{i}.mlp.experts.local_experts.{j}.linear_fc1.weight'

                gate_proj_weight, up_proj_weight = new_params[gate_name], new_params[up_name]
                new_params.pop(gate_name)
                new_params.pop(up_name)

                linear_fc1 = torch.cat([gate_proj_weight, up_proj_weight], dim=0)
                new_params[fc1_name] = linear_fc1

    # 合并 kv_a_proj_with_mqa + q_proj = linear_qkv
    for i in range(_num_layers):
        q_proj_name = f'text_decoder.decoder.layers.{i}.self_attention.q_proj.weight'
        kv_a_proj_name = f'text_decoder.decoder.layers.{i}.self_attention.kv_a_proj_with_mqa.weight'
        linear_qkv_name = f'text_decoder.decoder.layers.{i}.mlp.shared_experts.linear_qkv.weight'

        if q_proj_name in new_params.keys() and kv_a_proj_name in new_params.keys():
            q_proj_weight = new_params[q_proj_name]
            kv_a_proj_weight = new_params[kv_a_proj_name]
            new_params.pop(q_proj_name)
            new_params.pop(kv_a_proj_name)

            # 将q_proj 和 kv_a_proj 沿着0维concat
            linear_qkv = torch.cat([q_proj_weight, kv_a_proj_weight], dim=0)
            new_params[linear_qkv_name] = linear_qkv

    return new_params


def split_by_tp(_state_dict: dict[str, torch.Tensor], _tp_num: int = 1) -> list[dict[str, torch.Tensor]]:
    if _tp_num == 1:
        return [_state_dict]
    else:
        raise AssertionError("Don't support TP size > 1 now!")


def main(convert_config: ConvertMMConfig):
    config = convert_config.hf_config.config
    load_dir = convert_config.hf_config.hf_dir
    parallel_config = convert_config.parallel_config
    trust_remote_code = convert_config.trust_remote_code

    # 加载原始权重字典
    state_dict = load_from_hf(load_dir, trust_remote_code=trust_remote_code)

    # hf权重映射到mm
    state_dict = convert_hf_to_mm(state_dict, config.language_config.num_hidden_layers, config.language_config.n_routed_experts)

    # 权重字典按tp域切分
    tp_state_dicts = split_by_tp(state_dict, parallel_config.tp_size)

    # pp索引生成
    pp_split = merge_pp_index(parallel_config.vit_pp_layers, parallel_config.llm_pp_layers)

    # 处理每个tp域
    for tp_rank, tp_state_dict in enumerate(tqdm(tp_state_dicts, desc="tp step")):
        # 每个tp域对应的pp域拆分
        pp_state_dicts = split_model_by_pipeline(tp_state_dict, pp_split)
        save_by_pp(pp_state_dicts, convert_config.mm_dir, tp_rank=tp_rank)
