#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : qwen_vl_hf_to_mm.py
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

mindspeed-mm模型目录(这里是tp1/pp4训练保存的模型)：
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
from typing import Callable, Any, cast

from tqdm import tqdm

from checkpoint.operator import create_qwen2vl_ops, create_qwen2_5_vl_ops, create_qwen2_5_omni_ops, create_qwen3_vl_ops, \
    qwen2vl_tp_patterns, qwen2_5_vl_tp_patterns, qwen3_vl_tp_patterns, Operator
from checkpoint.utils import ConvertVppMMConfig, load_from_hf, merge_vpp_index, split_by_ep, split_model_by_pipeline, \
    save_by_vpp, split_by_tp, convert_hf_to_mm, merge_llm_weights_to_state_dict


def convert(convert_config: ConvertVppMMConfig, config: Any, ops: list[Operator],
            tp_patterns: dict[str, Callable]):
    parallel_config = convert_config.parallel_config
    llm_config = convert_config.llm_hf_config
    num_experts = getattr(config, 'num_experts', 0)
    # 校验tp切分数
    num_key_value_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else config.thinker_config.text_config.num_key_value_heads
    if num_key_value_heads % parallel_config.tp_size != 0:
        raise ValueError(
            f"Number of key-value heads ({num_key_value_heads}) must be divisible by TP size ({parallel_config.tp_size})"
        )
    # 加载权重字典
    state_dict = load_from_hf(convert_config.hf_config.hf_dir)

    if llm_config is not None:
        llm_state_dict = load_from_hf(llm_config.hf_dir)
        state_dict = merge_llm_weights_to_state_dict(state_dict, llm_state_dict)
        num_experts = getattr(llm_config.config, 'num_experts', 0)
    
    # 权重转换、合并
    state_dict = convert_hf_to_mm(state_dict, ops, config.tie_word_embeddings, parallel_config.is_pp())
   
    # 权重字典按ep域切分
    ep_state_dicts = split_by_ep(state_dict, parallel_config.ep_size, _num_experts=num_experts)
    
    # 权重字典按tp域切分
    ep_tp_state_dicts = []
    for ep_state_dict in ep_state_dicts:
        # 每个ep域对应的tp域拆分
        tp_state_dicts = split_by_tp(ep_state_dict, tp_patterns, parallel_config.tp_size)
        ep_tp_state_dicts.append(tp_state_dicts)

    # pp索引生成
    pp_split = merge_vpp_index(parallel_config.vit_pp_layers, parallel_config.llm_pp_layers, parallel_config.audio_pp_layers)
    for ep_rank, tp_state_dicts in enumerate(tqdm(ep_tp_state_dicts, desc="ep step")):
        for tp_rank, tp_state_dict in enumerate(tqdm(tp_state_dicts, desc="tp step")):
            pp_state_dicts = split_model_by_pipeline(tp_state_dict, pp_split)
            save_by_vpp(pp_state_dicts, convert_config.mm_dir, pp_and_vpp_size=(parallel_config.pp_size, parallel_config.vpp_size), ep_size=parallel_config.ep_size, ep_rank=ep_rank, tp_rank=tp_rank)


def convert_qwen2vl(convert_config: ConvertVppMMConfig):
    from transformers.models.qwen2_vl import Qwen2VLConfig
    config = convert_config.hf_config.config
    config = cast(Qwen2VLConfig, config)
    ops = create_qwen2vl_ops(config.vision_config.embed_dim,
                             config.vision_config.num_heads,
                             config.num_key_value_heads)
    convert(convert_config, config, ops, qwen2vl_tp_patterns)


def convert_qwen2_5_vl(convert_config: ConvertVppMMConfig):
    from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig
    config = convert_config.hf_config.config
    config = cast(Qwen2_5_VLConfig, config)
    # qwen2.5vl和qwen2vl的差异主要在权重转换的算子以及tp转换时的模式
    ops = create_qwen2_5_vl_ops(config.vision_config.hidden_size,
                                config.vision_config.num_heads,
                                config.num_key_value_heads)
    convert(convert_config, config, ops, qwen2_5_vl_tp_patterns)


def convert_qwen3_vl(convert_config: ConvertVppMMConfig):
    from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig
    config = convert_config.hf_config.config
    llm_config = convert_config.llm_hf_config
    config = cast(Qwen2_5_VLConfig, config)

    num_key_value_heads = llm_config.config.num_key_value_heads if llm_config is not None else config.num_key_value_heads
    ops = create_qwen3_vl_ops(config.vision_config.hidden_size,
                              config.vision_config.num_heads,
                              num_key_value_heads)
    convert(convert_config, config, ops, qwen3_vl_tp_patterns)


def convert_qwen2_5_omni(convert_config: ConvertVppMMConfig):
    from transformers.models.qwen2_5_omni import Qwen2_5OmniConfig
    config = convert_config.hf_config.config
    config = cast(Qwen2_5OmniConfig, config)
    ops = create_qwen2_5_omni_ops(config.thinker_config.vision_config.num_heads,
                                config.thinker_config.text_config.num_key_value_heads,
                                config.thinker_config.audio_config.encoder_attention_heads,
                                config.thinker_config.audio_config.d_model,
                                config.thinker_config.audio_config.encoder_layers
                                )
    convert(convert_config, config, ops, qwen2_5_vl_tp_patterns)
