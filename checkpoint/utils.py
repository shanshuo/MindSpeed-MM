#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : utils.py
@Time    : 2025/01/14
@Desc    : 模型相关的配置定义
"""
import os
import json
import re
import shutil
from copy import deepcopy
from functools import cached_property
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file
from pydantic import BaseModel, DirectoryPath, PositiveInt, NonNegativeInt, model_validator, computed_field
from tqdm import tqdm
from transformers import PretrainedConfig, AutoConfig

from checkpoint.operator import Operator, TieOp, STATE_DICT_T, TP_PATTERN_T

LATEST_TXT = "latest_checkpointed_iteration.txt"


class ParallelConfig(BaseModel):
    """权模型切分配置，包括tp的size，以及pp切分时vit和llm在pp域每张卡上切分的层数"""

    llm_pp_layers: list[NonNegativeInt]
    """llm模块pipeline parallel切分每张卡上切分几层"""

    vit_pp_layers: list[NonNegativeInt]
    """vit模块pipeline parallel切分每张卡上切分几层"""

    tp_size: PositiveInt = 1
    """tensor parallel张量并行组，模型转换时不同的tp组要切分到不同的目录下"""

    ep_size: PositiveInt = 1
    """expert parallel张量并行组，模型转换时不同的ep组要切分到不同的目录下"""

    @computed_field
    def pp_size(self) -> PositiveInt:
        return len(self.llm_pp_layers)

    def is_pp(self) -> bool:
        return self.pp_size > 1

    @model_validator(mode='after')
    def validate_pp_layers(self) -> "ParallelConfig":
        if len(self.vit_pp_layers) != len(self.llm_pp_layers):
            raise ValueError("vit和llm的pp_layers配置长度一定要一致")
        if len(self.vit_pp_layers) < 1:
            raise ValueError("pp layers长度至少为1")
        return self


class VppParallelConfig(BaseModel):
    """权模型切分配置，包括tp的size，以及pp切分时vit和llm在pp域每张卡上切分的层数"""

    llm_pp_layers: list[list[NonNegativeInt]]
    """llm模块pipeline parallel切分每张卡上切分几层, vpp切分配置参考docs/features/virtual_pipeline_parallel.md"""

    vit_pp_layers: list[list[NonNegativeInt]]
    """vit模块pipeline parallel切分每张卡上切分几层, vpp切分配置参考docs/features/virtual_pipeline_parallel.md"""

    tp_size: PositiveInt = 1
    """tensor parallel张量并行组，模型转换时不同的tp组要切分到不同的目录下"""

    @computed_field
    def pp_size(self) -> PositiveInt:
        return len(self.llm_pp_layers[0])

    @computed_field
    def vpp_size(self) -> PositiveInt:
        return len(self.llm_pp_layers)

    @model_validator(mode='after')
    def validate_pp_layers(self) -> "VppParallelConfig":
        if len(self.vit_pp_layers) != len(self.llm_pp_layers):
            raise ValueError("vit和llm的pp_layers配置长度一定要一致")
        if len(self.vit_pp_layers) < 1:
            raise ValueError("pp layers长度至少为1")
        return self

    @model_validator(mode='after')
    def validate_vpp_layers(self) -> "VppParallelConfig":
        pp_size = self.pp_size
        for vpp in self.llm_pp_layers:
            if len(vpp) != pp_size:
                raise ValueError("vit和llm的每个vpp配置长度一定要一致")
        for vpp in self.vit_pp_layers:
            if len(vpp) != pp_size:
                raise ValueError("vit和llm的每个vpp配置长度一定要一致")
        return self


class HfConfig(BaseModel):
    """huggingface下载的开源权重的配置，主要包括路径校验及AutoConfig"""
    hf_dir: DirectoryPath
    """huggingface下载的路径"""

    @cached_property
    def config(self) -> PretrainedConfig:
        return AutoConfig.from_pretrained(self.hf_dir, local_files_only=True)

    @model_validator(mode='after')
    def validate_hf_dir(self) -> "HfConfig":
        safetensors_files = list(self.hf_dir.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No *.safetensors files found in {self.hf_dir}")
        if not list(self.hf_dir.glob("config.json")):
            raise FileNotFoundError(f"No config.json in {self.hf_dir}")
        return self


# BaseModel/dataclasses注意要在field的下一行添加描述说明
class ConvertMMConfig(BaseModel):
    """huggingface权重转换为mindspeed-mm权重配置"""

    mm_dir: Path
    """mm保存的路径"""

    parallel_config: ParallelConfig
    """并行配置"""

    hf_config: HfConfig
    """hf下载的原始权重路径配置"""

    trust_remote_code: bool = False
    """trust_remote_code 默认设为False, 需要用户手动设为True"""


class ConvertHFConfig(ConvertMMConfig):
    """mindspeed-mm训练出来的权重转换为huggingface格式权重的配置"""
    save_hf_dir: Path
    """mm转回hf格式时保存的路径"""


class ConvertResplitConfig(BaseModel):
    """mindspeed-mm训练出来的权重pp重切分的配置，pp重切分一般用在推理时等设备变化的场景"""
    source_dir: DirectoryPath
    """原始训练出来的权重路径"""

    target_dir: Path
    """重切分后保存的权重路径"""

    source_parallel_config: ParallelConfig
    """原始训练出权重的并行配置"""

    target_parallel_config: ParallelConfig
    """重切分后的权重的并行配置"""

    @model_validator(mode='after')
    def validate_hf_dir(self) -> "ConvertResplitConfig":
        if sum(self.source_parallel_config.vit_pp_layers) != sum(self.target_parallel_config.vit_pp_layers):
            raise ValueError("vit pp layers not equal!")
        if sum(self.source_parallel_config.llm_pp_layers) != sum(self.target_parallel_config.llm_pp_layers):
            raise ValueError("llm pp layers not equal!")
        return self


# BaseModel/dataclasses注意要在field的下一行添加描述说明
class ConvertVppMMConfig(BaseModel):
    """huggingface权重转换为mindspeed-mm权重配置"""

    mm_dir: Path
    """mm保存的路径"""

    parallel_config: VppParallelConfig
    """并行配置"""

    hf_config: HfConfig
    """hf下载的原始权重路径配置"""

    trust_remote_code: bool = False
    """trust_remote_code 默认设为False, 需要用户手动设为True"""

    @model_validator(mode='after')
    def validate_sum_of_layers(self) -> "ConvertVppMMConfig":
        model_config = self.hf_config.config
        vit_pipeline_num_layers = self.parallel_config.vit_pp_layers
        llm_pipeline_num_layers = self.parallel_config.llm_pp_layers
        vit_num_layers = model_config.vision_config.num_hidden_layers
        llm_num_layers = model_config.llm_config.num_hidden_layers

        # Flatten the vit and llm layers for VPP
        vit_pipeline_num_layers_flat = [
            item
            for sublist in vit_pipeline_num_layers
            for item in sublist
        ]
        llm_pipeline_num_layers_flat = [
            item
            for sublist in llm_pipeline_num_layers
            for item in sublist
        ]

        # Validation for flattened lists
        expected_length = self.parallel_config.pp_size * self.parallel_config.vpp_size
        if len(vit_pipeline_num_layers_flat) != expected_length:
            raise AssertionError(f'Length of vit_pipeline_num_layers_flat must be equal to pp_size * vp_size, '
                                 f'but got {len(vit_pipeline_num_layers_flat)} and {expected_length}.')
        if sum(vit_pipeline_num_layers_flat) != vit_num_layers:
            raise AssertionError(f'Sum of vit_pipeline_num_layers_flat must be equal to vit_num_layers, '
                                 f'but got {sum(vit_pipeline_num_layers_flat)} and {vit_num_layers}.')
        if len(llm_pipeline_num_layers_flat) != expected_length:
            raise AssertionError(f'Length of llm_pipeline_num_layers_flat must be equal to pp_size * vp_size, '
                                 f'but got {len(llm_pipeline_num_layers_flat)} and {expected_length}.')
        if sum(llm_pipeline_num_layers_flat) != llm_num_layers:
            raise AssertionError(f'Sum of llm_pipeline_num_layers_flat must be equal to llm_num_layers, '
                                 f'but got {sum(llm_pipeline_num_layers_flat)} and {llm_num_layers}.')
        return self


class ConvertVppHFConfig(ConvertVppMMConfig):
    """mindspeed-mm训练出来的权重转换为huggingface格式权重的配置"""
    save_hf_dir: Path
    """mm转回hf格式时保存的路径"""


def save_by_index_json(_state_dicts, _save_dir):
    metadata = {
        'format': 'pt'
    }
    for index, state_dict in enumerate(_state_dicts, start=1):
        name = f'model-{index:05}-of-{len(_state_dicts):05}.safetensors'
        save_file(state_dict, Path(_save_dir).joinpath(name), metadata=metadata)


def split_by_index_json(state_dict: STATE_DICT_T, hf_dir: Path) -> list[STATE_DICT_T]:
    index_json_path = hf_dir.joinpath('model.safetensors.index.json')
    if not os.path.exists(index_json_path):
        raise ValueError(f"safetensors.index.json not in {index_json_path}")
    return_dicts = []
    weight_map = json.loads(index_json_path.read_text()).get('weight_map', {})
    for key, value in weight_map.items():
        index = int(value.split('-')[1])
        while index > len(return_dicts):
            return_dicts.append({})
        return_dicts[index - 1][key] = state_dict[key]
    return return_dicts


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


def load_from_hf(hf_dir: Path) -> STATE_DICT_T:
    # 注意AutoModel.from_pretrained转换成模型对象时，存在torch_dtype问题需确认，因此这里直接读取safetensors确保dtype一致
    files = list(hf_dir.glob("*.safetensors"))
    state_dict = {}
    for safe_path in files:
        state_dict.update(load_file(str(safe_path), device='cpu'))
    return state_dict


def merge_pp_index(vit_pipeline_num_layers: list[int], llm_pipeline_num_layers: list[int]) -> list[tuple[int, int]]:
    """返回每张卡上vit和llm各自的层数"""
    split_method = []
    for vit_num, llm_num in zip(vit_pipeline_num_layers, llm_pipeline_num_layers):
        split_method.append((vit_num, llm_num))
    return split_method


def split_model_by_pipeline(state_dict: STATE_DICT_T, pp_split: list[tuple[int, int]]) -> list[STATE_DICT_T]:
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


def split_by_tp(state_dict: STATE_DICT_T, patterns: TP_PATTERN_T, tp_size: int = 1) -> list[STATE_DICT_T]:
    if tp_size == 1:
        return [state_dict]
    return_dicts = []
    for tp_rank in range(tp_size):
        new_state_dict = {}
        for key, value in state_dict.items():
            for pattern, tp_split_func in patterns.items():
                if re.match(pattern, key):
                    value = tp_split_func(tp_size, tp_rank, value)
                    break
            new_state_dict[key] = value
        return_dicts.append(new_state_dict)
    return return_dicts


def convert_hf_to_mm(state_dict: STATE_DICT_T, ops: list[Operator], is_tie: bool, is_pp: bool) -> STATE_DICT_T:
    if is_tie and is_pp:
        # pp1时，output_layer从word_embedding处获取共享权重。pp>1时，流水线后面的卡无法获得word_embedding，因此需要加上该权重
        ops.append(TieOp(raw_name='text_decoder.embedding.word_embeddings.weight',
                         new_name='text_decoder.output_layer.weight'))
    for op in ops:
        op.handle(state_dict)
    return state_dict
