#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : utils.py
@Time    : 2025/01/14
@Desc    : 模型相关的配置定义
"""
import os
import json
import shutil
from functools import cached_property
from pathlib import Path

import torch
from safetensors.torch import save_file
from pydantic import BaseModel, DirectoryPath, PositiveInt, NonNegativeInt, model_validator, computed_field
from transformers import PretrainedConfig, AutoConfig

LATEST_TXT = "latest_checkpointed_iteration.txt"


class ParallelConfig(BaseModel):
    """权模型切分配置，包括tp的size，以及pp切分时vit和llm在pp域每张卡上切分的层数"""

    llm_pp_layers: list[NonNegativeInt]
    """llm模块pipeline parallel切分每张卡上切分几层"""

    vit_pp_layers: list[NonNegativeInt]
    """vit模块pipeline parallel切分每张卡上切分几层"""

    tp_size: PositiveInt = 1
    """tensor parallel张量并行组，模型转换时不同的tp组要切分到不同的目录下"""

    @computed_field
    def pp_size(self) -> PositiveInt:
        return len(self.llm_pp_layers)

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
        vit_pipeline_num_layers_flat = [item for sublist in vit_pipeline_num_layers for item in sublist]
        llm_pipeline_num_layers_flat = [item for sublist in llm_pipeline_num_layers for item in sublist]

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


def split_by_index_json(state_dict: dict[str, torch.Tensor], hf_dir: Path) -> list[dict[str, torch.Tensor]]:
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
