#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : utils.py
@Time    : 2025/01/14
@Desc    :
"""
from functools import cached_property
from pathlib import Path

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


class HfConfig(BaseModel):
    """huggingface下载的开源权重的配置，主要包括路径校验及AutoConfig"""
    hf_dir: DirectoryPath
    """huggingface下载的路径"""

    @cached_property
    def config(self) -> PretrainedConfig:
        return AutoConfig.from_pretrained(self.hf_dir)

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
