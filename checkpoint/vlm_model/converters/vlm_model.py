import os

from checkpoint.common.constant import SAFE_MODE
from checkpoint.common.converter import Converter
from checkpoint.vlm_model.config import ConvertResplitConfig, ConvertVppMMConfig, ConvertVppHFConfig
from checkpoint.vlm_model.converters import internvl2_hf_to_mm


class InternVLConverter(Converter):
    """InternVL模型转换工具"""

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfig):
        """huggingface模型转换mindspeed-mm模型权重"""
        internvl2_hf_to_mm.main(cfg)
        # 安全管控权限
        os.chmod(cfg.mm_dir, SAFE_MODE)

    @staticmethod
    def mm_to_hf(cfg: ConvertVppHFConfig):
        from checkpoint.vlm_model.converters import internvl2_mm_to_hf
        """mindspeed-mm模型转换huggingface模型权重"""
        internvl2_mm_to_hf.main(cfg)
        # 安全管控权限
        os.chmod(cfg.save_hf_dir, SAFE_MODE)

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        pass
