import os
from typing import cast, Any

from checkpoint.common.converter import Converter
from checkpoint.vlm_model import qwen_vl_hf_to_mm, internvl2_hf_to_mm, deepseekvl_hf_to_mm
from checkpoint.vlm_model import qwen_vl_mm_to_hf
from checkpoint.vlm_model.operator import create_qwen2vl_ops, qwen2vl_tp_patterns, create_qwen2_5_vl_ops, \
    qwen2_5_vl_tp_patterns, create_qwen2_5_omni_ops, qwen2_5_omni_tp_patterns, create_qwen3_vl_ops, qwen3_vl_tp_patterns
from checkpoint.vlm_model.qwen_vl_hf_to_mm import vision_schema, text_schema, audio_schema
from checkpoint.vlm_model.utils import ConvertHFConfig, ConvertResplitConfig, ConvertMMConfig, ConvertVppMMConfig, \
    ConvertVppHFConfig

SAFE_MODE = 0o750


class Qwen2VLConverter(Converter):
    """Qwen2VL模型转换工具"""

    @staticmethod
    # 创建转换操作,加下划线之后命令行会自动忽略这条子命令
    def _create_ops(config: Any):
        from transformers.models.qwen2_vl import Qwen2VLConfig
        config = cast(Qwen2VLConfig, config)
        llm_head_hidden_size = config.hidden_size // config.num_attention_heads
        llm_q_size = llm_head_hidden_size * config.num_attention_heads // config.num_key_value_heads
        llm_kv_size = llm_head_hidden_size
        ops = create_qwen2vl_ops(config.vision_config.embed_dim,
                                 config.vision_config.num_heads,
                                 config.num_key_value_heads,
                                 llm_q_size,
                                 llm_kv_size
                                 )
        return ops, config

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfig):
        """huggingface模型转换mindspeed-mm模型权重"""
        ops, config = Qwen2VLConverter._create_ops(cfg.hf_config.config)
        qwen_vl_hf_to_mm.convert(cfg, config, ops, qwen2vl_tp_patterns, [vision_schema, text_schema])
        # 安全管控权限
        os.chmod(cfg.mm_dir, SAFE_MODE)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        """mindspeed-mm模型转换huggingface模型权重"""
        ops, config = Qwen2VLConverter._create_ops(cfg.hf_config.config)
        qwen_vl_mm_to_hf.convert(cfg, config, ops, qwen2vl_tp_patterns)
        # 安全管控权限
        os.chmod(cfg.save_hf_dir, SAFE_MODE)

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        # qwen2vl_resplit会用到mindspeed.megatron_adaptor模块，提前import会引入问题，故在此import
        from checkpoint.vlm_model import qwen2vl_resplit
        qwen2vl_resplit.main(cfg)
        # 安全管控权限
        os.chmod(cfg.target_dir, SAFE_MODE)


class Qwen2_5_VLConverter(Converter):
    """Qwen2.5VL模型转换工具"""

    @staticmethod
    # 创建转换操作,加下划线之后命令行会自动忽略这条子命令
    def _create_ops(config: Any):
        from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig
        config = cast(Qwen2_5_VLConfig, config)
        # qwen2.5vl和qwen2vl的差异主要在权重转换的算子以及tp转换时的模式
        llm_head_hidden_size = config.hidden_size // config.num_attention_heads
        llm_q_size = llm_head_hidden_size * config.num_attention_heads // config.num_key_value_heads
        llm_kv_size = llm_head_hidden_size
        ops = create_qwen2_5_vl_ops(config.vision_config.hidden_size,
                                    config.vision_config.num_heads,
                                    config.num_key_value_heads,
                                    llm_q_size,
                                    llm_kv_size
                                    )
        return ops, config

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfig):
        """huggingface模型转换mindspeed-mm模型权重"""
        ops, config = Qwen2_5_VLConverter._create_ops(cfg.hf_config.config)
        qwen_vl_hf_to_mm.convert(cfg, config, ops, qwen2_5_vl_tp_patterns, [vision_schema, text_schema])
        # 安全管控权限
        os.chmod(cfg.mm_dir, SAFE_MODE)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        """mindspeed-mm模型转换huggingface模型权重"""
        ops, config = Qwen2_5_VLConverter._create_ops(cfg.hf_config.config)
        qwen_vl_mm_to_hf.convert(cfg, config, ops, qwen2_5_vl_tp_patterns)
        # 安全管控权限
        os.chmod(cfg.save_hf_dir, SAFE_MODE)

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        pass


class Qwen2_5_OmniConverter(Converter):
    """Qwen2.5Omni模型转换工具"""

    @staticmethod
    # 创建转换操作,加下划线之后命令行会自动忽略这条子命令
    def _create_ops(config: Any):
        from transformers.models.qwen2_5_omni import Qwen2_5OmniConfig
        config = cast(Qwen2_5OmniConfig, config)
        vit_head_hidden_size = config.thinker_config.vision_config.hidden_size // config.thinker_config.vision_config.num_heads
        audio_head_hidden_size = config.thinker_config.audio_config.d_model // config.thinker_config.audio_config.encoder_attention_heads
        llm_head_hidden_size = config.thinker_config.text_config.hidden_size // config.thinker_config.text_config.num_attention_heads
        llm_q_size = llm_head_hidden_size * config.thinker_config.text_config.num_attention_heads // config.thinker_config.text_config.num_key_value_heads
        llm_kv_size = llm_head_hidden_size
        ops = create_qwen2_5_omni_ops(config.thinker_config.vision_config.num_heads,
                                      config.thinker_config.text_config.num_key_value_heads,
                                      config.thinker_config.audio_config.encoder_attention_heads,
                                      (vit_head_hidden_size,
                                       audio_head_hidden_size,
                                       llm_q_size,
                                       llm_kv_size)
                                      )
        config.tie_word_embeddings = config.thinker_config.text_config.tie_word_embeddings
        return ops, config

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfig):
        """huggingface模型转换mindspeed-mm模型权重"""
        ops, config = Qwen2_5_OmniConverter._create_ops(cfg.hf_config.config)
        qwen_vl_hf_to_mm.convert(cfg, config, ops, qwen2_5_omni_tp_patterns, [vision_schema, text_schema, audio_schema])
        # 安全管控权限
        os.chmod(cfg.mm_dir, SAFE_MODE)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        """mindspeed-mm模型转换huggingface模型权重"""
        ops, config = Qwen2_5_OmniConverter._create_ops(cfg.hf_config.config)
        qwen_vl_mm_to_hf.convert(cfg, config, ops, qwen2_5_omni_tp_patterns)
        # 安全管控权限
        os.chmod(cfg.save_hf_dir, SAFE_MODE)

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        pass


class Qwen3_VLConverter(Converter):
    """Qwen3VL模型转换工具"""

    @staticmethod
    # 创建转换操作,加下划线之后命令行会自动忽略这条子命令
    def _create_ops(config: Any):
        from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig
        llm_config = cfg.llm_hf_config
        config = cast(Qwen2_5_VLConfig, config)
        num_key_value_heads = llm_config.config.num_key_value_heads if llm_config is not None else config.num_key_value_heads
        llm_head_hidden_size = config.hidden_size // config.num_attention_heads
        llm_q_size = llm_head_hidden_size * config.num_attention_heads // config.num_key_value_heads
        llm_kv_size = llm_head_hidden_size
        ops = create_qwen3_vl_ops(config.vision_config.hidden_size,
                                  config.vision_config.num_heads,
                                  num_key_value_heads,
                                  llm_q_size,
                                  llm_kv_size
                                  )
        return ops, config

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfig):
        """huggingface模型转换mindspeed-mm模型权重"""
        ops, config = Qwen3_VLConverter._create_ops(cfg.hf_config.config)
        qwen_vl_hf_to_mm.convert(cfg, config, ops, qwen3_vl_tp_patterns, [vision_schema, text_schema])
        # 安全管控权限
        os.chmod(cfg.mm_dir, SAFE_MODE)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        pass

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        pass


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
        """mindspeed-mm模型转换huggingface模型权重"""
        from checkpoint.vlm_model import internvl2_mm_to_hf
        internvl2_mm_to_hf.main(cfg)
        # 安全管控权限
        os.chmod(cfg.save_hf_dir, SAFE_MODE)

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        pass


class DeepSeekVLConverter(Converter):
    """DeepSeekVL模型转换工具"""

    @staticmethod
    def hf_to_mm(cfg: ConvertMMConfig):
        """huggingface模型转换mindspeed-mm模型权重"""
        deepseekvl_hf_to_mm.main(cfg)
        # 安全管控权限
        os.chmod(cfg.mm_dir, SAFE_MODE)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        """mindspeed-mm模型转换huggingface模型权重"""
        pass

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        pass