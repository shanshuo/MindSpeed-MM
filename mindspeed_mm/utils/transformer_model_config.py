from dataclasses import dataclass
import torch.nn.functional as F

from megatron.core import ModelParallelConfig
from megatron.core.transformer import TransformerConfig
from megatron.training import get_args

from mindspeed_mm.configs.config import ConfigReader
from .utils import get_dtype, quick_gelu


def get_class_variables(cls):
    all_members = dir(cls)
    filtered_members = [member for member in all_members if not member.startswith("__")]

    return filtered_members


def get_model_config(config):
    global_args = get_args()
    config_dict = config.to_dict()
    if 'model_id' in config_dict and config_dict['model_id'] == 'InternVLMLP':
        config_dict['params_dtype'] = "bf16"
        config_dict['hidden_size'] = 4096
        config_dict['num_attention_heads'] = 1
        config_dict['num_layers'] = 1
    if 'model_id' in config_dict and config_dict['model_id'] == 'Qwen2.5llm':
        config_dict['use_repeat_kv'] = True

    t_config = dict()
    tfc_variables = get_class_variables(TransformerConfig)
    mpc_variables = get_class_variables(ModelParallelConfig)
    for key in tfc_variables:
        if key in config_dict.keys():
            t_config[key] = config_dict[key]
        elif key in mpc_variables and hasattr(global_args, key):
            t_config[key] = getattr(global_args, key)

    t_config["params_dtype"] = get_dtype(t_config.get("params_dtype"))
    if t_config.get("activation_func") == "silu":
        t_config["activation_func"] = F.silu
    elif t_config.get("activation_func") == "quick_gelu":
        t_config["activation_func"] = quick_gelu
    else:
        t_config["activation_func"] = F.gelu
    
    trans_config = MMTransformerConfig(**t_config)

    for key in tfc_variables:
        config_dict[key] = getattr(trans_config, key)
    new_config = ConfigReader(config_dict)

    return new_config


@dataclass
class MMTransformerConfig(TransformerConfig):
    def __post_init__(self):
        """Modified from __post_init__ method of TransformerConfig to adapt MLP config"""
        if self.kv_channels is None and self.num_attention_heads == 0:
            self.kv_channels = 0
        if self.pipeline_dtype is None:
            self.pipeline_dtype = self.params_dtype
        super().__post_init__()