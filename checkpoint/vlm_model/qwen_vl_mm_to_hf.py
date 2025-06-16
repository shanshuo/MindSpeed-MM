from typing import Callable, Any, List

from checkpoint.vlm_model.operator import Operator, TP_PATTERN_T
from checkpoint.vlm_model.utils import ConvertHFConfig, load_from_mm, merge_by_tp, convert_mm_to_hf, \
    split_by_index_json, copy_files_except_suffix, save_by_index_json


def convert(convert_config: ConvertHFConfig, config: Any, ops: List[Operator],
            tp_patterns: TP_PATTERN_T):
    parallel_config = convert_config.parallel_config
    # 加载权重字典
    state_dicts = load_from_mm(convert_config.mm_dir, parallel_config.vit_pp_layers, parallel_config.llm_pp_layers,
                               parallel_config.tp_size, parallel_config.audio_pp_layers)
    state_dict = merge_by_tp(state_dicts, tp_patterns)
    state_dict = convert_mm_to_hf(state_dict, ops, True, True)
    state_dicts = split_by_index_json(state_dict, convert_config.hf_config.hf_dir)
    copy_files_except_suffix(convert_config.hf_config.hf_dir, convert_config.save_hf_dir)
    save_by_index_json(state_dicts, convert_config.save_hf_dir)



