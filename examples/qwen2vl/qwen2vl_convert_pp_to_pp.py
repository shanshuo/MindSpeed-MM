from qwen2vl_convert_to_hf import load_from_mm, check_pp_config
from qwen2vl_convert_to_mm_ckpt import split_model_by_pipeline, save_by_pp, merge_pp_index

if __name__ == "__main__":
    mm_save_dir = "save_dir"  # 微调后保存的权重目录
    new_save_dir = "new_pp_save_dir"  # 希望重新pp切分后保存的目录

    vit_num_layers = 32
    llm_num_layers = 28

    original_tp_size = 1

    original_pp_size = 4
    original_vit_pipeline_num_layers = [32, 0, 0, 0]
    original_llm_pipeline_num_layers = [1, 6, 11, 10]

    revised_pp_size = 2
    revised_vit_pipeline_num_layers = [32, 0]
    revised_llm_pipeline_num_layers = [14, 14]

    check_pp_config(original_pp_size, vit_num_layers, original_vit_pipeline_num_layers, llm_num_layers,
                    original_llm_pipeline_num_layers)
    check_pp_config(revised_pp_size, vit_num_layers, revised_vit_pipeline_num_layers, llm_num_layers,
                    revised_llm_pipeline_num_layers)
    state_dicts = load_from_mm(mm_save_dir, original_vit_pipeline_num_layers, original_llm_pipeline_num_layers)
    pp_split = merge_pp_index(revised_pp_size, vit_num_layers, revised_vit_pipeline_num_layers, llm_num_layers,
                              revised_llm_pipeline_num_layers)
    for i in range(original_tp_size):
        pp_state_dicts, _ = split_model_by_pipeline(state_dicts[i], pp_split)
        for rank, pipeline_state_dict in enumerate(state_dicts):
            print(20 * '#', f'stage {rank}', 20 * '#')
            for key, value in pipeline_state_dict.items():
                if value is not None:
                    print(key, value.shape)
        save_by_pp(pp_state_dicts, new_save_dir, _exists_ok=True, _tp_rank=i)
