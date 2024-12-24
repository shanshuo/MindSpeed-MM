from qwen2vl_convert_to_hf import load_from_mm, check_pp_config
from qwen2vl_convert_to_mm_ckpt import split_model_by_pipeline, save_by_pp, merge_pp_index

if __name__ == "__main__":
    mm_save_dir = "save_dir"  # 微调后保存的权重目录
    new_save_dir = "new_pp_save_dir"  # 希望重新pp切分后保存的目录

    vit_num_layers = 32
    llm_num_layers = 28

    old_pp_size = 4
    old_vit_pipeline_num_layers = [32, 0, 0, 0]
    old_llm_pipeline_num_layers = [1, 6, 11, 10]

    new_pp_size = 2
    new_vit_pipeline_num_layers = [32, 0]
    new_llm_pipeline_num_layers = [14, 14]

    check_pp_config(old_pp_size, vit_num_layers, old_vit_pipeline_num_layers, llm_num_layers,
                    old_llm_pipeline_num_layers)
    check_pp_config(new_pp_size, vit_num_layers, new_vit_pipeline_num_layers, llm_num_layers,
                    new_llm_pipeline_num_layers)
    state_dict = load_from_mm(mm_save_dir, old_vit_pipeline_num_layers, old_llm_pipeline_num_layers)
    pp_split = merge_pp_index(new_pp_size, vit_num_layers, new_vit_pipeline_num_layers, llm_num_layers,
                              new_llm_pipeline_num_layers)
    state_dicts, _ = split_model_by_pipeline(state_dict, pp_split)

    for rank, pipeline_state_dict in enumerate(state_dicts):
        print(20 * '#', f'stage {rank}', 20 * '#')
        for key, value in pipeline_state_dict.items():
            if value is not None:
                print(key, value.shape)
    save_by_pp(state_dicts, new_save_dir, _exists_ok=True)
