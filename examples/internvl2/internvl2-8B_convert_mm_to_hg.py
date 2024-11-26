import json
from pathlib import Path
import torch
from safetensors.torch import save_file


def merge_by_pp(pp_ckpt_file, pp_rank: int, _pipeline_num_layers=None):
    # pp length: 6, 9, 9, 8
    pp_start_index = 0
    if pp_rank > 0:
        pp_start_index = sum(_pipeline_num_layers[0: pp_rank - 1])
    new_dict = {}
    for parameter_name, tensor in torch.load(pp_ckpt_file)['model'].items():
        if parameter_name.startswith('text_decoder.decoder.layers.'):
            parameter_name_list = parameter_name.split('.')
            layer_id = int(parameter_name_list[3])
            merge_pp_mapp_layer_id = layer_id + pp_start_index
            parameter_name_list[3] = str(merge_pp_mapp_layer_id)
            merge_parameter_name = '.'.join(parameter_name_list)
            new_dict[merge_parameter_name] = tensor
        else:
            new_dict[parameter_name] = tensor
    return new_dict


def load_from_mm(_load_dir, _pipeline_num_layers=None):
    latest_iter_txt = "latest_checkpointed_iteration.txt"
    mm_load_dir = Path(_load_dir)
    ckpt_iteration_str = mm_load_dir.joinpath(latest_iter_txt).read_text().strip()
    ckpt_iter_dir_str = 'release'
    if ckpt_iteration_str != 'release':
        ckpt_iter_dir_str = 'iter_{:07d}'.format(int(ckpt_iteration_str))
    ckpt_iter_dir = mm_load_dir.joinpath(ckpt_iter_dir_str)
    _merge_state_dict = {}
    for pp_ckpt_dir in ckpt_iter_dir.iterdir():
        if not pp_ckpt_dir.is_dir():
            print(f'mm ckpt: {ckpt_iter_dir} has no pp split subdir, please check.')
            return _merge_state_dict
        for pp_ckpt_file in pp_ckpt_dir.glob('*.pt'):
            print(str(pp_ckpt_file).center(100, '_'))
            pp_ckpt_file_list = str(pp_ckpt_dir).split("_")
            pp_rank = int(pp_ckpt_file_list[-1])
            # if you want to get tp_rank add code: int(pp_ckpt_file_list[-2])
            _merge_state_dict.update(merge_by_pp(pp_ckpt_file, pp_rank, _pipeline_num_layers))

    if not _merge_state_dict:
        print(f'mm ckpt: {_load_dir} load failed, please check.')
    return _merge_state_dict


def convert_mm_to_hg(_mm_state_dict, _num_layers):
    _hg_state_dict = {}
    for key, value in _mm_state_dict.items():
        new_key = None
        if key.startswith('image_encoder'):
            new_key = key.replace('image_encoder.projector.norm', 'mlp1.0')
            new_key = new_key.replace('image_encoder.projector.linear_fc1', 'mlp1.1')
            new_key = new_key.replace('image_encoder.projector.linear_fc2', 'mlp1.3')
            new_key = new_key.replace('image_encoder.encoder', 'vision_model')
            new_key = new_key.replace('input_layernorm', 'norm1')
            new_key = new_key.replace('pre_mlp_layernorm', 'norm2')
            new_key = new_key.replace('self_attention', 'attn')
            new_key = new_key.replace('core_attention', 'inner_attn')
            new_key = new_key.replace('q_layernorm', 'q_norm')
            new_key = new_key.replace('k_layernorm', 'k_norm')
            new_key = new_key.replace('linear_qkv', 'qkv')
            new_key = new_key.replace('linear_proj', 'proj')
            new_key = new_key.replace('mlp.linear_fc1', 'mlp.fc1')
            new_key = new_key.replace('mlp.linear_fc2', 'mlp.fc2')

        elif key.startswith('text_decoder'):
            new_key = key.replace('text_decoder.embedding.word_embeddings', 'language_model.model.tok_embeddings')
            new_key = new_key.replace('text_decoder.decoder.final_layernorm', 'language_model.model.norm')
            new_key = new_key.replace('text_decoder.output_layer', 'language_model.output')
            new_key = new_key.replace('text_decoder.decoder', 'language_model.model')
            new_key = new_key.replace('input_layernorm', 'attention_norm')
            new_key = new_key.replace('self_attention.linear_proj', 'attention.wo')
            new_key = new_key.replace('self_attention.linear_qkv', 'attention.wqkv')
            new_key = new_key.replace('pre_mlp_layernorm', 'ffn_norm')
            # gate && up for w1 && w3
            new_key = new_key.replace('mlp.linear_fc1', 'feed_forward.w1w3')
            new_key = new_key.replace('mlp.linear_fc2', 'feed_forward.w2')

        print(f'mapping {key} to {new_key}')
        _hg_state_dict[new_key] = value

    # split w1 and w3 weight
    for i in range(_num_layers):
        gate_and_up_name = f'language_model.model.layers.{i}.feed_forward.w1w3.weight'
        gate_name = f'language_model.model.layers.{i}.feed_forward.w1.weight'
        up_name = f'language_model.model.layers.{i}.feed_forward.w3.weight'
        # split w1 和 w3
        if gate_and_up_name in _hg_state_dict.keys():
            gate_and_up_weight = _hg_state_dict[gate_and_up_name]
            # refer to: torch.cat([gate_proj_weight, up_proj_weight], dim=0)
            gate_weight, up_weight = torch.split(gate_and_up_weight, gate_and_up_weight.size(0) // 2, dim=0)
            _hg_state_dict[gate_name] = gate_weight
            _hg_state_dict[up_name] = up_weight
        # remove useless weight
        _hg_state_dict.pop(gate_and_up_name)
        print(f'split {gate_and_up_name} to {gate_name} and {up_name}')
    return _hg_state_dict


def split_by_index_json(_state_dict, _index_json_path):
    return_dicts = []
    with open(_index_json_path, 'r', encoding='utf-8') as file:
        weight_map = json.load(file)['weight_map']
    for key, value in weight_map.items():
        index = int(value.split('-')[1])
        while index > len(return_dicts):
            return_dicts.append({})
        return_dicts[index - 1][key] = _state_dict[key]
    return return_dicts


def save_by_index_json(_state_dicts, _save_dir):
    for index, state_dict in enumerate(_state_dicts, start=1):
        name = f'model-{index:05}-of-{len(_state_dicts):05}.safetensors'
        save_file(state_dict, Path(_save_dir).joinpath(name))


if __name__ == "__main__":
    mm_save_dir = "./save_ckpt/mm-InternVL2-8B_pp4"
    hg_save_dir = "./save_ckpt/hg_InternVL2-8B"
    index_json_path = "./raw_ckpt/InternVL2-8B/model.safetensors.index.json"
    # pp length: 6, 9, 9, 8
    pipeline_num_layers = [6, 9, 9, 8]
    num_layers = 32
    merge_state_dict = load_from_mm(mm_save_dir, pipeline_num_layers)
    hg_state_dict = convert_mm_to_hg(merge_state_dict, num_layers)
    state_dicts = split_by_index_json(hg_state_dict, index_json_path)
    save_by_index_json(state_dicts, hg_save_dir)
