import os
import stat
from safetensors.torch import load_file
import torch


def convert_hg_to_mm(_hg_ckpt_dir, _llm_path, _vit_hidden_size, _vit_attention_heads_num):
    hiddensize_per_head = _vit_hidden_size // _vit_attention_heads_num
    file_path = os.path.join(_hg_ckpt_dir, 'model-00001-of-00005.safetensors')
    params_visual = load_file(file_path, device='cpu')
    new_params = {}
    for key, value in params_visual.items():
        new_key = None
        #visual 权重转换部分
        if key.startswith('visual'):
            if 'merger' in key:
                if 'visual.merger.mlp.0' in key:
                    new_key = key.replace('visual.merger.mlp.0', 'image_encoder.projector.encoder.linear_fc1')
                if 'visual.merger.mlp.2' in key:
                    new_key = key.replace('visual.merger.mlp.2', 'image_encoder.projector.encoder.linear_fc2')
                if 'ln_q' in key:
                    new_key = key.replace('visual.merger.ln_q', 'image_encoder.projector.layernorm')
            if 'patch_embed' in key:
                new_key = key.replace('visual.patch_embed.proj', 'image_encoder.encoder.patch_embed.proj')
            if 'blocks' in key:
                new_key = key.replace('visual.blocks', 'image_encoder.encoder.blocks.layers')
                new_key = new_key.replace('attn.proj', 'self_attention.linear_proj')
                new_key = new_key.replace('attn.qkv', 'self_attention.linear_qkv')
                new_key = new_key.replace('mlp.fc1', 'mlp.linear_fc1')
                new_key = new_key.replace('mlp.fc2', 'mlp.linear_fc2')
                new_key = new_key.replace('norm1', 'input_layernorm')
                new_key = new_key.replace('norm2', 'pre_mlp_layernorm')

            if 'qkv.weight' in key:
                res = value * 0
                q_ = value[:_vit_hidden_size, :]
                k_ = value[_vit_hidden_size:_vit_hidden_size * 2, :]
                v_ = value[_vit_hidden_size * 2:_vit_hidden_size * 3, :]
                i = 0
                for j in range(_vit_attention_heads_num):
                    res[i * hiddensize_per_head:(i + 1) * hiddensize_per_head, :] = q_[j * hiddensize_per_head:(j + 1) * hiddensize_per_head, :]
                    res[(i + 1) * hiddensize_per_head:(i + 2) * hiddensize_per_head, :] = k_[j * hiddensize_per_head:(j + 1) * hiddensize_per_head, :]
                    res[(i + 2) * hiddensize_per_head:(i + 3) * hiddensize_per_head, :] = v_[j * hiddensize_per_head:(j + 1) * hiddensize_per_head, :]
                    i = i + 3
                new_params[new_key] = res

            elif 'qkv.bias' in key:
                res = value * 0
                q_ = value[:_vit_hidden_size]
                k_ = value[_vit_hidden_size:_vit_hidden_size * 2]
                v_ = value[_vit_hidden_size * 2:_vit_hidden_size * 3]

                i = 0
                for j in range(_vit_attention_heads_num):
                    res[i * hiddensize_per_head:(i + 1) * hiddensize_per_head] = q_[j * hiddensize_per_head:(j + 1) * hiddensize_per_head]
                    res[(i + 1) * hiddensize_per_head:(i + 2) * hiddensize_per_head] = k_[j * hiddensize_per_head:(j + 1) * hiddensize_per_head]
                    res[(i + 2) * hiddensize_per_head:(i + 3) * hiddensize_per_head] = v_[j * hiddensize_per_head:(j + 1) * hiddensize_per_head]
                    i = i + 3
                new_params[new_key] = res
            else:
                new_params[new_key] = value

    llm_ckpt = torch.load(_llm_path, map_location='cpu')['model']
    for k, v in llm_ckpt.items():
        new_params[k] = v          

    return new_params


def split_by_pp(_state_dict, _num_layers, _pipeline_layer_index=None):
    if _pipeline_layer_index is None:
        return [_state_dict, ]
    return_dicts = []

    for pp_rank, _ in enumerate(_pipeline_layer_index):
        is_first = False
        is_last = False

        if pp_rank == 0:
            is_first = True
        elif pp_rank == len(_pipeline_layer_index) - 1:
            is_last = True

        pp_start_index = _pipeline_layer_index[pp_rank]
        if is_last:
            pp_end_index = _num_layers
        else:
            pp_end_index = _pipeline_layer_index[pp_rank + 1]

        new_dict = {}
        for key, value in _state_dict.items():
            if key.startswith('image_encoder'):
                if is_first:
                    new_dict[key] = value
            elif key.startswith('embedding.word_embeddings'):
                if is_first:
                    new_dict[key] = value
            elif key.startswith('decoder.final_layernorm'):
                if is_last:
                    new_dict[key] = value
            elif key.startswith('output_layer'):
                if is_last:
                    new_dict[key] = value
            elif key.startswith('decoder.layers.'):
                layer = int(key.split('.')[2])
                if layer >= pp_start_index and layer < pp_end_index:
                    new_layer = layer - pp_start_index
                    key_li = key.split('.')
                    key_li[2] = str(new_layer)
                    new_key = '.'.join(key_li)
                    new_dict[new_key] = value

        return_dicts.append(new_dict)
    return return_dicts


def save_by_pp(_state_dicts, _save_dir, _lastest_checkpointed_iteration='release', _exists_ok=False):
    if os.path.exists(_save_dir):
        if not _exists_ok:
            print(f'save dir: {_save_dir} exists, please check.')
            return
    else:
        os.makedirs(_save_dir)
    
    flags = os.O_WRONLY | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(_save_dir, 'latest_checkpointed_iteration.txt'), flags, mode), 'w') as fout:
        fout.write(_lastest_checkpointed_iteration)

    if _lastest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(_lastest_checkpointed_iteration)
    
    if len(_state_dicts) > 1:
        for pp_rank, _state_dict in enumerate(_state_dicts):
            tp_rank = 0
            os.makedirs(os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}_{pp_rank:03d}'))
            save_path = os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}_{pp_rank:03d}', 'model_optim_rng.pt')
            save_dict = {}
            save_dict['model'] = _state_dict
            torch.save(save_dict, save_path)
    else:
        _state_dict = _state_dicts[0]
        tp_rank = 0
        os.makedirs(os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}'))
        save_path = os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}', 'model_optim_rng.pt')
        save_dict = {}
        save_dict['model'] = _state_dict
        torch.save(save_dict, save_path)


if __name__ == "__main__":
    hg_ckpt_dir = "raw_ckpt/Qwen2-VL-7B-Instruct"
    mm_save_dir = 'ckpt/Qwen2-VL-7B-Instruct'  
    pipeline_layer_index = [0, 0, 10, 20]
    num_layers = 28
    llm_path = 'llm_path/Qwen2-VL-7B-Instruct/inter_0000001/mp_rank/model_optim_rng.pt'

    vit_hidden_size = 1280
    vit_attention_heads_num = 16

    state_dict = convert_hg_to_mm(hg_ckpt_dir, llm_path, vit_hidden_size, vit_attention_heads_num)
    state_dicts = split_by_pp(state_dict, num_layers, pipeline_layer_index)
    save_by_pp(state_dicts, mm_save_dir, _exists_ok=True)
