import os
import stat
from pathlib import Path
from copy import deepcopy

import torch
from safetensors.torch import load_file

MODEL_CONFIG_DICT = {
    '2B': {
        'model_size': '2B',
        'vit_hidden_size': 1280,
        'vit_num_attention_heads': 16,
        'vit_num_layers': 32,
        'llm_hidden_size': 1536,
        'llm_num_query_groups': 2,
        'llm_num_attention_heads': 12,
        'llm_num_layers': 28,
    },
    '7B': {
        'model_size': '7B',
        'vit_hidden_size': 1280,
        'vit_num_attention_heads': 16,
        'vit_num_layers': 32,
        'llm_hidden_size': 3584,
        'llm_num_query_groups': 4,
        'llm_num_attention_heads': 28,
        'llm_num_layers': 28,
    },
    '72B': {
        'model_size': '72B',
        'vit_hidden_size': 1280,
        'vit_num_attention_heads': 16,
        'vit_num_layers': 32,
        'llm_hidden_size': 8192,
        'llm_num_query_groups': 8,
        'llm_num_attention_heads': 64,
        'llm_num_layers': 80,
    }
}


def load_from_hf(_load_dir):
    # Load Huggingface model 。
    load_dir = Path(_load_dir)
    safetensors_files = list(load_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No *.safetensors files found in {load_dir}")
    state_dict = {}
    for safe_path in safetensors_files:
        state_dict.update(load_file(str(safe_path), device='cpu'))
    return state_dict


def convert_hf_to_mm(_state_dict, _num_layers, _vit_hidden_size, _vit_attention_heads_num, _llm_num_query_groups):
    hiddensize_per_head = _vit_hidden_size // _vit_attention_heads_num
    new_params = {}
    for key, value in _state_dict.items():
        new_key = None
        # visual 权重转换部分
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
                    res[i * hiddensize_per_head:(i + 1) * hiddensize_per_head, :] = q_[j * hiddensize_per_head:(
                                                                                                                       j + 1) * hiddensize_per_head,
                                                                                    :]
                    res[(i + 1) * hiddensize_per_head:(i + 2) * hiddensize_per_head, :] = k_[j * hiddensize_per_head:(
                                                                                                                             j + 1) * hiddensize_per_head,
                                                                                          :]
                    res[(i + 2) * hiddensize_per_head:(i + 3) * hiddensize_per_head, :] = v_[j * hiddensize_per_head:(
                                                                                                                             j + 1) * hiddensize_per_head,
                                                                                          :]
                    i = i + 3
                new_params[new_key] = res

            elif 'qkv.bias' in key:
                res = value * 0
                q_ = value[:_vit_hidden_size]
                k_ = value[_vit_hidden_size:_vit_hidden_size * 2]
                v_ = value[_vit_hidden_size * 2:_vit_hidden_size * 3]

                i = 0
                for j in range(_vit_attention_heads_num):
                    res[i * hiddensize_per_head:(i + 1) * hiddensize_per_head] = q_[j * hiddensize_per_head:(
                                                                                                                    j + 1) * hiddensize_per_head]
                    res[(i + 1) * hiddensize_per_head:(i + 2) * hiddensize_per_head] = k_[j * hiddensize_per_head:(
                                                                                                                          j + 1) * hiddensize_per_head]
                    res[(i + 2) * hiddensize_per_head:(i + 3) * hiddensize_per_head] = v_[j * hiddensize_per_head:(
                                                                                                                          j + 1) * hiddensize_per_head]
                    i = i + 3
                new_params[new_key] = res
            else:
                new_params[new_key] = value
        else:
            if key.startswith('lm_head'):
                new_key = key.replace('lm_head', 'text_decoder.output_layer')
            elif key.startswith('model'):
                new_key = key.replace('model.layers', 'text_decoder.decoder.layers')
                new_key = new_key.replace('self_attn.o_proj.weight', 'self_attention.linear_proj.weight')
                new_key = new_key.replace('self_attn.q_proj.weight', 'self_attention.linear_q.weight')
                new_key = new_key.replace('self_attn.k_proj.weight', 'self_attention.linear_k.weight')
                new_key = new_key.replace('self_attn.v_proj.weight', 'self_attention.linear_v.weight')
                new_key = new_key.replace('self_attn.q_proj.bias', 'self_attention.linear_q.bias')
                new_key = new_key.replace('self_attn.k_proj.bias', 'self_attention.linear_k.bias')
                new_key = new_key.replace('self_attn.v_proj.bias', 'self_attention.linear_v.bias')
                new_key = new_key.replace('post_attention_layernorm.weight', 'pre_mlp_layernorm.weight')
                new_key = new_key.replace('mlp.gate_proj.weight', 'mlp.linear_fc1_gate.weight')
                new_key = new_key.replace('mlp.up_proj.weight', 'mlp.linear_fc1_up.weight')
                new_key = new_key.replace('mlp.down_proj.weight', 'mlp.linear_fc2.weight')
                new_key = new_key.replace('model.norm.weight', 'text_decoder.decoder.final_layernorm.weight')
                new_key = new_key.replace('model.embed_tokens.weight', 'text_decoder.embedding.word_embeddings.weight')

            new_params[new_key] = value
    for i in range(_num_layers):
        # 合并gate up
        gate_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1_gate.weight'
        up_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1_up.weight'
        fc1_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1.weight'
        # 如果权重名字在新字典中，则获取对应权重值
        # 合并 w1 和 w3
        if gate_name in new_params.keys():
            gate_proj_weight = new_params[gate_name]
        if up_name in new_params.keys():
            up_proj_weight = new_params[up_name]
        # 将 w1 和 w3 沿着第0维度进行拼接
        linear_fc1 = torch.cat([gate_proj_weight, up_proj_weight], dim=0)

        new_params[fc1_name] = linear_fc1
        # 移除合并前的权重
        if gate_name in new_params:
            new_params.pop(gate_name)
        if up_name in new_params:
            new_params.pop(up_name)

    for i in range(_num_layers):
        # 合并q k v weight
        attention_q = f'text_decoder.decoder.layers.{i}.self_attention.linear_q.weight'
        attention_k = f'text_decoder.decoder.layers.{i}.self_attention.linear_k.weight'
        attention_v = f'text_decoder.decoder.layers.{i}.self_attention.linear_v.weight'
        attention_qkv = f'text_decoder.decoder.layers.{i}.self_attention.linear_qkv.weight'
        if attention_q in new_params.keys():
            attention_q_weight = new_params[attention_q]
        if attention_k in new_params.keys():
            attention_k_weight = new_params[attention_k]
        if attention_v in new_params.keys():
            attention_v_weight = new_params[attention_v]

        q_chunks = torch.chunk(attention_q_weight, _llm_num_query_groups, dim=0)
        k_chunks = torch.chunk(attention_k_weight, _llm_num_query_groups, dim=0)
        v_chunks = torch.chunk(attention_v_weight, _llm_num_query_groups, dim=0)
        all_chunks = []
        for j in range(_llm_num_query_groups):
            all_chunks.append(q_chunks[j])
            all_chunks.append(k_chunks[j])
            all_chunks.append(v_chunks[j])
        concatenated_tensor = torch.cat(all_chunks, dim=0)
        new_params[attention_qkv] = concatenated_tensor
        if attention_q in new_params:
            new_params.pop(attention_q)
        if attention_k in new_params:
            new_params.pop(attention_k)
        if attention_v in new_params:
            new_params.pop(attention_v)

    for i in range(_num_layers):
        # 合并q k v bias
        attention_q1 = f'text_decoder.decoder.layers.{i}.self_attention.linear_q.bias'
        attention_k1 = f'text_decoder.decoder.layers.{i}.self_attention.linear_k.bias'
        attention_v1 = f'text_decoder.decoder.layers.{i}.self_attention.linear_v.bias'
        attention_qkv1 = f'text_decoder.decoder.layers.{i}.self_attention.linear_qkv.bias'
        if attention_q1 in new_params.keys():
            attention_q_bias = new_params[attention_q1]
        else:
            continue
        if attention_k1 in new_params.keys():
            attention_k_bias = new_params[attention_k1]
        else:
            continue
        if attention_v1 in new_params.keys():
            attention_v_bias = new_params[attention_v1]
        else:
            continue

        q_chunks1 = torch.chunk(attention_q_bias, _llm_num_query_groups, dim=0)
        k_chunks1 = torch.chunk(attention_k_bias, _llm_num_query_groups, dim=0)
        v_chunks1 = torch.chunk(attention_v_bias, _llm_num_query_groups, dim=0)
        all_chunks1 = []
        for j in range(_llm_num_query_groups):
            all_chunks1.append(q_chunks1[j])
            all_chunks1.append(k_chunks1[j])
            all_chunks1.append(v_chunks1[j])
        concatenated_tensor1 = torch.cat(all_chunks1, dim=0)
        new_params[attention_qkv1] = concatenated_tensor1
        if attention_q1 in new_params:
            new_params.pop(attention_q1)
        if attention_k1 in new_params:
            new_params.pop(attention_k1)
        if attention_v1 in new_params:
            new_params.pop(attention_v1)

    return new_params


def merge_pp_index(pp_size, vit_num_layers, vit_pipeline_num_layers, llm_num_layers, llm_pipeline_num_layers):
    if len(vit_pipeline_num_layers) != pp_size:
        raise AssertionError(f'length of vit_pipeline_num_layers must be equal to pp_size, '
                             f'but got {len(vit_pipeline_num_layers)} and {pp_size}.')
    if sum(vit_pipeline_num_layers) != vit_num_layers:
        raise AssertionError(f'sum of vit_pipeline_num_layers must be equal to vit_num_layers, '
                             f'but got {sum(vit_pipeline_num_layers)} and {vit_num_layers}.')
    if len(llm_pipeline_num_layers) != pp_size:
        raise AssertionError(f'length of llm_pipeline_num_layers must be equal to pp_size, '
                             f'but got {len(llm_pipeline_num_layers)} and {pp_size}.')
    if sum(llm_pipeline_num_layers) != llm_num_layers:
        raise AssertionError(f'sum of llm_pipeline_num_layers must be equal to llm_num_layers, '
                             f'but got {sum(llm_pipeline_num_layers)} and {llm_num_layers}.')
    split_method = []
    for vit_num, llm_num in zip(vit_pipeline_num_layers, llm_pipeline_num_layers):
        split_method.append((vit_num, llm_num))
    return split_method


def split_model_by_pipeline(state_dict, pp_split):
    if pp_split is None or len(pp_split) <= 1:
        return [state_dict], {}

    pp_size = len(pp_split)
    vit_range = [0, 0]
    llm_range = [pp_size - 1, pp_size - 1]
    for pp_rank, (vit_num, llm_num) in enumerate(pp_split):
        if vit_num > 0 and pp_rank > vit_range[1]:
            vit_range[1] = pp_rank
        if llm_num > 0 and pp_rank < llm_range[0]:
            llm_range[0] = pp_rank
    print(f'vit range: {vit_range[0]}~{vit_range[1]}')
    print(f'llm range: {llm_range[0]}~{llm_range[1]}')

    vit_start_idx = 0
    llm_start_idx = 0
    return_dicts = []
    copy_dict = deepcopy(state_dict)
    for pp_rank, (vit_num, llm_num) in enumerate(pp_split):
        vit_end_idx = vit_start_idx + vit_num
        llm_end_idx = llm_start_idx + llm_num
        new_dict = {}
        for key, value in state_dict.items():
            if key.startswith('image_encoder.encoder.patch_embed.'):
                if pp_rank == vit_range[0]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('image_encoder.encoder.blocks.layers.'):
                layer_idx = int(key.split('.')[4])
                if vit_start_idx <= layer_idx < vit_end_idx and vit_range[0] <= pp_rank <= vit_range[1]:
                    new_idx = layer_idx - vit_start_idx
                    new_key = key.replace(f'{layer_idx}', f'{new_idx}', 1)
                    new_dict[new_key] = value
                    copy_dict.pop(key)
            elif key.startswith('image_encoder.projector.'):
                if pp_rank == vit_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.embedding.'):
                if pp_rank == llm_range[0]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.decoder.layers.'):
                layer_idx = int(key.split('.')[3])
                if llm_start_idx <= layer_idx < llm_end_idx and llm_range[0] <= pp_rank <= llm_range[1]:
                    new_idx = layer_idx - llm_start_idx
                    new_key = key.replace(f'{layer_idx}', f'{new_idx}', 1)
                    new_dict[new_key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.decoder.final_layernorm.'):
                if pp_rank == llm_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.output_layer.'):
                if pp_rank == llm_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
        vit_start_idx = vit_end_idx
        llm_start_idx = llm_end_idx
        return_dicts.append(new_dict)
    return return_dicts, copy_dict


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
    hf_ckpt_dir = "ckpt/hf_path/Qwen2-VL-7B-Instruct"  # hugging face原始的权重保存路径
    mm_save_dir = 'ckpt/mm_path/Qwen2-VL-7B-Instruct'  # 转换后的权重保存路径
    model_size = "7B"  # 根据需要转换的模型，指定配置（ 2B 7B 72B ）
    #model parameters
    model_config = MODEL_CONFIG_DICT[model_size]

    #PP parameters: 7B
    pp_size = 4
    vit_pipeline_num_layers = [32, 0, 0, 0]
    llm_pipeline_num_layers = [1, 6, 11, 10]

    state_dict = load_from_hf(hf_ckpt_dir)
    state_dict = convert_hf_to_mm(state_dict, model_config["llm_num_layers"], model_config["vit_hidden_size"], model_config["vit_num_attention_heads"], model_config["llm_num_query_groups"])
    pp_split = merge_pp_index(pp_size, model_config["vit_num_layers"], vit_pipeline_num_layers, model_config["llm_num_layers"], llm_pipeline_num_layers)
    state_dicts, remains = split_model_by_pipeline(state_dict, pp_split)
    if len(remains) > 0:
        print(remains)
        raise RuntimeWarning("There are some weights ungrouped.")

    for rank, pipeline_state_dict in enumerate(state_dicts):
        print(20 * '#', f'stage {rank}', 20 * '#')
        for key, value in pipeline_state_dict.items():
            print(key, value.shape)
    save_by_pp(state_dicts, mm_save_dir, _exists_ok=True)
