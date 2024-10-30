import os
from copy import deepcopy
import stat

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from safetensors.torch import load_file as safe_load


def load_weight(weight_path):
    if weight_path.endswith('.safetensors'):
        return safe_load(weight_path)
    elif weight_path.endswith('.pt') or weight_path.endswith('.pth'):
        return torch.load(weight_path, map_location='cpu')
    else:
        raise ValueError(f"Unsupported file type: {weight_path}")


def load_from_hf(load_path):
    ckpt_dict = {}
    if os.path.isdir(load_path):
        for filename in os.listdir(load_path):
            file_path = os.path.join(load_path, filename)
            if os.path.isfile(file_path):
                ckpt_dict_part = load_weight(file_path)
                ckpt_dict.update(ckpt_dict_part)
    elif os.path.isfile(load_path):
        ckpt_dict = load_weight(load_path)
    else:
        raise ValueError(f"Invalid path: {load_path}")
    return ckpt_dict


def convert_hg_to_mm(state_dict):
    new_checkpoint = {}
    state_dict = state_dict.get("ema_state_dict", state_dict)

    for key, value in state_dict.items():
        new_key = key.replace("transformer_blocks", "videodit_sparse_blocks")
        new_key = new_key.replace("attn1", "self_atten")
        new_key = new_key.replace("attn2", "cross_atten")
        new_key = new_key.replace("to_q", "proj_q")
        new_key = new_key.replace("to_k", "proj_k")
        new_key = new_key.replace("to_v", "proj_v")
        new_key = new_key.replace("to_out.0", "proj_out")
        new_key = new_key.replace("to_out.1", "dropout")
        new_key = new_key.replace("module.", "")
        new_checkpoint[new_key] = value

    return new_checkpoint


def split_by_tp(state_dicts, tp_size):
    if tp_size == 0:
        return [state_dicts, ]

    copy_dict = deepcopy(state_dicts)
    print("total keys: %s" % len(copy_dict))
    total_params = sum(param.numel() for param in copy_dict.values())
    print(f"Total number of parameters before convert : {total_params / 1e9} B.")

    return_dicts = []
    suffixes_0 = ["atten.proj_q.weight", "atten.proj_q.bias", "atten.proj_k.weight", "atten.proj_k.bias",
                  "atten.proj_v.weight", "atten.proj_v.bias", "ff.net.0.proj.weight", "ff.net.0.proj.bias"]
    suffixes_1 = ["atten.proj_out.weight", "ff.net.2.weight"]

    for tp_rank in range(tp_size):
        new_dict = {}
        for key, value in copy_dict.items():
            if isinstance(value, torch.Tensor):
                if any(key.endswith(suffix) for suffix in suffixes_0):
                    value_copy = deepcopy(value)
                    value_copy = torch.chunk(value_copy, tp_size, dim=0)[tp_rank]
                    new_dict[key] = value_copy
                    del value_copy
                elif any(key.endswith(suffix) for suffix in suffixes_1):
                    value_copy = deepcopy(value)
                    value_copy = torch.chunk(value_copy, tp_size, dim=1)[tp_rank]
                    new_dict[key] = value_copy
                    del value_copy
                else:
                    new_dict[key] = value
            else:
                new_dict[key] = value
                print(f"key: {key}, Type:{type(value)}")

        total_params = sum(param.numel() for param in new_dict.values())
        print(f"Total number of parameters after convert on TP rank:{tp_rank} : {total_params / 1e9} B.")
        return_dicts.append(new_dict)
    del copy_dict
    return return_dicts


def save_by_tp(state_dicts, save_dir, latest_checkpointed_iteration='release', exists_ok=False):
    if os.path.exists(save_dir):
        if not exists_ok:
            print(f"save dir: {save_dir} exists, please check.")
            return
    else:
        os.makedirs(save_dir)

    flags = os.O_WRONLY | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(save_dir, 'latest_checkpointed_iteration.txt'), flags, mode), 'w') as fout:
        fout.write(latest_checkpointed_iteration)
    if latest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(latest_checkpointed_iteration)

    for tp_rank, state_dict in enumerate(state_dicts):
        os.makedirs(os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}"))
        save_path = os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
        save_dict = {}
        save_dict['model'] = state_dict
        torch.save(save_dict, save_path)


def save_vae(_state_dict, save_dir, exists_ok=False):
    if os.path.exists(save_dir):
        if not exists_ok:
            print(f"save dir: {save_dir} exists, please check.")
            return
        else:
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "wfvae_mm.pt")
        torch.save(_state_dict, save_path)


if __name__ == "__main__":
    # 参数配置
    TP_SIZE = 1

    dit_hg_weight_path = "local downloaded open sora plan weight path"
    dit_mm_save_dir = "dir to save dit weights after transfer to MindSpeed-MM"

    vae_hg_weight_path = "local downloaded vae weight path"
    vae_mm_save_dir = "dir to save vae weights after transfer to MindSpeed-MM"

    # 转换dit权重
    dit_state_dict = load_from_hf(dit_hg_weight_path)
    dit_state_dict = convert_hg_to_mm(dit_state_dict)
    dit_state_dicts = split_by_tp(dit_state_dict, TP_SIZE)
    save_by_tp(dit_state_dicts, dit_mm_save_dir, exists_ok=False)

    # 转换VAE权重
    vae_state_dict = load_from_hf(vae_hg_weight_path)
    vae_state_dict = convert_hg_to_mm(vae_state_dict)
    save_vae(vae_state_dict, vae_mm_save_dir, exists_ok=False)
