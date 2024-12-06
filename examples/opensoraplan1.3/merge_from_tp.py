import os
import stat
import torch
import mindspeed.megatron_adaptor


def merge_from_tp(load_dir, save_dir, mode="model", exists_ok=False):
    suffixes_0 = ["atten.proj_q.weight", "atten.proj_q.bias", "atten.proj_k.weight", "atten.proj_k.bias",
                  "atten.proj_v.weight", "atten.proj_v.bias", "ff.net.0.proj.weight", "ff.net.0.proj.bias"]
    suffixes_1 = ["atten.proj_out.weight", "ff.net.2.weight"]
    flags = os.O_RDONLY
    stat_mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(load_dir, 'latest_checkpointed_iteration.txt'), flags, stat_mode), 'r') as fout:
        latest_checkpointed_iteration = fout.read()
    if latest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(int(latest_checkpointed_iteration))
    directory_path = os.path.join(load_dir, directory)
    if os.path.exists(directory_path):
        tp_size = len(os.listdir(directory_path))
        dict_list = []
        for tp_rank in range(tp_size):
            dict_path = os.path.join(load_dir, directory, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
            load_dict = torch.load(dict_path, map_location='cpu')
            if mode == "model":
                model_dict = load_dict["model"]
            elif mode == "ema_model":
                model_dict = load_dict["ema_model"]
            else:
                raise ValueError(f"unsupported mode: {mode}")
            dict_list.append(model_dict)
        new_dict = {}
        for key, value in dict_list[0].items():
            if isinstance(value, torch.Tensor):
                if any(key.endswith(suffix) for suffix in suffixes_0):
                    new_values = []
                    for tp_rank in range(tp_size):
                        new_values.append(dict_list[tp_rank][key])
                    new_value = torch.cat(new_values, dim=0)
                    new_dict[key] = new_value
                    del new_value
                elif any(key.endswith(suffix) for suffix in suffixes_1):
                    new_values = []
                    for tp_rank in range(tp_size):
                        new_values.append(dict_list[tp_rank][key])
                    new_value = torch.cat(new_values, dim=1)
                    new_dict[key] = new_value
                    del new_value
                else:
                    new_dict[key] = value
            else:
                new_dict[key] = value
                print(f"key: {key}, Type: {type(value)}")
        if os.path.exists(save_dir):
            if not exists_ok:
                print(f"save dir: {save_dir} exists, please check.")
                return
        else:
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "model_mm.pt")
        torch.save(new_dict, save_path)
    else:
        raise ValueError(f"Invalid path: {directory_path}")


if __name__ == "__main__":
    dit_tp_weight_path = "local trained tp open sora plan weight path"
    dit_mm_save_dir = "dir to save dit weights after transfer to MindSpeed-MM"
    MODE = "model"

    merge_from_tp(dit_tp_weight_path, dit_mm_save_dir, mode=MODE, exists_ok=False)