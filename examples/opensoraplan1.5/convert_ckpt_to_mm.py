import os
import stat
import copy
import argparse
import torch


def convert_vae(state_dict, use_ema_model=True):
    if (
        "ema_state_dict" in state_dict
        and len(state_dict["ema_state_dict"]) > 0
        and use_ema_model
    ):
        state_dict = state_dict["ema_state_dict"]
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    elif "state_dict" in state_dict:
        if "gen_model" in state_dict["state_dict"]:
            state_dict = state_dict["state_dict"]["gen_model"]
        else:
            state_dict = state_dict["state_dict"]
    return state_dict


def replace_dit_state_dict(state_dict):
    old_keys = list(state_dict.keys())
    for old_key in old_keys:
        if "_extra_state" in old_key:
            state_dict.pop(old_key)
            continue
        key = old_key.replace("attn1.norm_q", "attn1.norm_proj_q")
        key = key.replace("attn1.norm_k", "attn1.norm_proj_k")
        key = key.replace("attn1.to_q", "attn1.proj_q")
        key = key.replace("attn1.to_k", "attn1.proj_k")
        key = key.replace("attn1.to_v", "attn1.proj_v")
        key = key.replace("attn1.add_q_proj", "attn1.added_proj_q")
        key = key.replace("attn1.add_k_proj", "attn1.added_proj_k")
        key = key.replace("attn1.add_v_proj", "attn1.added_proj_v")
        key = key.replace("attn1.to_out.0", "attn1.proj_out")
        key = key.replace("attn1.to_add_out", "attn1.added_proj_out")
        key = key.replace("attn1.norm_added_q", "attn1.norm_added_proj_q")
        key = key.replace("attn1.norm_added_k", "attn1.norm_added_proj_k")

        state_dict[key] = state_dict.pop(old_key)
    return state_dict


def split_by_tp(state_dict, tp_size):
    column_tp_names = [
        "attn1.proj_q.weight", "attn1.proj_q.bias",
        "attn1.proj_k.weight", "attn1.proj_k.bias", 
        "attn1.proj_v.weight", "attn1.proj_v.bias",
        "attn1.added_proj_q.weight", "attn1.added_proj_q.bias",
        "attn1.added_proj_k.weight", "attn1.added_proj_k.bias",
        "attn1.added_proj_v.weight", "attn1.added_proj_v.bias",
        "net.0.proj.weight", "net.0.proj.bias",
        "linear.weight", "linear.bias"
    ]
    row_tp_names = [
        "attn1.proj_out.weight", "attn1.added_proj_out.weight",
        "net.2.weight",
    ]

    tp_state_dicts = [copy.deepcopy(state_dict) for _ in range(tp_size)]

    def is_tp(name, tp_names):
        for tp_name in tp_names:
            if tp_name in name:
                return True
        return False

    for name, weight in state_dict.items():
        if is_tp(name, column_tp_names):
            for tp_rank in range(tp_size):
                tp_state_dicts[tp_rank][name] = torch.chunk(weight, tp_size, dim=0)[tp_rank]
        elif is_tp(name, row_tp_names):
            for tp_rank in range(tp_size):
                tp_state_dicts[tp_rank][name] = torch.chunk(weight, tp_size, dim=1)[tp_rank] 

    return tp_state_dicts


def save(state_dicts, save_dir: str, latest_checkpointed_iteration="release"):
    if not os.path.exists(save_dir):
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, choices=["dit", "vae"], default="dit", help="The module to convert")
    parser.add_argument("--source_path", type=str, default="./transformers/mp_rank_00/model_states.pt", help="Source path of checkpoint")
    parser.add_argument("--target_path", type=str, default="./ckpt/opensoraplan_1.5/", help="Save path of MM checkpoint")
    parser.add_argument("--tp_size", type=int, default=1, help="tp size")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    if args.module == "vae":
        state_dict = torch.load(args.source_path, map_location='cpu')
        state_dict = convert_vae(state_dict)
        torch.save(state_dict, args.target_path)
    else:
        state_dict = torch.load(args.source_path, map_location='cpu')
        state_dict = replace_dit_state_dict(state_dict)
        state_dicts = split_by_tp(state_dict, tp_size=args.tp_size)
        save(state_dicts, args.target_path)