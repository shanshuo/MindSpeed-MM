import argparse
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Dict, List

import mindspeed.megatron_adaptor  # noqa
import torch
from safetensors.torch import load_file as safe_load
from checkpoint.utils import copy_files_except_suffix, load_from_hf, save_by_index_json

DIT_CONVERSION_MAPPING = {
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.linear_1.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.linear_1.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.linear_2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.linear_2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "condition_embedder.image_embedder.ff.net.0.proj.weight": "img_emb.proj.1.weight",
    "condition_embedder.image_embedder.ff.net.0.proj.bias": "img_emb.proj.1.bias",
    "condition_embedder.image_embedder.ff.net.2.weight": "img_emb.proj.3.weight",
    "condition_embedder.image_embedder.ff.net.2.bias": "img_emb.proj.3.bias",
    "condition_embedder.image_embedder.norm1.weight": "img_emb.proj.0.weight",
    "condition_embedder.image_embedder.norm1.bias": "img_emb.proj.0.bias",
    "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight",
    "condition_embedder.image_embedder.norm2.bias": "img_emb.proj.4.bias",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight",
}


class CheckpointConverter:
    def __init__(self, source_path: str, ckpt_path: str, target_path: str, mode: str):
        self.source_path = source_path
        self.ckpt_path = ckpt_path
        self.target_path = target_path
        self.mode = mode
        self.state_dict = None

    def load_weight(self, _weight_path):
        if _weight_path.endswith(".safetensors"):
            return safe_load(_weight_path)
        else:
            return torch.load(_weight_path, map_location="cpu")

    def load_from_mm(self, _load_dir: str) -> list[dict]:
        LATEST_TXT = "latest_checkpointed_iteration.txt"
        mm_save_dir = Path(_load_dir)
        save_iteration = mm_save_dir.joinpath(LATEST_TXT).read_text()
        save_dir = mm_save_dir.joinpath(
            f"iter_{int(save_iteration):07}"
            if save_iteration != "release"
            else save_iteration
        )

        current_path = save_dir.joinpath("mp_rank_00")
        pt_path = current_path.joinpath("model_optim_rng.pt")

        state_dicts = torch.load(pt_path, map_location="cpu")["model"]

        return state_dicts

    def replace_state_dict(
        self,
        state_dict: Dict[str, Any],
        conversion_mapping: Dict,
    ) -> Dict[str, Any]:
        for ori_key, mm_key in conversion_mapping.items():
            if self.mode == "convert_to_hf" and mm_key in state_dict.keys():
                state_dict[ori_key] = state_dict.pop(mm_key)
            elif self.mode == "convert_to_mm" and ori_key in state_dict.keys():
                state_dict[mm_key] = state_dict.pop(ori_key)
        return state_dict

    def convert_attn_to_mm(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        new_checkpoint = {}
        state_dict = state_dict.get("blocks", state_dict)

        for key, value in state_dict.items():
            if "norm_added_q" in key:  # keys to ignore
                continue
            new_key = key.replace("attn1.norm_q", "self_attn.q_norm")
            new_key = new_key.replace("attn1.norm_k", "self_attn.k_norm")
            new_key = new_key.replace("attn2.norm_q", "cross_attn.q_norm")
            new_key = new_key.replace("attn2.norm_k", "cross_attn.k_norm")
            new_key = new_key.replace("attn1.to_q.", "self_attn.proj_q.")
            new_key = new_key.replace("attn1.to_k.", "self_attn.proj_k.")
            new_key = new_key.replace("attn1.to_v.", "self_attn.proj_v.")
            new_key = new_key.replace("attn1.to_out.0.", "self_attn.proj_out.")
            new_key = new_key.replace("attn2.to_q.", "cross_attn.proj_q.")
            new_key = new_key.replace("attn2.to_k.", "cross_attn.proj_k.")
            new_key = new_key.replace("attn2.to_v.", "cross_attn.proj_v.")
            new_key = new_key.replace("attn2.add_k_proj", "cross_attn.k_img")
            new_key = new_key.replace("attn2.add_v_proj", "cross_attn.v_img")
            new_key = new_key.replace("attn2.norm_added_k", "cross_attn.k_norm_img")
            new_key = new_key.replace("attn2.to_out.0.", "cross_attn.proj_out.")
            new_key = new_key.replace(".ffn.net.0.proj.", ".ffn.0.")
            new_key = new_key.replace(".ffn.net.2.", ".ffn.2.")
            new_key = new_key.replace("scale_shift_table", "modulation")
            new_key = new_key.replace(".norm2.", ".norm3.")

            new_checkpoint[new_key] = value

        return new_checkpoint

    def convert_attn_to_hf(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        new_checkpoint = {}
        state_dict = state_dict.get("blocks", state_dict)

        for key, value in state_dict.items():
            new_key = key.replace("self_attn.q_norm.", "attn1.norm_q.")
            new_key = new_key.replace("self_attn.k_norm.", "attn1.norm_k.")
            new_key = new_key.replace("cross_attn.q_norm.", "attn2.norm_q.")
            new_key = new_key.replace("cross_attn.k_norm.", "attn2.norm_k.")
            new_key = new_key.replace("self_attn.proj_q.", "attn1.to_q.")
            new_key = new_key.replace("self_attn.proj_k.", "attn1.to_k.")
            new_key = new_key.replace("self_attn.proj_v.", "attn1.to_v.")
            new_key = new_key.replace("self_attn.proj_out.", "attn1.to_out.0.")
            new_key = new_key.replace("cross_attn.proj_q.", "attn2.to_q.")
            new_key = new_key.replace("cross_attn.proj_k.", "attn2.to_k.")
            new_key = new_key.replace("cross_attn.proj_v.", "attn2.to_v.")
            new_key = new_key.replace("cross_attn.k_img.", "attn2.add_k_proj.")
            new_key = new_key.replace("cross_attn.v_img.", "attn2.add_v_proj.")
            new_key = new_key.replace("cross_attn.k_norm_img.", "attn2.norm_added_k.")
            new_key = new_key.replace("cross_attn.proj_out.", "attn2.to_out.0.")
            new_key = new_key.replace(".ffn.0.", ".ffn.net.0.proj.")
            new_key = new_key.replace(".ffn.2.", ".ffn.net.2.")
            new_key = new_key.replace("modulation", "scale_shift_table")
            new_key = new_key.replace(".norm3.", ".norm2.")

            new_checkpoint[new_key] = value

        return new_checkpoint

    def split_by_index_json(
        self, state_dict: Dict[str, Any], _model_path: str
    ) -> list[dict]:
        index_json_path = os.path.join(
            _model_path, "diffusion_pytorch_model.safetensors.index.json"
        )
        return_dicts = []
        with open(index_json_path, "r", encoding="utf-8") as file:
            weight_map = json.load(file)["weight_map"]
        for key, value in weight_map.items():
            if "norm_added_q" in key:  # keys to ignore
                continue
            index = int(value.split("-")[1])
            while index > len(return_dicts):
                return_dicts.append({})
            return_dicts[index - 1][key] = state_dict[key]
        return return_dicts

    def save_to_mm(
        self,
        state_dicts: List[Dict],
        _save_dir: str,
        latest_checkpointed_iteration="release",
    ):
        if not os.path.exists(_save_dir):
            os.makedirs(_save_dir)

        flags = os.O_WRONLY | os.O_CREAT
        mode = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(
            os.open(
                os.path.join(_save_dir, "latest_checkpointed_iteration.txt"),
                flags,
                mode,
            ),
            "w",
        ) as fout:
            fout.write(latest_checkpointed_iteration)
        if latest_checkpointed_iteration == "release":
            directory = "release"
        else:
            directory = "iter_{:07d}".format(latest_checkpointed_iteration)

        os.makedirs(os.path.join(_save_dir, directory, f"mp_rank_00"))
        save_path = os.path.join(
            _save_dir, directory, f"mp_rank_00", "model_optim_rng.pt"
        )
        save_dict = {}
        save_dict["model"] = state_dicts
        torch.save(save_dict, save_path)

    def forward(self):
        if self.mode == "convert_to_hf":
            self.state_dict = self.load_from_mm(self.source_path)
            self.state_dict = self.replace_state_dict(
                self.state_dict, conversion_mapping=DIT_CONVERSION_MAPPING
            )
            self.state_dict = self.convert_attn_to_hf(self.state_dict)
            self.state_dict = self.split_by_index_json(self.state_dict, self.ckpt_path)
            copy_files_except_suffix(Path(self.ckpt_path), Path(self.target_path))
            save_by_index_json(self.state_dict, self.target_path)
            print("Checkpoint successfully converted from MM to Hugging Face format.")
        elif self.mode == "convert_to_mm":
            self.state_dict = load_from_hf(Path(self.source_path))
            self.state_dict = self.replace_state_dict(
                self.state_dict, conversion_mapping=DIT_CONVERSION_MAPPING
            )
            self.state_dict = self.convert_attn_to_mm(self.state_dict)
            self.save_to_mm(self.state_dict, self.target_path)
            print("Checkpoint successfully converted from Hugging Face to MM format.")

        else:
            raise ValueError(
                "please select the mode only from convert_to_hf or convert_to_mm"
            )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path",
        type=str,
        default="./ckpt/wan2.1/iter_####",
        help="Source path of trained model for mm to hf, or path of weights for transformer",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./Wan2.1-T2V-14B-Diffusers/transformer",
        help="Checkpoint path of original model, only used for mm to hf mode",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="./save_ckpt/wan2.1/",
        help="Save path of MM checkpoint for mm to hf mode, or output path for tranformed weight from hf to mm",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["convert_to_hf", "convert_to_mm"],
        required=True,
        help="Selection of conversion mode: convert_to_hf or convert_to_mm",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    converter = CheckpointConverter(
        source_path=args.source_path,
        ckpt_path=args.ckpt_path,
        target_path=args.target_path,
        mode=args.mode,
    )
    converter.forward()
