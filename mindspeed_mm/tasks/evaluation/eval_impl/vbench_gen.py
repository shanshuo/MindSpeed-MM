# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import json
import os
from typing import Optional

import imageio
import pandas as pd
import torch
import vbench2_beta_i2v
from huggingface_hub import hf_hub_download
from megatron.core import mpu
from megatron.training.utils import print_rank_0
from peft.config import PeftConfigMixin
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.utils import CONFIG_NAME
from torch.nn.parallel.distributed import DistributedDataParallel as ddp
from transformers.utils import PushToHubMixin
from vbench import VBench
from vbench.third_party.RAFT.core.raft import RAFT
from vbench2_beta_i2v import VBenchI2V
from vbench2_beta_long import VBenchLong
from vbench2_beta_long.static_filter import StaticFilter

from mindspeed_mm.data import build_mm_dataloader
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.tasks.evaluation.eval_impl.base_gen import BaseGenEvalImpl
from mindspeed_mm.tasks.inference.pipeline.utils.sora_utils import safe_load_image
from mindspeed_mm.utils.utils import get_dtype, get_device

RATIO = ["1-1", "8-5", "7-4", "16-9"]


class VbenchGenEvalImpl(BaseGenEvalImpl):
    def __init__(self, dataset, inference_pipeline, args):
        super().__init__(dataset, inference_pipeline, args)
        self.eval_type = args.eval_config.eval_type
        self.result_output_path = args.eval_config.eval_result_path
        self.mode = "vbench_standard" if args.eval_config.eval_type in ["i2v", "t2v"] else "long_vbench_standard"
        self.dimensions = args.eval_config.dimensions
        self.videos_path = args.save_path
        self.load_ckpt_from_local = args.eval_config.load_ckpt_from_local
        self.full_json_dir = args.eval_config.dataset.basic_param.data_path
        self.prompt = args.eval_config.prompt if hasattr(args.eval_config, "prompt") else []
        self.ratio = args.eval_config.dataset.extra_param.ratio if hasattr(args.eval_config.dataset.extra_param, "ratio") else "16-9"
        self.vbench = None
        self.image_path = args.eval_config.image_path if hasattr(args.eval_config, "image_path") else None
        self.slow_fast_eval_config = args.eval_config.slow_fast_eval_config if hasattr(args.eval_config,
                                                                                       "slow_fast_eval_config") else None
        self.pipeline = inference_pipeline
        self.eval_args = args
        self.dataset = dataset

    def __call__(self):
        self.inference_video()
        self.analyze_result()

    def check_dimension_list(self):
        dimension_list = self.vbench.build_full_dimension_list()

        if not self.dimensions:
            self.dimensions = dimension_list

        if not set(self.dimensions).issubset(set(dimension_list)):
            raise NotImplementedError("Support dimensions contains:{}".format(dimension_list))

    def save_result_to_excel(self, data):
        excel_res_path = os.path.join(self.result_output_path, "eval_result.xlsx")
        with pd.ExcelWriter(excel_res_path) as writer:
            for key, value in data.items():
                score = value[0]
                rows = []
                result_list = value[1] if key != "camera_motion" else value[2]
                for item in result_list:
                    # split prompt from file path
                    prompt = os.path.splitext(os.path.basename(item["video_path"]))[0][:-2]
                    row = {
                        "total score": score,
                        "prompt": prompt,
                        "video_results": item["video_results"]
                    }
                    rows.append(row)
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=key, index=False)
        print_rank_0(f"Save excel to {excel_res_path}.")

    def analyze_result(self):
        device = torch.device("npu")
        result_file_name = f'{self.eval_type}'

        if "i2v_background" in self.dimensions or (self.mode == "long_vbench_standard" and "background_consistency"
                                                   in self.dimensions):
            PeftConfigMixin.from_pretrained = PatchPeftConfigMixin.from_pretrained

        if self.eval_type == "t2v":
            self.vbench = VBench(device, self.full_json_dir, self.result_output_path)
            self.check_dimension_list()

            self.vbench.evaluate(
                videos_path=self.videos_path,
                name=result_file_name,
                prompt_list=self.prompt,
                dimension_list=self.dimensions,
                local=self.load_ckpt_from_local,
                read_frame=False,
                mode=self.mode
            )
        elif self.eval_type == "i2v":
            vbench2_beta_i2v.utils.load_i2v_dimension_info = self.load_i2v_dimension_info
            self.vbench = VBenchI2V(device, self.full_json_dir, self.result_output_path)
            self.check_dimension_list()
            if self.ratio not in RATIO:
                raise ValueError(f"Not support this ratio {self.ratio}")

            self.vbench.evaluate(
                videos_path=self.videos_path,
                name=result_file_name,
                dimension_list=self.dimensions,
                resolution=self.ratio,
                mode=self.mode
            )
        elif self.eval_type == "long":
            self.vbench = VBenchLong(device, self.full_json_dir, self.result_output_path)
            StaticFilter.load_model = patch_static_filter_load_model
            self.check_dimension_list()

            # 额外参数
            kwargs = {"sb_clip2clip_feat_extractor": 'dinov2', "bg_clip2clip_feat_extractor": "dreamsim",
                      "clip_length_config": "clip_length_mix.yaml", "w_inclip": 1.0, "w_clip2clip": 0.0,
                      "use_semantic_splitting": False, "slow_fast_eval_config": self.slow_fast_eval_config,
                      "dev_flag": True, "num_of_samples_per_prompt": 5, "static_filter_flag": False}
            self.vbench.evaluate(
                videos_path=self.videos_path,
                name=result_file_name,
                prompt_list=self.prompt,
                dimension_list=self.dimensions,
                local=self.load_ckpt_from_local,
                read_frame=False,
                mode="long_vbench_standard",
                **kwargs
            )
        else:
            raise NotImplementedError("Not support evaluate type.")

        result_path = os.path.join(self.result_output_path, result_file_name + '_eval_results.json')
        with open(result_path, 'r', encoding='utf-8') as f:
            self.save_result_to_excel(json.load(f))

        print_rank_0("Evaluation Done.")

    def load_i2v_dimension_info(self, json_dir, dimension, lang, resolution):
        video_pair_list = []
        prompt_dict_ls = []
        with open(json_dir, 'r', encoding='utf-8') as f:
            full_prompt_list = json.load(f)

        if not self.image_path:
            raise ValueError("please set image_path in config.")
        image_root = os.path.join(self.image_path, resolution)
        for prompt_dict in full_prompt_list:
            if dimension in prompt_dict['dimension'] and 'video_list' in prompt_dict:
                prompt = prompt_dict[f'prompt_{lang}']
                cur_video_list = prompt_dict['video_list'] if isinstance(prompt_dict['video_list'], list) else [
                    prompt_dict['video_list']]
                # create image-video pair
                if "image_name" in prompt_dict:
                    image_path = os.path.join(image_root, prompt_dict["image_name"])
                elif "custom_image_path" in prompt_dict:
                    image_path = prompt_dict["custom_image_path"]
                else:
                    raise Exception("prompt_dict doesn't contain 'image_name' or 'custom_image_path' key")

                cur_video_pair = [(image_path, video) for video in cur_video_list]
                video_pair_list += cur_video_pair
                if 'auxiliary_info' in prompt_dict and dimension in prompt_dict['auxiliary_info']:
                    prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list,
                                        'auxiliary_info': prompt_dict['auxiliary_info'][dimension]}]
                else:
                    prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list}]
        return video_pair_list, prompt_dict_ls

    def inference_video(self):
        args = self.eval_args
        # prepare arguments
        torch.set_grad_enabled(False)
        dtype = get_dtype(args.dtype)
        device = get_device(args.device)

        # prepare datasets
        eval_dataloader = build_mm_dataloader(
            self.dataset,
            args.eval_config.dataloader_param,
            process_group=mpu.get_data_parallel_group(),
            dataset_param=args.eval_config.dataset,
        )
        data_iterator, _, _ = build_iterations(train_dl=eval_dataloader, iterator_type="single")

        # prepare pipeline
        save_fps = args.fps // args.frame_interval
        mask_type = args.mask_type if hasattr(args, "mask_type") else None
        crop_for_hw = args.crop_for_hw if hasattr(args, "crop_for_hw") else None
        max_hxw = args.max_hxw if hasattr(args, "max_hxw") else None

        image = None
        image_path = None
        for item in data_iterator:
            caption = item["caption"]
            prefix = item["prefix"]
            if self.eval_type == "i2v":
                image_path = item["image"]
                image = safe_load_image(image_path[0].strip())

            kwargs = {}
            if args.pipeline_class == "OpenSoraPlanPipeline":
                kwargs.update({"conditional_pixel_values_path": image_path,
                            "mask_type": mask_type,
                            "crop_for_hw": crop_for_hw,
                            "max_hxw": max_hxw})

            print(f"*** generator video now, eval_type: {self.eval_type}, prompt: {caption}, prefix: {prefix}, image_path: {image_path}")
            videos = self.pipeline(prompt=caption,
                                image=image,
                                fps=save_fps,
                                use_prompt_preprocess=args.use_prompt_preprocess,
                                device=device,
                                dtype=dtype,
                                **kwargs
                                )
            self.save_eval_videos(videos, args.save_path, save_fps, prefix)

    def save_eval_videos(self, videos, save_path, fps, save_names):
        os.makedirs(save_path, exist_ok=True)
        if isinstance(videos, (list, tuple)) or videos.ndim == 5:  # [b, t, h, w, c]
            for i, video in enumerate(videos):
                save_path_i = os.path.join(save_path, f"{save_names[0]}.mp4")
                imageio.mimwrite(save_path_i, video, fps=fps, quality=6)
        elif videos.ndim == 4:
            save_path = os.path.join(save_path, f"{save_names[0]}.mp4")
            imageio.mimwrite(save_path, videos, fps=fps, quality=6)
        else:
            raise ValueError("The video must be in either [b, t, h, w, c] or [t, h, w, c] format.")


def patch_static_filter_load_model(self):
    self.model = ddp(RAFT(self.args).to(self.device))
    self.model.load_state_dict(torch.load(self.args.model))

    self.model = self.model.module
    self.model.eval()


class PatchPeftConfigMixin(PushToHubMixin):
    @classmethod
    def _split_kwargs(cls, kwargs):
        hf_hub_download_kwargs = {}
        class_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value
            elif key in list(cls.__annotations__):
                class_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, class_kwargs, other_kwargs

    @classmethod
    def from_json_file(cls, path_json_file: str):
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: Optional[str] = None, **kwargs):
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs, class_kwargs, _ = cls._split_kwargs(kwargs)

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                print_rank_0(f"Warning: download {CONFIG_NAME}")
                config_file = hf_hub_download(
                    pretrained_model_name_or_path, CONFIG_NAME, subfolder=subfolder, local_files_only=True,
                    **hf_hub_download_kwargs
                )
            except Exception as e:
                raise ValueError(f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'") from e

        loaded_attributes = cls.from_json_file(config_file)

        if "peft_type" in loaded_attributes:
            peft_type = loaded_attributes["peft_type"]
            config_cls = PEFT_TYPE_TO_CONFIG_MAPPING.get(peft_type, cls)
        else:
            config_cls = cls

        kwargs = {**class_kwargs, **loaded_attributes}

        kwargs.pop('layer_replication', None)
        kwargs.pop('use_dora', None)
        kwargs.pop('use_rslora', None)
        config = config_cls(**kwargs)
        return config