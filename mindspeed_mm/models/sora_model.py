# coding=utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

from logging import getLogger
from typing import Any, Mapping

import torch
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from torch import nn

from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.text_encoder import TextEncoder

logger = getLogger(__name__)


class SoRAModel(nn.Module):
    """
    Instantiate a video generation model from config.
    SoRAModel is an assembled model, which may include text_encoder, video_encoder, predictor, and diffusion model

    Args:
        config (dict): the general config for Multi-Modal Model
        {
            "ae": {...},
            "text_encoder": {...},
            "predictor": {...},
            "diffusion": {...},
            "load_video_features":False,
            ...
        }
    """

    def __init__(self, config):
        super().__init__()
        self.config = core_transformer_config_from_args(get_args())
        self.task = config.task if hasattr(config, "task") else "t2v"

        if mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
            raise NotImplementedError("Not support virtual_pipeline_model_parallel now. ")
        else:
            self.pp_rank = mpu.get_pipeline_model_parallel_rank()

        self.pre_process = mpu.is_pipeline_first_stage()
        self.post_process = mpu.is_pipeline_last_stage()
        self.input_tensor = None
        # to avoid grad all-reduce and reduce-scatter in megatron, since SoRAModel has no embedding layer.
        self.share_embeddings_and_output_weights = False

        if self.pp_rank == 0:
            self.load_video_features = config.load_video_features
            self.load_text_features = config.load_text_features
            if not self.load_video_features:
                print_rank_0(f"init AEModel....")
                self.ae = AEModel(config.ae).eval()
                self.ae.requires_grad_(False)
            if not self.load_text_features:
                print_rank_0(f"init TextEncoder....")
                self.text_encoder = TextEncoder(config.text_encoder).eval()
                self.text_encoder.requires_grad_(False)

        self.diffusion = DiffusionModel(config.diffusion).get_model()
        self.predictor = PredictModel(config.predictor).get_model()

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor
        self.predictor.set_input_tensor(input_tensor)

    def forward(self, video, prompt_ids, video_mask=None, prompt_mask=None, **kwargs):
        """
        video: raw video tensors, or ae encoded latent
        prompt_ids: tokenized input_ids, or encoded hidden states
        video_mask: mask for video/image
        prompt_mask: mask for prompt(text)
        """

        if self.pre_process:
            with torch.no_grad():
                # Visual Encode
                if self.load_video_features:
                    latents = video
                else:
                    if self.task == "t2v":
                        latents, _ = self.ae.encode(video)
                    elif self.task == "i2v":
                        latents, i2v_results = self.ae.encode(video, **kwargs)
                        kwargs.update(i2v_results)
                    else:
                        raise NotImplementedError(f"Task {self.task} if not Implemented!")

                # Text Encode
                if self.load_text_features:
                    prompt = prompt_ids
                else:
                    B, N, L = prompt_ids.shape
                    prompt_ids = prompt_ids.view(-1, L)
                    prompt_mask = prompt_mask.view(-1, L)
                    hidden_states = self.text_encoder.encode(prompt_ids, prompt_mask)
                    prompt = hidden_states["last_hidden_state"].view(B, N, L, -1)

            noised_latents, noise, timesteps = self.diffusion.q_sample(latents, model_kwargs=kwargs, mask=video_mask)
            predictor_input_latent, predictor_timesteps, predictor_prompt = noised_latents, timesteps, prompt
            predictor_video_mask, predictor_prompt_mask = video_mask, prompt_mask
        else:
            if not hasattr(self.predictor, "pipeline_set_prev_stage_tensor"):
                raise ValueError(f"PP has not been implemented for {self.predictor_cls} yet. ")
            predictor_input_list, training_loss_input_list = self.predictor.pipeline_set_prev_stage_tensor(
                self.input_tensor, extra_kwargs=kwargs)
            predictor_input_latent, predictor_timesteps, predictor_prompt, predictor_video_mask, predictor_prompt_mask \
                = predictor_input_list
            latents, noised_latents, timesteps, noise, video_mask = training_loss_input_list

        output = self.predictor(
            predictor_input_latent,
            timestep=predictor_timesteps,
            prompt=predictor_prompt,
            video_mask=predictor_video_mask,
            prompt_mask=predictor_prompt_mask,
            **kwargs,
        )

        if self.post_process:
            loss = self.compute_loss(
                output if isinstance(output, torch.Tensor) else output[0],
                latents,
                noised_latents,
                timesteps,
                noise,
                video_mask,
                **kwargs
            )
            return [loss]

        return self.predictor.pipeline_set_next_stage_tensor(
            input_list=[latents, noised_latents, timesteps, noise, video_mask],
            output_list=output,
            extra_kwargs=kwargs)

    def compute_loss(
        self, model_output, latents, noised_latents, timesteps, noise, video_mask, **kwargs
    ):
        """compute diffusion loss"""
        loss_dict = self.diffusion.training_losses(
            model_output=model_output,
            x_start=latents,
            x_t=noised_latents,
            noise=noise,
            t=timesteps,
            mask=video_mask,
            **kwargs
        )
        return loss_dict
    
    def train(self, mode=True):
        self.predictor.train()

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """Customized state_dict"""
        return self.predictor.state_dict(prefix=prefix, keep_vars=keep_vars)
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """Customized load."""
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

        missing_keys, unexpected_keys = self.predictor.load_state_dict(state_dict, False)

        if missing_keys is not None:
            logger.info(f"Missing keys in state_dict: {missing_keys}.")
        if unexpected_keys is not None:
            logger.info(f"Unexpected key(s) in state_dict: {unexpected_keys}.")
