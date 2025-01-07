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
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import mpu
from torch import nn

from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.common.communications import collect_tensors_across_ranks
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.data.data_utils.constants import (
    LATENTS,
    PROMPT,
    VIDEO_MASK,
    PROMPT_MASK,
    MASKED_VIDEO,
    INPUT_MASK
)

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
        self.load_video_features = config.load_video_features
        self.load_text_features = config.load_text_features
        self.enable_encoder_dp = config.enable_encoder_dp if hasattr(config, "enable_encoder_dp") else False
        if self.enable_encoder_dp and mpu.get_pipeline_model_parallel_world_size() > 1:
            raise AssertionError("Encoder DP cannot be used with PP")
        # Track the current index to save or fetch the encoder cache when encoder dp is enabled
        self.cache = {}
        self.index = 0
        if not self.load_video_features:
            self.ae = AEModel(config.ae).eval()
            self.ae.requires_grad_(False)
        if not self.load_text_features:
            self.text_encoder = TextEncoder(config.text_encoder).eval()
            self.text_encoder.requires_grad_(False)

        self.predictor = PredictModel(config.predictor).get_model()
        self.diffusion = DiffusionModel(config.diffusion).get_model()

    def set_input_tensor(self, input_tensor):
        self.predictor.set_input_tensor(input_tensor)

    def forward(self, video, prompt_ids, video_mask=None, prompt_mask=None, **kwargs):
        """
        video: raw video tensors, or ae encoded latent
        prompt_ids: tokenized input_ids, or encoded hidden states
        video_mask: mask for video/image
        prompt_mask: mask for prompt(text)
        """
        with torch.no_grad():
            i2v_results = None
            if video is not None:
                self.index = 0
                # Visual Encode
                if self.load_video_features:
                    latents = video
                else:
                    if self.task == "t2v":
                        latents, _ = self.ae.encode(video)
                    elif self.task == "i2v":
                        latents, i2v_results = self.ae.encode(video, **kwargs)
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

        # Gather the results after encoding of ae and text_encoder
        if self.enable_encoder_dp:
            if self.index == 0:
                self.init_cache(latents, prompt, video_mask, prompt_mask, i2v_results)
            latents, prompt, video_mask, prompt_mask, i2v_results = self.get_feature_from_cache()

        if self.task == "i2v":
            kwargs.update(i2v_results)

        noised_latents, noise, timesteps = self.diffusion.q_sample(latents, model_kwargs=kwargs, mask=video_mask)

        model_output = self.predictor(
            noised_latents,
            timestep=timesteps,
            prompt=prompt,
            video_mask=video_mask,
            prompt_mask=prompt_mask,
            **kwargs,
        )
        return model_output, latents, noised_latents, timesteps, noise, video_mask

    def compute_loss(
        self, model_output, latents, noised_latents, timesteps, noise, video_mask
    ):
        """compute diffusion loss"""
        loss_dict = self.diffusion.training_losses(
            model_output=model_output,
            x_start=latents,
            x_t=noised_latents,
            noise=noise,
            t=timesteps,
            mask=video_mask,
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
        
        missing_keys, unexpected_keys = self.predictor.load_state_dict(state_dict, strict)

        if missing_keys is not None:
            logger.info(f"Missing keys in state_dict: {missing_keys}.")
        if unexpected_keys is not None:
            logger.info(f"Unexpected key(s) in state_dict: {unexpected_keys}.")

    def init_cache(self, latents, prompt, video_mask, prompt_mask, i2v_results=None):
        """Initialize cache in the first step."""
        self.cache = {}
        group = mpu.get_tensor_and_context_parallel_group()
        # gather as list
        self.cache = {
            LATENTS: collect_tensors_across_ranks(latents, group=group),
            PROMPT: collect_tensors_across_ranks(prompt, group=group),
            VIDEO_MASK: collect_tensors_across_ranks(video_mask, group=group),
            PROMPT_MASK: collect_tensors_across_ranks(prompt_mask, group=group)
        }

        if not self.task == "i2v" or not i2v_results:
            return

        for key in [MASKED_VIDEO, INPUT_MASK]:
            if key in i2v_results:
                self.cache[key] = collect_tensors_across_ranks(i2v_results[key], group=group)

    def get_feature_from_cache(self):
        """Get from the cache"""
        latents = self.cache[LATENTS][self.index]
        prompt = self.cache[PROMPT][self.index]
        video_mask = self.cache[VIDEO_MASK][self.index]
        prompt_mask = self.cache[PROMPT_MASK][self.index]

        i2v_results = {}
        if self.task == "i2v":
            i2v_results[MASKED_VIDEO] = self.cache[MASKED_VIDEO][self.index] if MASKED_VIDEO in self.cache else None
            i2v_results[INPUT_MASK] = self.cache[INPUT_MASK][self.index] if INPUT_MASK in self.cache else None

        self.index += 1
        return latents, prompt, video_mask, prompt_mask, i2v_results
