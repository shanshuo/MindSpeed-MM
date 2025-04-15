from typing import Optional, Union, List

import torch
from megatron.training import get_args

from mindspeed_mm.tasks.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin


POSITIVE_MAGIC = "超高清、HDR 视频、环境光、杜比全景声、画面稳定、流畅动作、逼真的细节、专业级构图、超现实主义、自然、生动、超细节、清晰。"
NEGATIVE_MAGIC = "画面暗、低分辨率、不良手、文本、缺少手指、多余的手指、裁剪、低质量、颗粒状、签名、水印、用户名、模糊。"


class StepVideoPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):
    def __init__(self, vae, text_encoder, tokenizer, scheduler, predict_model, config=None):
        self.vae = vae
        self.text_encoders = text_encoder
        self.tokenizers = tokenizer
        self.scheduler = scheduler
        self.predict_model = predict_model

        config = config.to_dict()
        self.guidance_scale = config.get("guidance_scale", 7.5)
        self.num_frames, self.height, self.width = config.get("input_size", [204, 768, 768])
        self.generator = torch.Generator().manual_seed(config.get("seed", 42))

    @staticmethod
    def apply_template(text, template):
        if isinstance(text, str):
            return [template.format(text)]
        elif isinstance(text, list) or isinstance(text, tuple):
            return [template.format(one_text) for one_text in text]
        else:
            raise NotImplementedError(f"Not Support text type {type(text)}") 

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        device: torch.device = "npu",
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
        use_prompt_preprocess: Optional[bool] = False,
        **kwargs
    ):
        """
        The call function to the pipeline for generation

        Inputs:
            prompt (`str` or `List[str]`): 
                The prompt or prompts to guide video/image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in video/image generation.
                Ignored when not using guidance (`guidance_scale < 1`)
        Returns:
            video (`torch.Tensor` or `List[torch.Tensor]`) 
        """
        args = get_args()

        # 1. Check inputs
        self.text_prompt_checks(
            prompt,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds
        )

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        do_classifier_free_guidance = self.guidance_scale > 1
        
        # 3. Encode input prompt
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = NEGATIVE_MAGIC
        if not isinstance(negative_prompt, str):
            raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")
        
        if isinstance(prompt, str):
            prompt = [prompt + POSITIVE_MAGIC]
        elif isinstance(prompt, list) or isinstance(prompt, tuple):
            prompt = [one_text + POSITIVE_MAGIC for one_text in prompt]
        else:
            raise NotImplementedError(f"Not Support text type {type(prompt)}") 

        # Text Encoder load to device
        self.text_encoders.to(device)

        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self.encode_texts(
            prompt,
            device,
            tokenizer=self.tokenizers[0],
            text_encoder=self.text_encoders[0],
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
            max_length=self.tokenizers[0].model_max_length,
            use_prompt_preprocess=use_prompt_preprocess
        )
        clip_embedding, clip_mask, negative_clip_embedding, negative_clip_mask = self.encode_texts(
            prompt,
            device,
            tokenizer=self.tokenizers[1],
            text_encoder=self.text_encoders[1],
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            clip_skip=clip_skip,
            max_length=self.tokenizers[1].model_max_length,
            use_prompt_preprocess=use_prompt_preprocess
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])# 2, s1, d1
            clip_embedding = torch.cat([negative_clip_embedding, clip_embedding, ]).to(prompt_embeds.dtype)# 2, s2, d2 
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
                clip_mask = torch.cat([negative_clip_mask, clip_mask])

        # Text Encoder offload to `cpu`
        self.text_encoders.to("cpu")
        torch.cuda.empty_cache()

        # 4. Prepare latents
        latent_channels = self.predict_model.in_channels
        shape = (
            batch_size,
            self.num_frames // self.vae.frame_len * self.vae.latent_len,
            latent_channels,
            int(self.height) // 16,
            int(self.width) // 16,
        )
        latents = self.prepare_latents(
            shape,
            generator=self.generator,
            device=device,
            dtype=prompt_embeds.dtype
        )

        # 5. Denoising
        prompt_embeds = prompt_embeds.unsqueeze(1)# b s d -> b 1 s d
        clip_embedding = clip_embedding.unsqueeze(1)# b s d -> b 1 s d
        self.predict_model.to(device)
        latents = self.scheduler.sample(
            model=self.predict_model,
            latents=latents.to(self.predict_model.dtype),
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=self.guidance_scale,
            model_kwargs={"prompt": [prompt_embeds, clip_embedding], "prompt_mask": [prompt_mask, clip_mask]}
        )

        # 6. Decode
        video = self.decode_latents(latents.to(self.vae.dtype))# b t c h w
        
        return video