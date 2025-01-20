import inspect
from typing import Optional, Union, List
import torch

from mindspeed_mm.tasks.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin

# When using decoder-only models, we must provide a prompt template to instruct the text encoder
# on how to generate the text.
# --------------------------------------------------------------------
PROMPT_TEMPLATE_ENCODE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
) 
PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)  

NEGATIVE_PROMPT = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"

PROMPT_TEMPLATE = {
    "dit-llm-encode": {
        "template": PROMPT_TEMPLATE_ENCODE,
        "crop_start": 36,
    },
    "dit-llm-encode-video": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO,
        "crop_start": 95,
    },
}


class HunyuanVideoPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds"
    ]

    def __init__(self, vae, text_encoder, tokenizer, scheduler, predict_model, config=None):
        self.predict_model = predict_model
        self.vae = vae
        self.text_encoders = text_encoder
        self.tokenizers = tokenizer
        self.scheduler = scheduler

        config = config.to_dict()
        self.generator = torch.Generator().manual_seed(config.get("seed", 42))
        self.frames, self.height, self.width = config.get("input_size", [65, 256, 256])
        self.generate_params_checks(self.frames, self.height, self.width)
        self.prompt_template = PROMPT_TEMPLATE[config.get("prompt_template")] if config.get("prompt_template") is not None else None
        self.prompt_template_video = PROMPT_TEMPLATE[config.get("prompt_template_video")] if config.get("prompt_template_video") is not None else None
        self.use_template = config.get("prompt_template") is not None
        self.guidance_scale = config.get("guidance_scale", 7.5)
        self.guidance_rescale = config.get("guidance_rescale", 0.0)
        self.embedded_guidance_scale = config.get("embedded_guidance_scale", None)

        self.eta = config.get("eta", 0.0)

    @staticmethod
    def generate_params_checks(frames, height, width):
        if height % 16 != 0 or width % 16 != 0 or (frames - 1) % 4 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

    @staticmethod
    def apply_template(text, template):
        if isinstance(text, str):
            return [template.format(text)]
        elif isinstance(text, list) or isinstance(text, tuple):
            return [template.format(one_text) for one_text in text]
        else:
            raise NotImplementedError(f"Not Support text type {type(text)}")       
    
    def prepare_extra_func_kwargs(self, func, kwargs):
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        device: torch.device = "npu",
        data_type: str = "video",
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

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = NEGATIVE_PROMPT
        if not isinstance(negative_prompt, str):
            raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")
        negative_prompt = [negative_prompt.strip()]

        # 3. Encode input prompt
        if self.prompt_template_video is not None:
            crop_start = self.prompt_template_video.get("crop_start", 0)
        elif self.prompt_template is not None:
            crop_start = self.prompt_template.get("crop_start", 0)
        else:
            crop_start = 0

        do_classifier_free_guidance = self.guidance_scale > 1
        
        if data_type == "image":
            prompt_template = self.prompt_template["template"]
        elif data_type == "video":
            prompt_template = self.prompt_template_video["template"]
        else:
            raise NotImplementedError(f"Unsupported data type: {data_type}")
        
        prompt = self.apply_template(prompt, prompt_template)

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
            max_length=self.tokenizers[0].model_max_length + crop_start,
            use_prompt_preprocess=use_prompt_preprocess
        )
        prompt_embeds = prompt_embeds[:, crop_start:]
        prompt_mask = prompt_mask[:, crop_start:]
        prompt_embeds_2, _, negative_prompt_embeds_2, _ = self.encode_texts(
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
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])

        # 4. Prepare latents
        shape = (
            batch_size,
            self.predict_model.in_channels,
            (self.frames - 1) // self.vae.time_compression_ratio + 1,
            self.height // self.vae.spatial_compression_ratio,
            self.width // self.vae.spatial_compression_ratio
        )
        latents = self.prepare_latents(
            shape,
            generator=self.generator,
            device=device,
            dtype=prompt_embeds.dtype
        )

        # 5. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": self.generator, "eta": self.eta}
        )

        # 6. denoise
        latents = self.scheduler.sample(
            model=self.predict_model,
            latents=latents.to(self.predict_model.dtype),
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=self.guidance_scale,
            guidance_rescale=self.guidance_rescale,
            embedded_guidance_scale=self.embedded_guidance_scale,
            model_kwargs={"prompt": [prompt_embeds, prompt_embeds_2], "prompt_mask": prompt_mask},
            extra_step_kwargs=extra_step_kwargs
        )

        if hasattr(self.vae, "shift_factor"):
            latents = latents / self.vae.scaling_factor + self.vae.shift_factor
        else:
            latents = latents / self.vae.scaling_factor

        self.vae.enable_tiling()
        video = self.decode_latents(latents.to(self.vae.dtype))

        return video
