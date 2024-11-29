from typing import List, Dict

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.generation.streamers import TextStreamer

from pretrain_qwen2vl import model_provider
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.generation_mixin import GenerationMixin
from mindspeed_mm.models.text_encoder import Tokenizer


class Qwen2VlPipeline(GenerationMixin, InputsCheckMixin, MMEncoderMixin):

    def __init__(self, infer_config):
        self.infer_config = infer_config
        self.tokenizer = Tokenizer(infer_config.tokenizer).get_tokenizer()

        self.model = model_provider()
        state_dict = torch.load(infer_config.from_pretrained_with_deal, map_location='cpu')
        self.model.load_state_dict(state_dict=state_dict["model"])
        self.model.eval()
        self.model.to(dtype=infer_config.dtype, device=infer_config.device)

        self.image_processor = AutoProcessor.from_pretrained(infer_config.tokenizer.from_pretrained, local_files_only=True)
        self.generation_config = infer_config.generation_config
        self.model_config = infer_config.text_decoder
        self.main_input_name = 'input_ids'
        self.min_pixels = infer_config.min_pixels if hasattr(infer_config, "min_pixels") else None
        self.max_pixels = infer_config.max_pixels if hasattr(infer_config, "max_pixels") else None

    def __call__(self, prompt=None, images=None, return_ids=False):
        if images:
            if isinstance(images, list):
                image = images[0]
            else:
                image = images
        else:
            image = self.infer_config.image_path

        if not prompt:
            prompt = self.infer_config.prompts

        inputs = self.prepare_inputs(prompt=prompt, images=image)

        if return_ids:
            streamer = None
        else:

            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generated_ids = self.generate(**inputs,
                                      do_sample=True if self.generation_config.temperature > 0 else False,
                                      temperature=self.generation_config.temperature,
                                      max_new_tokens=self.generation_config.max_new_tokens,
                                      streamer=streamer)
        if return_ids:
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.image_processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = out[0]
            return response

    def prepare_inputs(self, prompt=None, images=None, messages=None):
        if not messages:
            messages = [[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": images,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]]

        prompt = self.image_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.image_processor(
            text=prompt,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.infer_config.device)
        inputs['pixel_values'] = inputs['pixel_values'].type(torch.bfloat16)
        return inputs

    def prepare_inputs_for_generation(self, **kwargs):
        return kwargs

    def evaluate(self, message):
        messages = [{'role': 'user', 'content': self._prepare_content(message)}]
        inputs = self.prepare_inputs(messages=[messages])

        generated_ids = self.generate(**inputs,
                                      do_sample=True if self.generation_config.temperature > 0 else False,
                                      temperature=self.generation_config.temperature,
                                      max_new_tokens=self.generation_config.max_new_tokens)

        #  把input_ids 截取掉
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.image_processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        return response

    def _prepare_content(self, inputs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': 'file://' + s['value']}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content
