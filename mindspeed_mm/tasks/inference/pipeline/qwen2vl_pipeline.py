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

        self.image_processor = AutoProcessor.from_pretrained(infer_config.tokenizer.from_pretrained)
        self.generation_config = infer_config.generation_config
        self.model_config = infer_config.text_decoder
        self.main_input_name = 'input_ids'

    def __call__(self):
        inputs = self.prepare_inputs()
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.generate(**inputs,
                      do_sample=True if self.generation_config.temperature > 0 else False,
                      temperature=self.generation_config.temperature,
                      max_new_tokens=self.generation_config.max_new_tokens,
                      streamer=streamer)

    def prepare_inputs(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.infer_config.image_path,
                    },
                    {"type": "text", "text": self.infer_config.prompts},
                ],
            }
        ]

        prompt = self.image_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.image_processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        inputs['pixel_values'] = inputs['pixel_values'].type(torch.bfloat16)
        return inputs

    def prepare_inputs_for_generation(self, **kwargs):
        return kwargs
