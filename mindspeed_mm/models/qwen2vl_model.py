# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Dict, Union
import torch
from torch.nn import CrossEntropyLoss

from megatron.core import InferenceParams, mpu
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.common.module_spec.qwen2vl_layer_spec import get_qwen2vl_layer_spec, get_mlp_module_spec, get_qwen2vlllm_layer_local_spec
from mindspeed_mm.models.vision.vision_model import Qwen2vlVisionModel
from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import Qwen2VLRotaryEmbedding_llm


class Qwen2VLModel(LanguageModule):
    """
    Vision-Language multi-modal model.
    Qwen2VLModel is an assembled model, which include image_encoder, text_decoder model.

    Args:
        mm_config (dict): the general config for VLModel, model.json中的配置
        {
            "pre_process": (bool),  # Include the embedding leayer in the gpt decoder (used with pipeline parallelism).
            "post_process": (bool),  # Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism).
            "add_text_encoder": (bool),  # Whether to construct the text encoder. not used now. 
            "add_image_encoder": (bool),  # Whether to construct the image encoder.
            "add_video_encoder": (bool),  # Whether to construct the video encoder. not used now.
            "add_text_decoder": (bool),  # Whether to construct the text decoder.
            "img_context_token_id": (int),  # Index in the language_embeddings tensor where image_embeddings should be inserted.
            "text_encoder": {...},  # Config for the text encoder. not used now.
            "image_encoder": {...},  # Config for the image encoder.
            "video_encoder": {...},  # Config for the video encoder. not used now.
            "text_decoder": {...},  # Config for the text decoder.
        }
    """

    def __init__(self, mm_config) -> None:
        super().__init__(config=mm_config)

        self.config = core_transformer_config_from_args(get_args())
        self.pre_process: bool = mm_config.pre_process
        self.post_process: bool = mm_config.post_process
        self.add_text_encoder = mm_config.text_encoder is not None
        self.add_image_encoder = mm_config.image_encoder is not None
        self.add_video_encoder = mm_config.video_encoder is not None
        self.add_text_decoder = mm_config.text_decoder is not None
        self.share_embeddings_and_output_weights = False
        self.position_embedding_type = mm_config.text_decoder.position_embedding_type
        self.img_context_token_id = mm_config.img_context_token_id

        # initialize pipeline parallel configs
        self.pp_size = mpu.get_pipeline_model_parallel_world_size()
        if mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
            raise NotImplementedError("Not support virtual_pipeline_model_parallel now")
        else:
            self.pp_rank = mpu.get_pipeline_model_parallel_rank()
        
        if self.add_text_decoder:
            self.decoder, self.rotary_pos_emb = self._build_text_decoder_model(mm_config.text_decoder)

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=mm_config.text_decoder,
                vocab_size=mm_config.text_decoder.padded_vocab_size,
                max_sequence_length=mm_config.text_decoder.max_position_embeddings,
                position_embedding_type=mm_config.text_decoder.position_embedding_type,
            )

        if self.post_process:
            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs stored
                # in gradient buffer to calculate the weight gradients for the embedding final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                mm_config.text_decoder.hidden_size,
                mm_config.text_decoder.padded_vocab_size,
                config=mm_config.text_decoder,
                init_method=mm_config.text_decoder.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not mm_config.text_decoder.parallel_output,
                skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

        if self.add_image_encoder:
            if self.pp_size > 1:
                mm_config.image_encoder.vision_encoder.num_layers *= self.pp_size
            self.image_encoder = Qwen2vlVisionModel(
                config=mm_config.image_encoder,
                encoder_transformer_layer_spec=get_qwen2vl_layer_spec(is_vit=True),
                projector_layer_spec=get_mlp_module_spec(use_te=False).submodules,
            )
            if self.pp_size > 1:
                mm_config.image_encoder.vision_encoder.num_layers //= self.pp_size

    def _build_text_decoder_model(self, config) -> TransformerBlock:
        if self.pp_size <= 1:
            if config.position_embedding_type == 'rope':
                # rope Parameters should configured by json
                config.rope_theta = 1000000.0
                config.rope_scaling = {
                    "rope_type": "default",
                    "type": "default",
                    "mrope_section": [16, 24, 24]
                }
                rotary_pos_emb = Qwen2VLRotaryEmbedding_llm(config=config)
            else:
                raise ValueError(f'only support rope now !')
            
            decoder = TransformerBlock(
                config=config,
                spec=get_qwen2vlllm_layer_local_spec(),
                pre_process=self.pre_process,
                post_process=self.post_process,
            )
            return decoder, rotary_pos_emb
        
        if self.pp_size != len(config.num_layer_list):
            raise ValueError(f"length of num_layer_list must equal to pipeline-model-parallel-size, "
                             f"but got num_layer_list length:{len(config.num_layer_list)} "
                             f"and pipeline-model-parallel-size:{self.pp_size}.")

        if not mpu.is_pipeline_first_stage():
            self.pre_process = False
            self.add_text_encoder = False
            self.add_image_encoder = False
            self.add_video_encoder = False

        if not mpu.is_pipeline_last_stage():
            self.post_process = False

        pipeline_start_index = config.num_layer_list[self.pp_rank]
        if mpu.is_pipeline_last_stage():
            pipeline_end_index = config.num_layers
        else:
            pipeline_end_index = config.num_layer_list[self.pp_rank + 1]

        if pipeline_end_index < pipeline_start_index:
            raise ValueError(f"each index in num_layer_list must equal or large than last, "
                             f"but got {pipeline_end_index} and {pipeline_start_index}.")
        if pipeline_end_index - pipeline_start_index > 0:
            config.num_layers = pipeline_end_index - pipeline_start_index
        else:
            return None, None

        if config.position_embedding_type == 'rope':
            # rope Parameters should configured by json
            config.rope_theta = 1000000.0
            config.rope_scaling = {
                "rope_type": "default",
                "type": "default",
                "mrope_section": [16, 24, 24]
            }
            rotary_pos_emb = Qwen2VLRotaryEmbedding_llm(config=config)
        else:
            raise ValueError(f'only support rope now !')

        # GPTModel will divide num_layers by pp_size
        config.num_layers *= self.pp_size

        decoder = TransformerBlock(
            config=config,
            spec=get_qwen2vlllm_layer_local_spec(),
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        config.num_layers //= self.pp_size
        return decoder, rotary_pos_emb

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        if self.decoder is not None:
            self.decoder.set_input_tensor(input_tensor[0])

    def freeze(
            self,
            freeze_text_decoder: bool = False,
            freeze_image_encoder: bool = False,
            freeze_image_projection: bool = False,
    ):
        """
        Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_text_decoder (bool): Freeze the text decoder module.
            freeze_image_encoder (bool): Freeze the image encoder module.
            freeze_image_projection (bool): Freeze the image projector module.
        """
        if freeze_text_decoder and self.decoder is not None:
            for param in self.decoder.parameters():
                param.requires_grad = False
            if self.embedding is not None:
                for param in self.embedding.parameters():
                    param.requires_grad = False
            if self.output_layer is not None:
                for param in self.output_layer.parameters():
                    param.requires_grad = False

        self.image_encoder.freeze(freeze_image_encoder, freeze_image_projection)

    def compute_loss(self, logits, labels, ignore_flag=False):
        # shift tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, 152064)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        if ignore_flag:
            loss = loss * 0.0

        return loss
    
    def _build_causal_mask(self, inputs_embeddings, attention_mask):
        past_seen_token = 0
        cache_position = torch.arange(
            past_seen_token, past_seen_token + inputs_embeddings.shape[1], device=inputs_embeddings.device)
        dtype, device = inputs_embeddings.dtype, inputs_embeddings.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = inputs_embeddings.shape[1]

        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.tensor)
            else past_seen_token + sequence_length + 1
        ) 
        batch_size = inputs_embeddings.shape[0]

        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        else:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
            return causal_mask

    def forward(
            self,
            input_ids: torch.Tensor,
            pixel_values: torch.Tensor,
            image_grid_thw: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            inference_params: Optional[InferenceParams] = None,
            decoder_input: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            packed_seq_params: Optional[PackedSeqParams] = None,
            extra_block_kwargs: Optional[dict] = None,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward function of the VLModel.

        Args:
            extra_block_kwargs:
            packed_seq_params:
            decoder_input:
            image_grid_thw:
            pixel_values:
            input_ids (torch.Tensor): Input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): Input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Attention mask for the text decoder model [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference parameter for the forward method of GPTModel.
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        """
        if self.add_image_encoder:
            if pixel_values is not None and image_grid_thw is not None:
                image_embeddings = self.image_encoder(pixel_values, image_grid_thw)
            else:
                raise ValueError(f'must input pixel_values and image_grid_thw')

        if self.add_text_decoder:
            inputs_embeddings = None
            if self.pre_process:
                inputs_embeddings = self.embedding(input_ids=input_ids, position_ids=position_ids)
                if self.add_image_encoder:
                    inputs_embeddings = inputs_embeddings.transpose(0, 1)
                    image_mask = torch.eq(input_ids, self.img_context_token_id).unsqueeze(-1).expand_as(inputs_embeddings)
                    image_embeddings = image_embeddings.to(inputs_embeddings.device, inputs_embeddings.dtype)
                    inputs_embeddings = inputs_embeddings.masked_scatter(image_mask, image_embeddings)
                    inputs_embeddings = inputs_embeddings.transpose(0, 1).clone()

            if self.decoder is not None:
                from megatron.core import parallel_state
                past_seen_tokens = 0
                seq_len = input_ids.shape[1]
                if self.config.sequence_parallel:
                    seq_len *= parallel_state.get_tensor_model_parallel_world_size()
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + seq_len, device=input_ids.device
                )
                position_ids = cache_position.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
                x_dtype = torch.bfloat16

                rotary_pos_emb = self.rotary_pos_emb(input_ids.device, x_dtype, position_ids)

                hidden_states = self.decoder(
                    hidden_states=inputs_embeddings,
                    attention_mask=attention_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb,
                    packed_seq_params=packed_seq_params,
                    **(extra_block_kwargs or {}),
                )
            else:
                return inputs_embeddings

            if self.post_process:
                logits, _ = self.output_layer(hidden_states)
                logits = logits.transpose(0, 1).contiguous().float()
                return {
                    "loss": None if labels is None else self.compute_loss(logits, labels),
                    "logits": logits
                }
            else:
                return hidden_states
