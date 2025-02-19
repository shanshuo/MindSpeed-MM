# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import List, Optional
import torch
from torch import Tensor
from megatron.training import get_args
from megatron.core import InferenceParams, mpu


def _build_attentionmask_positionid_qwenllm(config, input_ids, attention_mask, inference_params, position_ids, *args, **kwargs):
    seq_len = input_ids.shape[1]
    if config.sequence_parallel:
        seq_len *= mpu.get_tensor_model_parallel_world_size()
    if inference_params is not None:
        past_seen_tokens = attention_mask.shape[1] - 1 if inference_params.key_value_memory_dict else 0
    else:
        past_seen_tokens = 0
    if position_ids is None:
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_len, device=input_ids.device
        )
        position_ids = cache_position.view(1, 1, -1).expand(3, input_ids.shape[0], -1)

    if get_args().use_flash_attn:
        return attention_mask, position_ids
    
    seq_len = input_ids.shape[1]
    past_seen_token = 0
    cache_position = torch.arange(
        past_seen_token, past_seen_token + seq_len, device=input_ids.device)
    dtype, device = torch.bfloat16, input_ids.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_ids.shape[1]

    target_length = (
        attention_mask.shape[-1]
        if isinstance(attention_mask, torch.Tensor)
        else past_seen_token + sequence_length + 1
    )
    batch_size = input_ids.shape[0]

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
        return causal_mask < 0, position_ids


def _build_attentionmask_positionid_internllm(attention_mask, position_ids, dtype=torch.float32, device=torch.device("npu"), past_key_values_length=0, *args, **kwargs):
    # create causal mask

    # Copied from transformers.models.bart.modeling_bart._make_causal_mask
    def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    # Copied from transformers.models.bart.modeling_bart._expand_mask
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    input_shape = attention_mask.shape
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            dtype,
            device=device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1]).to(device)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask.bool(), position_ids


attention_mask_list = {'qwen2lm': _build_attentionmask_positionid_qwenllm, 'internllm': _build_attentionmask_positionid_internllm}


def prepare_positionsids_mask_for_llm(config=None, input_ids=None, inference_params=None, attention_mask=None, position_ids=None, *args, **kwargs):
    global_args = get_args()
    llm_model_id = getattr(global_args.mm.model.text_decoder, 'model_id', None)

    if llm_model_id and llm_model_id in attention_mask_list:
        return attention_mask_list[llm_model_id](config=config, input_ids=input_ids, inference_params=inference_params, attention_mask=attention_mask, position_ids=position_ids)
    else:
        raise AssertionError(f'the attention mask for llm is not build or model_id has error {llm_model_id}')