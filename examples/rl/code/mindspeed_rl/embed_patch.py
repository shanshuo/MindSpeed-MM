# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import List, Mapping, Optional, Tuple, Union, cast, TYPE_CHECKING
from typing_extensions import assert_never

import vllm
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalEncDecInputs,
                                    MultiModalInputs)
from vllm.inputs.data import (DecoderOnlyInputs, EncoderDecoderInputs, ProcessorInputs,
                   PromptType, SingletonInputs, SingletonPrompt, token_inputs)
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.multimodal.base import MultiModalPlaceholderMap

if TYPE_CHECKING:
    from vllm.sequence import SequenceGroupMetadata


def _process_multimodal_patch(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Optional[Mapping[str, object]],
        lora_request: Optional[LoRARequest],
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
    """
    Apply the model's multi-modal processor to a multi-modal prompt,
    returning the corresponding token IDs and metadata.
    """
    tokenizer = self._get_mm_tokenizer(lora_request)

    if "image_embeds" in mm_data and "image_grid_thw" in mm_data:
        return MultiModalInputs(
            type="multimodal",
            prompt=tokenizer.decode(prompt),
            prompt_token_ids=prompt,
            mm_kwargs=mm_data,
            mm_hashes=None,
            mm_placeholders=None,
        )

    mm_processor = self.mm_registry.create_processor(self.model_config,
                                                        tokenizer=tokenizer)

    if mm_processor_kwargs is None:
        mm_processor_kwargs = {}

    return mm_processor.apply(prompt, mm_data, mm_processor_kwargs,
                                return_mm_hashes)


async def _process_multimodal_async_patch(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Optional[Mapping[str, object]],
        lora_request: Optional[LoRARequest],
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
    tokenizer = await self._get_mm_tokenizer_async(lora_request)

    if "image_embeds" in mm_data and "image_grid_thw" in mm_data:
        return MultiModalInputs(
            type="multimodal",
            prompt=tokenizer.decode(prompt),
            prompt_token_ids=prompt,
            mm_kwargs=mm_data,
            mm_hashes=None,
            mm_placeholders=None,
        )

    mm_processor = self.mm_registry.create_processor(self.model_config,
                                                        tokenizer=tokenizer)
    if mm_processor_kwargs is None:
        mm_processor_kwargs = {}

    return mm_processor.apply(prompt, mm_data, mm_processor_kwargs,
                                return_mm_hashes)


@classmethod
def from_seq_group(
    cls, seq_group: "SequenceGroupMetadata", positions: range
) -> tuple[MultiModalKwargs, dict[str, "MultiModalPlaceholderMap"]]:

    seq_mm_data = seq_group.multi_modal_data
    seq_mm_placeholders = seq_group.multi_modal_placeholders

    if seq_mm_data and not seq_mm_placeholders:
        return seq_mm_data, {}

    if not seq_mm_data or not seq_mm_placeholders:
        return MultiModalKwargs({}), {}

    placeholder_maps = dict[str, MultiModalPlaceholderMap]()

    for modality, placeholders in seq_mm_placeholders.items():
        placeholder_map = MultiModalPlaceholderMap()

        if positions:
            placeholder_map.append_items_from_seq_group(
                positions,
                # Dummy, since we don't care about intersecting items
                [None] * len(placeholders),
                placeholders,
            )

        placeholder_maps[modality] = placeholder_map

    return seq_mm_data, placeholder_maps


def image_emb_reuse():
    if '0.9.1' in vllm.__version__:
        vllm.inputs.preprocess.InputPreprocessor._process_multimodal = _process_multimodal_patch
        vllm.inputs.preprocess.InputPreprocessor._process_multimodal_async = _process_multimodal_async_patch
        vllm.multimodal.base.MultiModalPlaceholderMap.from_seq_group = from_seq_group
    else:
        # This patch enables ViT embedding reuse by passing and reusing pre-computed embeds,
        # to reduce unnecessary vision encoder computation.
        # However, the patch logic is version-dependent and may need adjustment
        # as vLLM evolves with internal API changes.

        import warnings
        warnings.warn(
            "ViT reuse feature requires patching vLLM internals to pass and use pre-computed embeddings. "
            "The current implementation is only verified for vLLM version 0.9.1. "
            "This patch may be incompatible or unsafe with other versions. "
            "Please adjust the patch logic accordingly when upgrading vLLM."
        )