# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from mindspeed_rl.utils.utils import is_multimodal
from .qwen2_5_vl_visionmlp_patch import replace_qwen2_5_visionmlp
from .embed_patch import image_emb_reuse
from .rotary_embedding_patch import MRotaryEmbedding_forward_patch

if is_multimodal():
    replace_qwen2_5_visionmlp()
    image_emb_reuse()
    MRotaryEmbedding_forward_patch()