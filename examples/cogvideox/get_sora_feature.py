import os
import time
import torch
import mindspeed.megatron_adaptor

from megatron.core import mpu
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.initialize import set_jit_fusion_options

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.utils.utils import get_dtype, get_device, is_npu_available
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset

from mindspeed_mm.data.data_utils.constants import (
    VIDEO, 
    PROMPT_IDS, 
    PROMPT_MASK, 
    VIDEO_MASK,
    PROMPT_IDS_2, 
    PROMPT_MASK_2, 
)

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False


def prepare_model(args, device):
    vae = AEModel(args.mm.model.ae).to(device, args.mm.model.ae.dtype).eval()
    text_encoder = TextEncoder(args.mm.model.text_encoder).to(device).eval()
    return vae, text_encoder


def main(extract_video_feature=True, extract_text_feature=True):
    initialize_megatron(extra_args_provider=mm_extra_args_provider, args_defaults={})
    args = get_args()
    merge_mm_args(args)

    set_jit_fusion_options()
    torch.set_grad_enabled(False)
    dtype = get_dtype(args.mm.model.ae.dtype)
    device = get_device("npu")

    train_dataset = build_mm_dataset(args.mm.data.dataset_param)
    train_dataloader = build_mm_dataloader(
        train_dataset,
        args.mm.data.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
    )
    
    vae, text_encoder = prepare_model(args, device)
    
    start_time = time.time()
    for batch in train_dataloader:
        
        video = batch.pop(VIDEO).to(device, dtype)
        prompt_ids = batch.pop(PROMPT_IDS).to(device)
        prompt_mask = batch.pop(PROMPT_MASK).to(device)
        
        if extract_video_feature:
            latents = vae.encode(video)[0]
        else:
            latents = video
        
        if extract_text_feature:
            B, N, L = prompt_ids.shape
            prompt_ids = prompt_ids.view(-1, L)
            prompt_mask = prompt_mask.view(-1, L)
            hidden_states = text_encoder.encode(prompt_ids, prompt_mask)
            prompt = hidden_states["last_hidden_state"].view(B, N, L, -1)
        else:
            prompt = prompt_ids
            prompt_mask = prompt_mask
        
        data_to_save = {
            "latents": latents.squeeze(0),
            "prompt": prompt.squeeze(0),
            "prompt_mask": prompt_mask.squeeze(0)
        }
    
        loca = time.strftime('%Y-%m-%d-%H-%M-%S')
        pt_name = "feature" + loca + ".pt"
        torch.save(data_to_save, "save_path" + pt_name)

    duration = time.time() - start_time
    print(f"Feature extraction completed in {duration:.2f} seconds.")


if __name__ == "__main__":
    extract_video_feature = True
    extract_text_feature = True
    main(extract_video_feature=extract_video_feature, extract_text_feature=extract_text_feature)
