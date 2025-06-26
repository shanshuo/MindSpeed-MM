import json
import os
import time
import uuid
import copy

import mindspeed.megatron_adaptor
import torch
import torch.distributed
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from numpy import save

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.constants import (
    FILE_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO,
    VIDEO_MASK,
)
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.tools.profiler import Profiler
from mindspeed_mm.utils.utils import get_device, get_dtype, is_npu_available

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False


def prepare_model(args, device):
    vae = AEModel(args.mm.model.ae).to(device, args.mm.model.ae.dtype).eval()
    text_encoder = TextEncoder(args.mm.model.text_encoder).to(device).eval()
    return vae, text_encoder


def get_pt_name(file_name):
    pt_name = os.path.basename(file_name).replace(".", "_") + ".pt"
    return pt_name


def extract_feature():
    
    initialize_megatron(extra_args_provider=mm_extra_args_provider, args_defaults={})
    args = get_args()
    merge_mm_args(args)
    
    save_path = args.mm.tool.sorafeature.save_path
    
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(os.path.join(save_path, 'features')):
            os.makedirs(os.path.join(save_path, 'features'))

    
    set_jit_fusion_options()
    torch.set_grad_enabled(False)
    dtype = get_dtype(args.mm.model.ae.dtype)
    device = get_device("npu")

    train_dataset = build_mm_dataset(args.mm.data.dataset_param)
    train_dataloader = build_mm_dataloader(
        train_dataset,
        args.mm.data.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
        dataset_param=args.mm.data.dataset_param,
    )

    # master rank, write data info jsonl
    if torch.distributed.get_rank() == 0:
        with open(os.path.join(save_path, 'data.jsonl'), 'w', encoding="utf-8") as json_file:
            for data_sample in train_dataset.data_samples:
                source_data_storage_mode = args.mm.data.dataset_param.basic_parameters.data_storage_mode
                if source_data_storage_mode == "combine":
                    source_file_key = "path"
                elif source_data_storage_mode == "standard":
                    source_file_key = FILE_INFO
                else: 
                    raise NotImplementedError(f"Extract features from data storage mode {source_data_storage_mode} is not implemented")
                
                file_name = data_sample[source_file_key]
                pt_name = get_pt_name(file_name)
                data_info = copy.deepcopy(data_sample)

                data_info.update({
                    FILE_INFO: f"features/{pt_name}"
                })
                json_file.write(json.dumps(data_info) + '\n')
    
    vae, text_encoder = prepare_model(args, device)
    
    start_time = time.time()
    print_rank_0(f"Features extraction begins. {len(train_dataloader)} data in total.")
    counter = 0
    if hasattr(args.mm.tool, "profile"):
        prof = Profiler(args.mm.tool.profile)
        prof.start()

    for batch in train_dataloader:
        if batch:  
            video = batch.pop(VIDEO).to(device, dtype)
            prompt_ids = batch.pop(PROMPT_IDS)
            prompt_mask = batch.pop(PROMPT_MASK)
            file_names = batch.pop(FILE_INFO)
        else:
            raise ValueError("Batch is None!")
        
        # extract feature
        latents, latents_dict = vae.encode(video, **batch)
        prompt, prompt_mask = text_encoder.encode(prompt_ids, prompt_mask)

        bs = video.shape[0]
        counter += bs

        for i in range(bs):
            pt_name = get_pt_name(file_names[i])
            latent_i = latents[i].cpu()
            if isinstance(prompt_ids, (list, tuple)):
                prompts_i = [_prompt[i].cpu() for _prompt in prompt]
                prompt_masks_i = [_prompt_mask[i].cpu() for _prompt_mask in prompt_mask]
            else:
                prompts_i = prompt[i].cpu()
                prompt_masks_i = prompt_mask[i].cpu()

            data_to_save = {
                VIDEO: latent_i,
                PROMPT_IDS: prompts_i,
                PROMPT_MASK: prompt_masks_i
            }

            # other i2v variables
            if latents_dict:
                for key in latents_dict.keys():
                    if isinstance(latents_dict[key][i], torch.Tensor):
                        data_to_save[key] = latents_dict[key][i].cpu()
                    else:
                        data_to_save[key] = latents_dict[key][i]

            torch.save(data_to_save, os.path.join(save_path, 'features', pt_name))

        print_rank_0(f"consumed sample {counter} | elapsed time {(time.time() - start_time):.2f} | file {file_names}")
        
        if hasattr(args.mm.tool, "profile"):
            prof.step()
    if hasattr(args.mm.tool, "profile"):
        prof.stop()

    duration = time.time() - start_time
    print_rank_0(f"{counter} feature vectors extracted in {duration:.2f} seconds.")


if __name__ == "__main__":
    extract_feature()
