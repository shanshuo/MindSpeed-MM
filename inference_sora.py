import os

import torch
import mindspeed.megatron_adaptor
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.tasks.inference.pipeline import sora_pipeline_dict
from mindspeed_mm.tasks.inference.pipeline.utils.sora_utils import (
    save_videos, 
    save_video_grid,
    load_prompts,
    load_images,
    load_conditional_pixel_values
)
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm import Tokenizer
from mindspeed_mm.utils.utils import get_dtype, get_device, is_npu_available

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False


def prepare_pipeline(args, device):
    vae = AEModel(args.ae).get_model().to(device, args.ae.dtype).eval()
    text_encoder = TextEncoder(args.text_encoder).get_model().to(device).eval()
    predict_model = PredictModel(args.predictor).get_model().to(device, args.predictor.dtype).eval()
    scheduler = DiffusionModel(args.diffusion).get_model()
    tokenizer = Tokenizer(args.tokenizer).get_tokenizer()
    if not hasattr(vae, 'dtype'):
        vae.dtype = args.ae.dtype
    tokenizer.model_max_length = args.model_max_length
    sora_pipeline_class = sora_pipeline_dict[args.pipeline_class]
    sora_pipeline = sora_pipeline_class(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler,
                                        predict_model=predict_model, config=args.pipeline_config)
    return sora_pipeline


def main():
    initialize_megatron(extra_args_provider=mm_extra_args_provider, args_defaults={})
    args = get_args()
    merge_mm_args(args)
    args = args.mm.model
    # prepare arguments
    torch.set_grad_enabled(False)
    dtype = get_dtype(args.dtype)
    device = get_device(args.device)

    prompts = load_prompts(args.prompt)
    images = load_images(args.image) if hasattr(args, "image") else None
    conditional_pixel_values_path = load_conditional_pixel_values(args.conditional_pixel_values_path) if hasattr(args, "conditional_pixel_values_path") else None
    mask_type = args.mask_type if hasattr(args, "mask_type") else None
    crop_for_hw = args.crop_for_hw if hasattr(args, "crop_for_hw") else None
    max_hxw = args.max_hxw if hasattr(args, "max_hxw") else None

    if images is not None and len(prompts) != len(images):
        raise AssertionError(f'The number of images {len(images)} and the numbers of prompts {len(prompts)} do not match')

    save_fps = args.fps // args.frame_interval

    # prepare pipeline
    sora_pipeline = prepare_pipeline(args, device)

    # == Iter over all samples ==
    video_grids = []
    start_idx = 0
    for i in range(0, len(prompts), args.micro_batch_size):
        # == prepare batch prompts ==
        batch_prompts = prompts[i: i + args.micro_batch_size]
        kwargs = {}
        if conditional_pixel_values_path:
            batch_pixel_values_path = conditional_pixel_values_path[i: i + args.micro_batch_size]
            kwargs.update({"conditional_pixel_values_path": batch_pixel_values_path,
                           "mask_type": mask_type,
                           "crop_for_hw": crop_for_hw,
                           "max_hxw": max_hxw})

        if images is not None:
            batch_images = images[i: i + args.micro_batch_size]
        else:
            batch_images = None

        videos = sora_pipeline(prompt=batch_prompts, 
                               image=batch_images, 
                               fps=save_fps, 
                               max_sequence_length=args.model_max_length,
                               use_prompt_preprocess=args.use_prompt_preprocess,
                               device=device, 
                               dtype=dtype,
                               **kwargs
                               )
        save_videos(videos, start_idx, args.save_path, save_fps)
        start_idx += len(batch_prompts)
        video_grids.append(videos)
        print("Saved %s samples to %s" % (start_idx, args.save_path))

    video_grids = torch.cat(video_grids, dim=0)
    save_video_grid(video_grids, args.save_path, save_fps)
    print("Inference finished.")


if __name__ == "__main__":
    main()
