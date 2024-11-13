import os

import torch
import mindspeed.megatron_adaptor
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.tasks.inference.pipeline import SoraPipeline_dict
from mindspeed_mm.tasks.inference.pipeline.utils.sora_utils import (
    save_videos, 
    save_video_grid,
    load_prompts,
    load_images
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
    sora_pipeline_class = SoraPipeline_dict[args.pipeline_class]
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
                               dtype=dtype
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
