import torch
import torch_npu

import mindspeed.megatron_adaptor
from megatron.training import get_args
from mindspeed_mm.tasks.inference.pipeline import VlmPipeline_dict
from mindspeed_mm.configs.config import mm_extra_args_provider


def main():
    from megatron.training.initialize import initialize_megatron
    from mindspeed_mm.configs.config import merge_mm_args
    initialize_megatron(
        extra_args_provider=mm_extra_args_provider, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
    args = get_args()
    merge_mm_args(args)
    inference_config = args.mm.model
    inference_pipeline = VlmPipeline_dict[inference_config.model_id](inference_config)
    response = inference_pipeline()
    print(response)


if __name__ == '__main__':
    main()