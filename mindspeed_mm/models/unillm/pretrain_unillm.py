# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain SoRA."""

import json
from typing import Dict, Any, Callable, Optional, Union

import torch
import torch_npu
from torch.utils.data import DataLoader, RandomSampler

import mindspeed.megatron_adaptor
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
import datasets

from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    unwrap_model,
)
from transformers.trainer_utils import has_length, seed_worker
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.utils import is_datasets_available


from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.training import pretrain
from mindspeed_mm.models.unillm.uni_dataset import UniDataset, DataCollatorWithFlatteningForSupervisedDataset, VLChatProcessor
from mindspeed_mm.models.unillm.unillm import UnifiedMultiModal


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_args()
    print_rank_0("building UnifiedMultiModal model ...")
    umm_config = args.mm.model
    model = UnifiedMultiModal(umm_config)

    return model


def get_batch(data_iterator):
    """Generate a batch."""
    batch = None
    if data_iterator is not None:
        data_item = next(data_iterator, None)
        batch = {
            "input_ids": data_item["input_ids"].to(torch_npu.npu.current_device()),
            "labels": data_item["labels"].to(torch_npu.npu.current_device()),
            "pixel_values": data_item["pixel_values"].to(torch_npu.npu.current_device()),
            "modals": data_item["modals"]
        }

    return batch


def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    batch = get_batch(data_iterator)
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    pixel_values = batch["pixel_values"]
    modals = batch["modals"]

    output_tensor = model(input_ids=input_ids, labels=labels, pixel_values=pixel_values, modals=modals)
    return output_tensor, loss_func


def _get_train_sampler(train_dataset) -> Optional[torch.utils.data.Sampler]:
    if train_dataset is None or not has_length(train_dataset):
        return None

    args = get_args()
    dataloader_param = args.mm.data.dataloader_param
    # Build the sampler.
    if dataloader_param.group_by_length:
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            lengths = (
                train_dataset[args.length_column_name]
                if args.length_column_name in train_dataset.column_names
                else None
            )
        else:
            lengths = None
        model_input_name = None
        return LengthGroupedSampler(
            args.train_batch_size * args.gradient_accumulation_steps,
            dataset=train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )

    else:
        return RandomSampler(train_dataset)


def train_valid_test_dataset_provider(train_val_test_num_samples) -> Dict:
    """Make batch flattened dataset and collator for supervised fine-tuning."""
    args = get_args()
    print("============dataset loader=============")
    data_arg_file_path = args.mm_data
    # 读取 JSON 文件
    with open(data_arg_file_path, 'r', encoding='utf-8') as f:
        data_args = json.load(f)
    tokenizer_config = args.mm.data.dataset_param.tokenizer_config
    basic_parameters = args.mm.data.dataset_param.basic_parameters
    preprocess_parameters = args.mm.data.dataset_param.preprocess_parameters
    dataloader_param = args.mm.data.dataloader_param
    vl_chat_processor = VLChatProcessor.from_pretrained(tokenizer_config.from_pretrained)
    vl_chat_processor.tokenizer_model_max_length = tokenizer_config.model_max_length
    vl_chat_processor.use_tokenizer_truncation = tokenizer_config.use_tokenizer_truncation
    basic_parameters.image_under_data_files = [filepath.strip() for filepath in basic_parameters.image_under_data_files.split(',')]
    basic_parameters.image_gen_data_files = [filepath.strip() for filepath in basic_parameters.image_gen_data_files.split(',')]
    basic_parameters.text_chat_data_files = [filepath.strip() for filepath in basic_parameters.text_chat_data_files.split(',')]

    print(f'{data_args=}!!!')
    train_dataset = UniDataset(
        vlprocessor=vl_chat_processor,
        image_under_data_files=basic_parameters.image_under_data_files,
        image_under_rootdir=basic_parameters.image_under_rootdir,
        image_gen_data_files=basic_parameters.image_gen_data_files,
        image_gen_rootdir=basic_parameters.image_gen_rootdir,
        text_chat_data_files=basic_parameters.text_chat_data_files,
        samples_per_epoch=preprocess_parameters.samples_per_epoch,
        dataset=preprocess_parameters.dataset,
        sample_rate=preprocess_parameters.sample_rate,
        batchsize_list=preprocess_parameters.batchsize_list
    )

    data_collator = DataCollatorWithFlatteningForSupervisedDataset(vlprocessor=vl_chat_processor)
    dataloader_params = {
        "batch_size": preprocess_parameters.batchsize_list[0],
        "collate_fn": data_collator,
        "num_workers": dataloader_param.num_workers,
        "pin_memory": dataloader_param.pin_memory,
        "persistent_workers": dataloader_param.persistent_workers,
    }
    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = _get_train_sampler(train_dataset)
        dataloader_params["drop_last"] = dataloader_param.drop_last
        dataloader_params["worker_init_fn"] = seed_worker
        dataloader_params["prefetch_factor"] = dataloader_param.dataloader_prefetch_factor
    train_dataloader = DataLoader(train_dataset, **dataloader_params)

    data_iterator, _, _ = build_iterations(train_dl=train_dataloader)
    return data_iterator, None, None

if __name__ == "__main__":
    train_valid_test_dataset_provider.is_distributed = True
    pretrain(
        train_valid_test_dataset_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external", "vision_pretraining": False, "curr_forward_iteration": 0},
    )
