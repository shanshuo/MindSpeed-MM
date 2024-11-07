from functools import partial

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor

from mindspeed_mm.data.data_utils.func_utils.convert import DataArguments, DatasetAttr, load_tokenizer, \
    convert_sharegpt, preprocess_supervised_dataset
from mindspeed_mm.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.data.data_utils.func_utils.template import get_template_and_fix_tokenizer


def get_qwen2vl_dataset(basic_param, preprocess_param, dataset_param):
    data_args = DataArguments(**basic_param)
    process_args = ProcessorArguments(**preprocess_param)
    dataset_attr = DatasetAttr(**dataset_param["attr"])
    tokenizer_module = load_tokenizer(process_args)
    tokenizer = tokenizer_module['tokenizer']
    processor = AutoProcessor.from_pretrained(process_args.model_name_or_path)
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
    # -----------------load dataset from file-------------------------------------------------------------------------
    dataset = load_dataset(
        path="json",
        name=None,
        data_dir=None,
        data_files=[data_args.dataset],
        split="train",
        cache_dir=data_args.cache_dir,
        token=None,
        streaming=data_args.streaming,
        trust_remote_code=True,
    )
    # -----------------convert to sharegpt ---------------------------------------------------------------------------
    convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, dataset_dir=data_args.dataset_dir)
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        local_process_index = 0
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (local_process_index != 0),
            desc="Converting format of dataset",
        )
    if data_args.max_samples:
        dataset = dataset.select(range(data_args.max_samples))
    dataset = dataset.map(
        convert_func,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )
    # -----------------convert text to token id ----------------------------------------------------------------------
    preprocess_func = partial(
        preprocess_supervised_dataset,
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (local_process_index != 0),
            desc="Running tokenizer on dataset",
        )
    dataset = dataset.map(
        preprocess_func,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )
    return dataset


class Qwen2vlDataset(Dataset):
    def __init__(self, basic_param, preprocess_param, dataset_param):
        self.dataset = get_qwen2vl_dataset(basic_param, preprocess_param, dataset_param)
        super().__init__()

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
