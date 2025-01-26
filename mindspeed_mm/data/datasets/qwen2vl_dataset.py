import os
from functools import partial

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers.training_args import TrainingArguments

from mindspeed_mm.data.data_utils.func_utils.convert import (
    DataArguments,
    DatasetAttr,
    load_tokenizer,
    convert_sharegpt,
    preprocess_supervised_dataset,
    preprocess_pairwise_dataset
)
from mindspeed_mm.data.data_utils.func_utils.log import get_logger
from mindspeed_mm.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.data.data_utils.func_utils.template import get_template_and_fix_tokenizer

logger = get_logger(__name__)


def get_qwen2vl_dataset(basic_param, preprocess_param, dataset_param):
    data_args = DataArguments(**basic_param)
    process_args = ProcessorArguments(**preprocess_param)
    dataset_attr = DatasetAttr(**dataset_param["attr"])

    tokenizer_module = load_tokenizer(process_args)
    tokenizer, processor = tokenizer_module['tokenizer'], tokenizer_module['processor']
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
    # 确保主进程进行数据处理，其他进程复用缓存避免重复计算，该策略和llamafactory对数据处理策略一致
    with TrainingArguments(output_dir='./').main_process_first(desc="pre-process dataset"):
        # -----------------load dataset from file-------------------------------------------------------------------------
        dataset = load_dataset(path="json", data_files=data_args.dataset, split="train", cache_dir=data_args.cache_dir,
                               streaming=data_args.streaming)
        if data_args.max_samples:
            dataset = dataset.select(range(data_args.max_samples))
        local_process_index = int(os.getenv("LOCAL_RANK", -1))
        if data_args.streaming:
            kwargs = {}
        else:
            kwargs = {
                "num_proc": data_args.preprocessing_num_workers,
                # 配置了overwrite_cache为false（默认为false)时，非rank0节点读取cache不再进行map处理
                # 配置了overwrite_cache为true（默认为false)时，所有节点都读取cache不再进行map处理
                "load_from_cache_file": (not data_args.overwrite_cache) or (local_process_index != 0)
            }
        logger.debug(f'Rank: %s, kwargs: %s', local_process_index, kwargs)
        # -----------------convert to sharegpt ---------------------------------------------------------------------------
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, dataset_dir=data_args.dataset_dir)
        dataset = dataset.map(
            convert_func,
            batched=False,
            remove_columns=(list(next(iter(dataset)).keys())),
            desc=f"Rank {local_process_index}, Converting format of dataset",
            **kwargs,
        )
        # -----------------convert text to token id ----------------------------------------------------------------------
        if dataset_attr.ranking:
            preprocess_func = partial(
                preprocess_pairwise_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )
        else:
            preprocess_func = partial(
                preprocess_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )
        dataset = dataset.map(
            preprocess_func,
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            remove_columns=(list(next(iter(dataset)).keys())),
            desc=f"Rank {local_process_index}, Running tokenizer on dataset",
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
