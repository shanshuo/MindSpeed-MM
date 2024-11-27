from .datasets_base import BaseEvalDataset
from .datasets_mmmu import MMMUEvalDataset
from .datasets_vqa import VQAEvalDataset

eval_dataset_dict = {"mmmu_dev_val": MMMUEvalDataset, "ai2d_test": BaseEvalDataset, "chartqa_test": VQAEvalDataset,
                     "docvqa_val": VQAEvalDataset}
