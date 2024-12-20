from .datasets_mmmu import MMMUEvalDataset
from .datasets_vqa import VQAEvalDataset
from .datasets_ai2d import AI2DEvalDataset

eval_dataset_dict = {"mmmu_dev_val": MMMUEvalDataset, "ai2d_test": AI2DEvalDataset, "chartqa_test": VQAEvalDataset,
                     "docvqa_val": VQAEvalDataset}
