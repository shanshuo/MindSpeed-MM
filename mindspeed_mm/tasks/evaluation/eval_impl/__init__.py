from mindspeed_mm.tasks.evaluation.eval_impl.impl_mmmu import MMMUEvalImpl
from mindspeed_mm.tasks.evaluation.eval_impl.impl_vqa import VQAEvalImpl
from mindspeed_mm.tasks.evaluation.eval_impl.impl_ai2d import AI2DEvalImpl
from mindspeed_mm.tasks.evaluation.eval_impl.vbench_gen import VbenchGenEvalImpl

from mindspeed_mm.tasks.inference.pipeline import vlm_pipeline_dict, sora_pipeline_dict

eval_impl_dict = {"mmmu_dev_val": MMMUEvalImpl, "ai2d_test": AI2DEvalImpl, "chartqa_test": VQAEvalImpl,
                  "docvqa_val": VQAEvalImpl, "vbench_eval": VbenchGenEvalImpl}

eval_pipeline_dict = {"llava_v1.5_7b": vlm_pipeline_dict["LlavaPipeline"],
                      "internvl2_8b": vlm_pipeline_dict["InternVLPipeline"],
                      "qwen2_vl_7b": vlm_pipeline_dict["Qwen2VlPipeline"],
                      "cogvideox-1.5": sora_pipeline_dict["CogVideoXPipeline"],
                      "OpenSoraPlan-1.3": sora_pipeline_dict["OpenSoraPlanPipeline"],
                      }
