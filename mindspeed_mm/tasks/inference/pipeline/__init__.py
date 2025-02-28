__all__ = ["sora_pipeline_dict", "vlm_pipeline_dict"]

from mindspeed_mm.tasks.inference.pipeline.opensora_pipeline import OpenSoraPipeline
from mindspeed_mm.tasks.inference.pipeline.opensoraplan_pipeline import OpenSoraPlanPipeline
from mindspeed_mm.tasks.inference.pipeline.cogvideox_pipeline import CogVideoXPipeline
from mindspeed_mm.tasks.inference.pipeline.internvl_pipeline import InternVLPipeline
from mindspeed_mm.tasks.inference.pipeline.llava_pipeline import LlavaPipeline
from mindspeed_mm.tasks.inference.pipeline.qihoo_pipeline import QihooPipeline
from mindspeed_mm.tasks.inference.pipeline.qwen2vl_pipeline import Qwen2VlPipeline
from mindspeed_mm.tasks.inference.pipeline.hunyuanvideo_pipeline import HunyuanVideoPipeline

sora_pipeline_dict = {"OpenSoraPlanPipeline": OpenSoraPlanPipeline,
                      "OpenSoraPipeline": OpenSoraPipeline,
                      "QihooPipeline": QihooPipeline,
                      "CogVideoXPipeline": CogVideoXPipeline,
                      "HunyuanVideoPipeline": HunyuanVideoPipeline}

vlm_pipeline_dict = {
    "InternVLPipeline": InternVLPipeline,
    "LlavaPipeline": LlavaPipeline,
    "Qwen2VlPipeline": Qwen2VlPipeline
}

