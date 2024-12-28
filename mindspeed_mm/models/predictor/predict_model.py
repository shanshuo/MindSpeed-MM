from torch import nn
from megatron.training.utils import print_rank_0
from megatron.core import mpu

from mindspeed_mm.models.common.checkpoint import load_checkpoint
from .dits import VideoDiT, Latte, STDiT, STDiT3, VideoDitSparse, SatDiT, VideoDitSparseI2V, PTDiT

PREDICTOR_MODEL_MAPPINGS = {
    "videodit": VideoDiT,
    "videoditsparse": VideoDitSparse,
    "videoditsparsei2v": VideoDitSparseI2V,
    "latte": Latte,
    "stdit": STDiT,
    "stdit3": STDiT3,
    "satdit": SatDiT,
    "ptdit": PTDiT,
}


class PredictModel(nn.Module):
    """
    The backnone of the denoising model
    PredictModel is the factory class for all unets and dits

    Args:
        config[dict]: for Instantiating an atomic methods
    """

    def __init__(self, config):
        super().__init__()
        model_cls = PREDICTOR_MODEL_MAPPINGS[config.model_id]
        config = self._build_predictor_layers_config(config)
        self.predictor = model_cls(**config.to_dict())
        if config.from_pretrained is not None:
            load_checkpoint(self.predictor, config.from_pretrained)
            print_rank_0("load predictor's checkpoint sucessfully")

    def get_model(self):
        return self.predictor

    def _build_predictor_layers_config(self, config):
        if mpu.get_pipeline_model_parallel_world_size() <= 1:
            return config
        
        pp_rank = mpu.get_pipeline_model_parallel_rank()

        if not hasattr(config, "pipeline_num_layers"):
            raise ValueError(f"The `pipeline_num_layers` must be specified in the config for pipeline parallel")
        if sum(config.pipeline_num_layers) != config.num_layers:
            raise ValueError(f"The sum of `pipeline_num_layers` must be equal to the `num_layers`")
        
        local_num_layers = config.pipeline_num_layers[pp_rank]
        if local_num_layers <= 0:
            raise ValueError(f"for pp_rank {pp_rank}, the predictor layer is {local_num_layers}, "
                             f"which is invalid. ")

        pipeline_start_idx = sum(config.pipeline_num_layers[:pp_rank])
        pipeline_end_idx = sum(config.pipeline_num_layers[:pp_rank + 1])

        config.num_layers = local_num_layers
        config.pre_process = mpu.is_pipeline_first_stage()
        config.post_process = mpu.is_pipeline_last_stage()
        config.global_layer_idx = tuple(range(pipeline_start_idx, pipeline_end_idx))

        return config