import mindspore
from mindspeed.patch_utils import MindSpeedPatchesManager as aspm
from mindspeed_mm.mindspore.data.datasets.utils import process_in_cpu_wrapper
from mindspeed_mm.mindspore.data.data_utils.func_utils.convert import preprocess_dataset


def masked_scatter_(self, mask, updates):
    origin_dtype = None
    if self.dtype in (mindspore.float16, mindspore.bfloat16):
        origin_dtype = self.dtype
        self = self.to(mindspore.float32)
    if updates.dtype in (mindspore.float16, mindspore.bfloat16):
        updates = updates.to(mindspore.float32)
    self = mindspore.ops.MaskedScatter()(self, mask, updates)
    if origin_dtype is not None:
        self = self.to(origin_dtype)
    return self


def apply_mindspore_patch():
    aspm.register_patch('mindspeed_mm.data.datasets.qwen2vl_dataset.get_qwen2vl_dataset', process_in_cpu_wrapper) # process dataset on cpu
    aspm.register_patch('torch.Tensor.masked_scatter', masked_scatter_)
    aspm.register_patch('mindspeed_mm.data.data_utils.func_utils.convert.SupervisedDatasetProcessor.preprocess_dataset', preprocess_dataset)
    aspm.apply_patches()

apply_mindspore_patch()
