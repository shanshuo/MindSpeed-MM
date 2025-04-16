import os
import importlib
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as ddp


def evaluate_long(self, videos_path, name, prompt_list=None, dimension_list=None, local=False, read_frame=False,
             mode='vbench_standard', **kwargs):
    from vbench.utils import init_submodules, save_json
    from vbench.distributed import get_rank, print0

    _dimensions = self.build_full_dimension_list()
    is_dimensional_structure = any(os.path.isdir(os.path.join(videos_path, dim)) for dim in _dimensions)
    kwargs['preprocess_dimension_flag'] = dimension_list
    if is_dimensional_structure:
        # 1. Under dimensions folders
        for dimension in _dimensions:
            dimension_path = os.path.join(videos_path, dimension)
            self.preprocess(dimension_path, mode, **kwargs)
    else:
        self.preprocess(videos_path, mode, **kwargs)

    # Now, long videos have been splitted into clips
    results_dict = {}
    if dimension_list is None:
        dimension_list = self.build_full_dimension_list()
    submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)
    # loop for build_full_info_json for clips

    cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, prompt_list, mode=mode, **kwargs)
    for dimension in dimension_list:
        try:
            dimension_module = importlib.import_module(f'vbench2_beta_long.{dimension}')
            evaluate_func = getattr(dimension_module, f'compute_long_{dimension}')
        except Exception as e:
            raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
        submodules_list = submodules_dict[dimension]
        print(f'cur_full_info_path: {cur_full_info_path}')

        results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
        results_dict[dimension] = results
    output_name = os.path.join(self.output_path, name + '_eval_results.json')
    if get_rank() == 0:
        save_json(results_dict, output_name)
        print0(f'Evaluation results saved to {output_name}')


def patch_static_filter_load_model(self):
    from vbench.third_party.RAFT.core.raft import RAFT

    self.model = ddp(RAFT(self.args).to(self.device))
    self.model.load_state_dict(torch.load(self.args.model))

    self.model = self.model.module
    self.model.eval()

