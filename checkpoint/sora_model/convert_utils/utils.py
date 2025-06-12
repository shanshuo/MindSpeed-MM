from collections import OrderedDict
from functools import wraps


def flip_mapping(mapping):
    if mapping:
        mapping = {value: key for key, value in mapping.items()}
        return OrderedDict(reversed(mapping.items()))
    else:
        return None


def replace_name(name, str_replace_mapping: dict):
    for old_str, new_str in str_replace_mapping.items():
        if len(old_str) > 1:
            name = name.replace(old_str, new_str)
    return name


# check method support wrapper
def check_method_support(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Get the list of supported methods defined in the caller's class.
        supported_methods = getattr(self, '_supported_methods', set())
        version = getattr(self, 'version', '')

        # Get the name of the currently called method.
        method_name = func.__name__

        # check support method
        if method_name not in supported_methods:
            raise NotImplementedError(
                f"Method '{method_name}' is not supported by Converter '{self.__class__.__name__}.{version}'."
            )
        
        # check support parallel config
        if method_name in ["hf_to_mm", "resplit", "layerzero_to_mm", "merge_lora_to_base", "mm_to_hf"]:
            cfg = kwargs['cfg']
            tp_size = cfg.target_parallel_config.tp_size
            pp_layers = cfg.target_parallel_config.pp_layers

            # check tp
            if tp_size > 1 and not getattr(self, '_enable_tp', False):
                raise NotImplementedError(
                    f"Tensor Parallel is not support by Converter '{self.__class__.__name__}.{version}'"
                )
            
            # check pp
            if len(pp_layers) > 1 and not getattr(self, '_enable_pp', False):
                raise NotImplementedError(
                    f"Pipeline Parallel is not support by Converter '{self.__class__.__name__}.{version}'"
                )

            # check vpp
            if len(pp_layers) > 1 and isinstance(pp_layers[0], list) and not getattr(self, '_enable_vpp', False):
                raise NotImplementedError(
                    f"Virtual Pipeline Parallel is not support by Converter '{self.__class__.__name__}.{version}'"
                )
        
        return func(self, *args, **kwargs)
    return wrapper