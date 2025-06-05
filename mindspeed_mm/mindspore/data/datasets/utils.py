import mindspore


def process_in_cpu_wrapper(func):
    """
    By default, MindSpore do everything on npu. Here manually set cpu when processing dataset
    """
    def wrapper(*args, **kwargs):
        # set device to CPU
        mindspore.set_context(device_target="CPU")
 
        # process dataset
        result = func(*args, **kwargs)

        #set device to Ascend
        mindspore.set_context(device_target="Ascend")
        
        return result
    return wrapper