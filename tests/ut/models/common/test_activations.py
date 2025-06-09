import torch.nn as nn

from mindspeed_mm.models.common.activations import get_activation_layer
from tests.ut.utils import judge_expression


class TestActivation:
    """ 
    Test activation basic function.
    """
    def test_activation_when_get_right_act_type(self):
        act_type = "relu"
        res = get_activation_layer(act_type)
        judge_expression(isinstance(res, type))