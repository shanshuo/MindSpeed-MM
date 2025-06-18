import torch

import mindspeed.megatron_adaptor
from mindspeed_mm.models.text_decoder.moe_model import topk_softmax_with_capacity
from tests.ut.utils import judge_expression

FINAL_PROBSE_MEAN = 0.8359375
FINAL_INDICES_MEAN = 4
TOKENS_PER_EXPERT_BEFORE_CAPACITY_MAX = 2
TOKENS_PER_EXPERT_BEFORE_CAPACITY_MIN = 0


class TestTopKSoftmaxWithCapacity:
    '''
    Test class for topk_softmax_with_capacity function.
    '''
    def test_topk_softmax_with_capacity(self):
        input_logits = torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4],
        ], dtype=torch.bfloat16, device="npu") # [num_tokens, num_experts]

        final_probs, final_indices, tokens_per_expert_before_capacity = topk_softmax_with_capacity(
            logits=input_logits,
            topk=3,
            capacity_factor=None,
            use_pre_softmax=True,
            num_groups=4,
            group_topk=2,
            scaling_factor=2.5,
            score_function="sigmoid",
            expert_bias=True,
            norm_topk_prob=True
        )

        judge_expression(final_probs.mean().item() == FINAL_PROBSE_MEAN)
        judge_expression(final_indices.mean().item() == FINAL_INDICES_MEAN)
        judge_expression(tokens_per_expert_before_capacity.max().item() == TOKENS_PER_EXPERT_BEFORE_CAPACITY_MAX)
        judge_expression(tokens_per_expert_before_capacity.min().item() == TOKENS_PER_EXPERT_BEFORE_CAPACITY_MIN)
