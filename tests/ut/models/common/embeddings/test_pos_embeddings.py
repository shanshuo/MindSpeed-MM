import mindspeed.megatron_adaptor

from mindspeed_mm.models.common.embeddings.pos_embeddings import RoPE3DSORA, RoPE3DStepVideo
from tests.ut.utils import judge_expression


class TestPosEmbeddings:
    def test_init_rope3dsora(self):
        rope = RoPE3DSORA(30000, freq=10000.0, interpolation_scale=(1, 1, 1))
        judge_expression(isinstance(rope, RoPE3DSORA))

    def test_init_rope3dstepvideo(self):
        rope = RoPE3DStepVideo([64, 32, 32], freq=10000.0)
        judge_expression(isinstance(rope, RoPE3DStepVideo))