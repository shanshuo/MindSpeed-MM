import re
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Dict, List, Tuple, Any

import torch
from torch import Tensor

STATE_DICT_T = Dict[str, torch.Tensor]
TP_PATTERN_T = Dict[str, "BaseSplit"]
DIGIT_FMT = r'(\d+)'
QV_NAME_T = Tuple[str, str]
QKV_NAME_T = Tuple[str, str, str]


def interleaved_qkv_to_concated(megatron_qkv: Tensor, num_key_value_heads: int, split_size: List[int]) -> Tensor:
    """mindspeed-mm中的qkv排布和vllm中qkv的排布不同，因此需要重新转换，qwen2vl的转换逻辑如下：
    qkv = [nq1,k1,v1,  nq2,k2,v2,  nq3,k3,v3,  nq4,k4,v4]
    new_qkv = [nq1,nq2,nq3,nq4,k1,k2,k3,k4,v1,v2,v3,v4]
    """
    groups_qkv = [torch.split(group, split_size) for group in torch.chunk(megatron_qkv, num_key_value_heads)]
    return torch.cat([head for group_head in zip(*groups_qkv) for head in group_head])


def concated_qkv_to_interleaved(qkv: Tensor, num_key_value_heads: int, split_size: List[int]) -> Tensor:
    """
    qkv : [nq1,nq2,nq3,nq4,k1,k2,k3,k4]
    new_qkv : [nq1,k1,v1,  nq2,k2,v2,  nq3,k3,v3,  nq4,k4,v4]
    """
    qkv_split = torch.split(qkv, split_size)
    qkv_chunk_group = [torch.chunk(i, num_key_value_heads) for i in qkv_split]
    return torch.cat([i for j in zip(*qkv_chunk_group) for i in j])


def merge_to_interleaved_qkv(q: Tensor, k: Tensor, v: Tensor, group: int) -> Tensor:
    """原顺序: q=[nq1,nq2,nq3,...], k=[k1,k2,k3,...], v=[v1,v2,v3,...]
    转换后顺序: qkv = [nq1,k1,v1,nq1,k2,v2,nq3,k3,v3,...]
    这里nq1，group query attention时，多个q共享一个k/v， k1/v1即一个key head对应的head_dim维度
    """
    qkv_chunks = [torch.chunk(x, chunks=group, dim=0) for x in [q, k, v]]
    # zip(*[])转换迭代顺序，qkv_group即一组qkv，包含一个k/v头以及对应的多个q，[nq1,k1,v1]
    return torch.concat([head for qkv_group in zip(*qkv_chunks) for head in qkv_group], dim=0)


def split_to_interleaved_qkv(value: Tensor, num_heads: int, q_size: int, k_size: int, v_size: int) -> Tuple[
    Tensor, Tensor, Tensor]:
    # 分块 + zip 重组逻辑（逆向操作）
    qkv_chunks = torch.chunk(value, num_heads, dim=0)
    q_parts, k_parts, v_parts = zip(*[torch.split(chunk, [q_size, k_size, v_size]) for chunk in qkv_chunks])
    return torch.cat(q_parts), torch.cat(k_parts), torch.cat(v_parts)


def get_layer_num(pattern: str, names: Iterable[str]) -> int:
    """在names中找到pattern正则匹配到的数数字的最大值"""
    layer_ids = [int(re.findall(pattern, k)[0]) for k in names if re.match(pattern, k)]
    if len(layer_ids) == 0:
        return 0
    return max(layer_ids) + 1


def get_layer_and_expert_num(pattern: str, names: Iterable[str]) -> Tuple[int, int]:
    """仅在含有expert相关的参数时调用，匹配到的第一个数字为层数，第二个数字为专家数"""
    layer_num = max(int(re.findall(pattern, k)[0][0]) for k in names if re.match(pattern, k)) + 1
    expert_num = max(int(re.findall(pattern, k)[0][1]) for k in names if re.match(pattern, k)) + 1
    return layer_num, expert_num


class Operator(ABC):

    @abstractmethod
    def apply(self, weights: STATE_DICT_T):
        """原地修改weights字典"""
        pass

    @abstractmethod
    def revert(self, weights: STATE_DICT_T):
        """原地修改weights字典"""
        pass


class ZeroWeightOp(Operator):
    """
    权重置零操作，将权重值赋值为零或向权重字典添加全零权重
    权重转换需要把q,k,v权重合并成一个完整的qkv大矩阵。
    而由于在Qwen2.5-Omni模型的音频模型中，k没有bias，所以需要将k的bias以全零权重的形式添加到权重字典
    """

    def __init__(self, name: str, shape: List[int], layers: int):
        """
        初始化权重信息

        参数:
        - name (str): 权重的名称。
        - shape (List[int]): 权重的的形状，以整数列表形式表示。
        - layers (int): 模型的层数。
        """
        self.name = name
        self.shape = shape
        self.layers = layers

    def apply(self, weights: STATE_DICT_T):
        """向weights字典中添加权重或修改权重为0"""
        for layer_num in range(self.layers):
            tensor = torch.zeros(self.shape)
            name = self.name.replace(DIGIT_FMT, str(layer_num))
            weights[name] = tensor

    def revert(self, weights: STATE_DICT_T) -> None:
        """反向操作：删除由 apply 添加的零权重键"""
        for layer_num in range(self.layers):
            name = self.name.replace(DIGIT_FMT, str(layer_num))
            # 安全删除（仅在键存在时移除）
            weights.pop(name, None)


class RenameOp(Operator):

    def __init__(self, patterns: Tuple[Tuple[str, str], ...]):
        self.patterns = patterns

    @staticmethod
    def replace_parentheses(text):
        """
        使用序号替换文本中的括号内容。

        该方法使用正则表达式匹配文本中所有的括号内容，并使用一个递增的序号替换它们。
        序号以反斜杠开头，如 \1, \2 等。

        参数:
        text -- 待处理的文本字符串。

        返回:
        替换后的文本字符串。

        doctest 示例:

        >>> RenameOp.replace_parentheses("这是一个测试 (示例) 和另一个 (例子)")
        '这是一个测试 \\1 和另一个 \\2'

        >>> RenameOp.replace_parentheses("无括号文本")
        '无括号文本'

        >>> RenameOp.replace_parentheses("(第一个) 包含多个 (括号内容) 示例")
        '\\1 包含多个 \\2 示例'
        """
        count = 0  # 初始化计数器，用于生成序号

        def repl(match):
            """
            替换匹配到的括号内容为序号。

            参数:
            match -- 正则表达式匹配到的对象。

            返回:
            包含序号的字符串，如 \1, \2 等。
            """
            nonlocal count  # 使用 nonlocal 声明，以便能够修改外部的 count 变量
            count += 1  # 计数器递增
            return f'\\{count}'  # 返回如 \1, \2 的字符串

        # 使用正则表达式匹配所有的括号内容，并使用 repl 函数进行替换
        return re.sub(r'\([^)]*\)', repl, text)

    def apply(self, weights: STATE_DICT_T):
        mapping = {}
        for weight_name in weights:
            new_name = weight_name
            for pattern, replacement in self.patterns:
                replacement = self.replace_parentheses(replacement)
                new_name = re.sub(pattern, replacement, new_name)
            if new_name != weight_name:
                mapping[weight_name] = new_name
        for old_name, new_name in mapping.items():
            weights[new_name] = weights.pop(old_name)

    def revert(self, weights: STATE_DICT_T):
        self.patterns = tuple((right, left) for left, right in self.patterns)
        self.apply(weights=weights)


class MergeOp(Operator):
    def __init__(self, raw_names: Any, new_name: str):
        self.raw_names = raw_names
        self.new_name = new_name

    @abstractmethod
    def merge(self, tensors: List[Tensor]) -> Tensor:
        pass

    @abstractmethod
    def split(self, tensor: Tensor) -> List[Tensor]:
        pass

    def apply(self, weights: STATE_DICT_T):
        layer_num = get_layer_num(self.raw_names[0], weights.keys())
        for num in range(layer_num):
            merged_tensor = self.merge([weights.pop(name.replace(DIGIT_FMT, str(num))) for name in self.raw_names])
            weights[self.new_name.replace(DIGIT_FMT, str(num))] = merged_tensor

    def revert(self, weights: STATE_DICT_T) -> None:
        """反向操作：拆分合并后的权重并恢复原始键"""
        layer_num = get_layer_num(self.new_name, weights.keys())
        for num in range(layer_num):
            # 获取合并后的张量
            merged_name = self.new_name.replace(DIGIT_FMT, str(num))
            merged_tensor = weights.pop(merged_name)  # 移除合并后的键
            # 拆分回原始小矩阵列表
            split_tensors = self.split(merged_tensor)
            if len(split_tensors) != len(self.raw_names):
                raise ValueError(f"Split tensor count {len(split_tensors)} != raw names count {len(self.raw_names)}")
            # 将拆分后的张量写回原始键
            for raw_name, tensor in zip(self.raw_names, split_tensors):
                weights[raw_name.replace(DIGIT_FMT, str(num))] = tensor


class UpGateMergeOp(MergeOp):
    """GLU门控单元中权重合并逻辑，从transformers的gate up两个小矩阵到megatron的mlp中linear_fc1中的一个大矩阵是直接concat的"""

    def merge(self, tensors: List[Tensor]) -> Tensor:
        return torch.concat(tensors, dim=0)

    def split(self, tensor: Tensor) -> List[Tensor]:
        return list(torch.chunk(tensor, chunks=2, dim=0))


class ExpertUpGateMergeOp(UpGateMergeOp):
    def handle(self, weights: STATE_DICT_T):
        layer_num, expert_num = get_layer_and_expert_num(self.raw_names[0], weights.keys())
        for layer_id in range(layer_num):
            for expert_id in range(expert_num):
                merged_tensor = self.merge([weights.pop(name.replace(DIGIT_FMT, str(layer_id), 1).replace(DIGIT_FMT, str(expert_id), 1)) for name in self.raw_names])
                weights[self.new_name.replace(DIGIT_FMT, str(layer_id), 1).replace(DIGIT_FMT, str(expert_id), 1)] = merged_tensor


class QKVMergeOp(MergeOp):
    """qkv中权重合并逻辑，从transformers的q1q2/k1k2/v1v2三个小矩阵到megatron的linear_qkv一个大矩阵是q1k1v1q2k2v2交织拼接的"""

    def __init__(self, raw_names: QKV_NAME_T, new_name: str, group: int, q_size: int, k_size: int, v_size: int):
        self.group = group
        self.q_size = q_size
        self.k_size = k_size
        self.v_size = v_size
        super().__init__(raw_names, new_name)

    def merge(self, tensors: QKV_NAME_T) -> Tensor:
        return merge_to_interleaved_qkv(*tensors, group=self.group)

    def split(self, tensor: Tensor) -> List[Tensor]:
        qkv = split_to_interleaved_qkv(tensor, self.group, self.q_size, self.k_size, self.v_size)
        return list(qkv)


class QVToQKVMergeOp(MergeOp):
    """构造k权重，并执行qkv中权重合并逻辑，从transformers的q1q2/k1k2/v1v2三个小矩阵到megatron的linear_qkv一个大矩阵是q1k1v1q2k2v2交织拼接的"""

    def __init__(self, raw_names: QV_NAME_T, new_name: str, group: int, q_size: int, k_size: int, v_size: int):
        self.group = group
        self.q_size = q_size
        self.k_size = k_size
        self.v_size = v_size
        super().__init__(raw_names, new_name)

    def merge(self, tensors: List[Tensor]) -> Tensor:
        tensors = [tensors[0], torch.zeros_like(tensors[0]), tensors[1]]
        return merge_to_interleaved_qkv(*tensors, group=self.group)

    def split(self, tensor: Tensor) -> List[Tensor]:
        qkv = split_to_interleaved_qkv(tensor, self.group, self.q_size, self.k_size, self.v_size)
        return [qkv[0], qkv[2]]


class RelocateOp(Operator):
    """处理 QKV 权重在直接拼接和交织结构之间的转换,对于vit，transformers和megatron都是一个大矩阵，但transformer中的qkv是q1q2k1k2v1v2直接拼接，而megatron中是q1k1v1q2k2v2交织拼接"""

    def __init__(self, name: str, new_name: str, group: int, split_size: List[int]):
        """
        Args:
            name: 权重名称模板（如 "decoder.layers.{}.self_attention.linear_qkv.weight"）
            group: 分组数（如多头注意力的头数）
            split_size: 拆分维度 [q_size, k_size, v_size]
        """
        self.name = name
        self.new_name = new_name
        self.group = group
        self.split_size = split_size

    def apply(self, weights: STATE_DICT_T) -> None:
        """正向操作：将直接拼接的 Q/K/V 转换为交织结构"""
        self._apply_transformation(weights, concated_qkv_to_interleaved)

    def revert(self, weights: STATE_DICT_T) -> None:
        self.name = self.new_name
        """反向操作：将交织结构恢复为直接拼接的 Q/K/V"""
        self._apply_transformation(weights, interleaved_qkv_to_concated)

    def _apply_transformation(
            self,
            weights: STATE_DICT_T,
            transform_func: Callable[[Tensor, int, List[int]], Tensor]
    ) -> None:
        """统一处理权重重新排布逻辑"""
        layer_num = get_layer_num(self.name, weights.keys())
        for num in range(layer_num):
            name = self.name.replace(DIGIT_FMT, str(num))
            tensor = weights.get(name)
            if tensor is not None:
                weights[name] = transform_func(tensor, self.group, self.split_size)


class TieOp(Operator):
    """对于tie word embedding，其output layer是直接复用的word_embedding的权重"""

    def __init__(self, raw_name: str, new_name: str):
        self.raw_name = raw_name
        self.new_name = new_name

    def apply(self, weights: STATE_DICT_T):
        weights[self.new_name] = weights.get(self.raw_name)

    def revert(self, weights: STATE_DICT_T) -> None:
        pass


class BaseSplit(ABC):
    """Abstract base class for tensor splitting strategies."""

    @staticmethod
    @abstractmethod
    def split(tp_size: int, value: Tensor) -> List[Tensor]:
        """Split tensor into TP_SIZE partitions.

        Args:
            tp_size: Number of tensor parallel partitions
            value: Tensor to split

        Returns:
            List of split tensors, one per TP rank
        """
        pass

    @staticmethod
    @abstractmethod
    def merge(tp_values: List[Tensor]) -> Tensor:
        """Merge split tensors back into original tensor.

        Args:
            tp_values: List of tensors from TP ranks

        Returns:
            Merged tensor
        """
        pass


class ColWeightSplit(BaseSplit):
    """按列切分权重"""

    @staticmethod
    def split(tp_size: int, value: Tensor) -> List[Tensor]:
        size_per_tp = value.shape[1] // tp_size
        return [
            value[:, rank * size_per_tp:(rank + 1) * size_per_tp]
            for rank in range(tp_size)
        ]

    @staticmethod
    def merge(tp_values: List[Tensor]) -> Tensor:
        return torch.cat(tp_values, dim=1)


class RowBiasSplit(BaseSplit):
    """按行切分偏置"""

    @staticmethod
    def split(tp_size: int, value: Tensor) -> List[Tensor]:
        size_per_tp = value.shape[0] // tp_size
        return [
            value[rank * size_per_tp:(rank + 1) * size_per_tp]
            for rank in range(tp_size)
        ]

    @staticmethod
    def merge(tp_values: List[Tensor]) -> Tensor:
        return torch.cat(tp_values, dim=0)


class RowWeightSplit(BaseSplit):
    """按行切分权重"""

    @staticmethod
    def split(tp_size: int, value: Tensor) -> List[Tensor]:
        size_per_tp = value.shape[0] // tp_size
        return [
            value[rank * size_per_tp:(rank + 1) * size_per_tp, :]
            for rank in range(tp_size)
        ]

    @staticmethod
    def merge(tp_values: List[Tensor]) -> Tensor:
        return torch.cat(tp_values, dim=0)


class GLUSplit(BaseSplit):
    """GLU 通用切分基类"""

    @staticmethod
    def split(tp_size: int, value: Tensor) -> List[Tensor]:
        # 统一处理逻辑，自动适应不同维度
        size_per_tp = value.shape[0] // tp_size // 2
        gate, up = torch.chunk(value, 2, dim=0)

        return [
            torch.cat((
                gate[rank * size_per_tp: (rank + 1) * size_per_tp],
                up[rank * size_per_tp: (rank + 1) * size_per_tp]
            ), dim=0)
            for rank in range(tp_size)
        ]

    @staticmethod
    def merge(tp_values: List[Tensor]) -> Tensor:
        # 统一合并逻辑
        split_results = [torch.chunk(v, 2, dim=0) for v in tp_values]
        all_gates = [chunk[0] for chunk in split_results]
        all_ups = [chunk[1] for chunk in split_results]
        return torch.cat([
            torch.cat(all_gates, dim=0),
            torch.cat(all_ups, dim=0)
        ], dim=0)


def create_qwen2vl_ops(vit_embed_dim: int, vit_num_heads: int, llm_num_query_groups: int,
                       llm_q_size: int, llm_kv_size: int) -> List[Operator]:
    """qwen2vl权重转换逻辑"""
    ops = [
        UpGateMergeOp(raw_names=[r"model.layers.(\d+).mlp.gate_proj.weight", r"model.layers.(\d+).mlp.up_proj.weight"],
                      new_name=r"text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight"),
        QKVMergeOp(raw_names=(r"model.layers.(\d+).self_attn.q_proj.weight",
                              r"model.layers.(\d+).self_attn.k_proj.weight",
                              r"model.layers.(\d+).self_attn.v_proj.weight"),
                   new_name=r"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight",
                   group=llm_num_query_groups,
                   q_size=llm_q_size,
                   k_size=llm_kv_size,
                   v_size=llm_kv_size,
                   ),
        QKVMergeOp(raw_names=(r"model.layers.(\d+).self_attn.q_proj.bias",
                              r"model.layers.(\d+).self_attn.k_proj.bias",
                              r"model.layers.(\d+).self_attn.v_proj.bias"),
                   new_name=r"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.bias",
                   group=llm_num_query_groups,
                   q_size=llm_q_size,
                   k_size=llm_kv_size,
                   v_size=llm_kv_size,
                   ),
        RelocateOp(name=r"visual.blocks.(\d+).attn.qkv.weight",
                   new_name=r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight",
                   group=vit_num_heads,
                   split_size=[vit_embed_dim] * 3,  # vit的qkv不是gqa，所以切分的三份是相同的
                   ),
        RelocateOp(name=r"visual.blocks.(\d+).attn.qkv.bias",
                   new_name=r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias",
                   group=vit_num_heads,
                   split_size=[vit_embed_dim] * 3,  # vit的qkv不是gqa，所以切分的三份是相同的
                   ),
        RenameOp(
            (
                (r'visual.blocks.(\d+).attn.proj',
                 r'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj'),
                (r'visual.blocks.(\d+).attn.qkv',
                 r'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv'),
                (r'visual.blocks.(\d+).mlp.down_proj', r'image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2'),
                (r'visual.blocks.(\d+).mlp.fc', r'image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc'),
                (r'visual.blocks.(\d+).norm1', r'image_encoder.encoder.blocks.layers.(\d+).input_layernorm'),
                (r'visual.blocks.(\d+).norm2', r'image_encoder.encoder.blocks.layers.(\d+).pre_mlp_layernorm'),
                (r'visual.merger.ln_q', r'image_encoder.projector.layernorm'),
                (r'visual.merger.mlp.0', r'image_encoder.projector.encoder.linear_fc1'),
                (r'visual.merger.mlp.2', r'image_encoder.projector.encoder.linear_fc2'),
                (r'visual.patch_embed.proj', r'image_encoder.encoder.patch_embed.proj'),
                (r'model.embed_tokens', r'text_decoder.embedding.word_embeddings'),
                (r'model.layers.(\d+).input_layernorm', r'text_decoder.decoder.layers.(\d+).input_layernorm'),
                (r'model.layers.(\d+).mlp.down_proj', r'text_decoder.decoder.layers.(\d+).mlp.linear_fc2'),
                (
                    r'model.layers.(\d+).post_attention_layernorm',
                    r'text_decoder.decoder.layers.(\d+).pre_mlp_layernorm'),
                (r'model.layers.(\d+).self_attn.o_proj',
                 r'text_decoder.decoder.layers.(\d+).self_attention.linear_proj'),
                (r'lm_head', r'text_decoder.output_layer'),
                (r'model.norm', r'text_decoder.decoder.final_layernorm')
            )
        ),
    ]
    return ops


def create_qwen2_5_vl_ops(vit_embed_dim: int, vit_num_heads: int, llm_num_query_groups: int,
                          llm_q_size: int, llm_kv_size: int) -> List[Operator]:
    """qwen2.5vl在qwen2vl的基础上vit的mlp变成了glu模式、需要增加合并处理逻辑"""
    ops = [
              UpGateMergeOp(
                  raw_names=[r"visual.blocks.(\d+).mlp.gate_proj.weight", r"visual.blocks.(\d+).mlp.up_proj.weight"],
                  new_name=r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight"),
              UpGateMergeOp(
                  raw_names=[r"visual.blocks.(\d+).mlp.gate_proj.bias", r"visual.blocks.(\d+).mlp.up_proj.bias"],
                  new_name=r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias")
          ] + create_qwen2vl_ops(vit_embed_dim, vit_num_heads, llm_num_query_groups, llm_q_size, llm_kv_size)
    return ops


def create_qwen3_vl_ops(vit_embed_dim: int, vit_num_heads: int, llm_num_query_groups: int, llm_q_size: int,
                        llm_kv_size: int) -> List[Operator]:
    """qwen3vl 在qwen2.5vl的基础上增加了moe相关的专家参数，需要增加映射逻辑"""
    ops = [
              RenameOp(
                  (
                      # 处理 MoE专家相关
                      (r'\.mlp\.gate\.', '.mlp.router.'),
                      (r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj',
                       r'text_decoder.decoder.layers.\1.mlp.experts.local_experts.\2.linear_fc2'),
                      # 处理 q, k layernorm
                      (r'\.self_attn\.q_norm\.', '.self_attention.q_layernorm.'),
                      (r'\.self_attn\.k_norm\.', '.self_attention.k_layernorm.'),
                  )
              ),
              ExpertUpGateMergeOp(
                  raw_names=[r"model.layers.(\d+).mlp.experts.(\d+).gate_proj.weight",
                             r"model.layers.(\d+).mlp.experts.(\d+).up_proj.weight"],
                  new_name=r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1.weight"
              )
          ] + create_qwen2_5_vl_ops(vit_embed_dim, vit_num_heads, llm_num_query_groups, llm_q_size, llm_kv_size)
    return ops


def create_qwen2_5_omni_ops(vit_num_heads: int, llm_num_query_groups: int, audio_num_heads: int,
                            head_sizes: Tuple[int, int, int, int]) -> List[Operator]:
    """
    创建qwen2.5-omni的权重转换操作列表

    参数:
        vit_num_heads: ViT中的注意力头数量
        llm_num_query_groups: LLM中的查询组数
        audio_num_heads: 音频处理中的注意力头数量
        head_sizes: 包含四个整数的元组，依次为:
                   - vit_head_hidden_size (ViT中每个头的隐藏层大小)
                   - audio_head_hidden_size (音频处理中每个头的隐藏层大小)
                   - llm_q_size (LLM中查询向量的大小)
                   - llm_kv_size (LLM中键/值向量的大小)
    """
    vit_head_hidden_size, audio_head_hidden_size, llm_q_size, llm_kv_size = head_sizes
    """qwen2.5-omni 权重转换逻辑"""
    ops = [
        UpGateMergeOp(
            raw_names=[r"thinker.visual.blocks.(\d+).mlp.gate_proj.weight",
                       r"thinker.visual.blocks.(\d+).mlp.up_proj.weight"],
            new_name=r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight"),
        UpGateMergeOp(
            raw_names=[r"thinker.visual.blocks.(\d+).mlp.gate_proj.bias",
                       r"thinker.visual.blocks.(\d+).mlp.up_proj.bias"],
            new_name=r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias"),
        UpGateMergeOp(raw_names=[r"thinker.model.layers.(\d+).mlp.gate_proj.weight",
                                 r"thinker.model.layers.(\d+).mlp.up_proj.weight"],
                      new_name=r"text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight"),
        QKVMergeOp(raw_names=(r"thinker.model.layers.(\d+).self_attn.q_proj.weight",
                              r"thinker.model.layers.(\d+).self_attn.k_proj.weight",
                              r"thinker.model.layers.(\d+).self_attn.v_proj.weight"),
                   new_name=r"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight",
                   group=llm_num_query_groups,
                   q_size=llm_q_size,
                   k_size=llm_kv_size,
                   v_size=llm_kv_size,
                   ),
        QKVMergeOp(raw_names=(r"thinker.model.layers.(\d+).self_attn.q_proj.bias",
                              r"thinker.model.layers.(\d+).self_attn.k_proj.bias",
                              r"thinker.model.layers.(\d+).self_attn.v_proj.bias"),
                   new_name=r"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.bias",
                   group=llm_num_query_groups,
                   q_size=llm_q_size,
                   k_size=llm_kv_size,
                   v_size=llm_kv_size,
                   ),
        QKVMergeOp(raw_names=(r"thinker.visual.blocks.(\d+).attn.q.weight",
                              r"thinker.visual.blocks.(\d+).attn.k.weight",
                              r"thinker.visual.blocks.(\d+).attn.v.weight"),
                   new_name=r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight",
                   group=vit_num_heads,
                   q_size=vit_head_hidden_size,
                   k_size=vit_head_hidden_size,
                   v_size=vit_head_hidden_size,
                   ),
        QKVMergeOp(raw_names=(r"thinker.visual.blocks.(\d+).attn.q.bias",
                              r"thinker.visual.blocks.(\d+).attn.k.bias",
                              r"thinker.visual.blocks.(\d+).attn.v.bias"),
                   new_name=r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias",
                   group=vit_num_heads,
                   q_size=vit_head_hidden_size,
                   k_size=vit_head_hidden_size,
                   v_size=vit_head_hidden_size,
                   ),
        QKVMergeOp(raw_names=(r"thinker.audio_tower.layers.(\d+).self_attn.q_proj.weight",
                              r"thinker.audio_tower.layers.(\d+).self_attn.k_proj.weight",
                              r"thinker.audio_tower.layers.(\d+).self_attn.v_proj.weight"),
                   new_name=r"audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight",
                   group=audio_num_heads,
                   q_size=audio_head_hidden_size,
                   k_size=audio_head_hidden_size,
                   v_size=audio_head_hidden_size,
                   ),
        # 音频模型中，k没有bias，所以需要将k的bias以全零权重的形式添加到权重字典，以便进行后续的qkv拼接
        QVToQKVMergeOp(raw_names=(r"thinker.audio_tower.layers.(\d+).self_attn.q_proj.bias",
                                  r"thinker.audio_tower.layers.(\d+).self_attn.v_proj.bias"),
                       new_name=r"audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias",
                       group=audio_num_heads,
                       q_size=audio_head_hidden_size,
                       k_size=audio_head_hidden_size,
                       v_size=audio_head_hidden_size,
                       ),
        RenameOp(
            (
                (r'thinker.visual.blocks.(\d+).attn.proj',
                 r'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj'),
                (r'thinker.visual.blocks.(\d+).mlp.down_proj',
                 r'image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2'),
                (r'thinker.visual.blocks.(\d+).norm1', r'image_encoder.encoder.blocks.layers.(\d+).input_layernorm'),
                (r'thinker.visual.blocks.(\d+).norm2', r'image_encoder.encoder.blocks.layers.(\d+).pre_mlp_layernorm'),
                (r'thinker.visual.merger.ln_q', r'image_encoder.projector.layernorm'),
                (r'thinker.visual.merger.mlp.0', r'image_encoder.projector.encoder.linear_fc1'),
                (r'thinker.visual.merger.mlp.2', r'image_encoder.projector.encoder.linear_fc2'),
                (r'thinker.visual.patch_embed.proj', r'image_encoder.encoder.patch_embed.proj'),
                (r'thinker.model.embed_tokens', r'text_decoder.embedding.word_embeddings'),
                (r'thinker.model.layers.(\d+).input_layernorm', r'text_decoder.decoder.layers.(\d+).input_layernorm'),
                (r'thinker.model.layers.(\d+).mlp.down_proj', r'text_decoder.decoder.layers.(\d+).mlp.linear_fc2'),
                (r'thinker.model.layers.(\d+).post_attention_layernorm',
                 r'text_decoder.decoder.layers.(\d+).pre_mlp_layernorm'),
                (r'thinker.model.layers.(\d+).self_attn.o_proj',
                 r'text_decoder.decoder.layers.(\d+).self_attention.linear_proj'),
                (r'thinker.lm_head', r'text_decoder.output_layer'),
                (r'thinker.model.norm', r'text_decoder.decoder.final_layernorm'),
                (r'thinker.audio_tower.layers.(\d+).fc1', r'audio_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1'),
                (r'thinker.audio_tower.layers.(\d+).fc2', r'audio_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2'),
                (r'thinker.audio_tower.layers.(\d+).final_layer_norm',
                 r'audio_encoder.encoder.blocks.layers.(\d+).pre_mlp_layernorm'),
                (r'thinker.audio_tower.layers.(\d+).self_attn.out_proj',
                 r'audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj'),
                (r'thinker.audio_tower.layers.(\d+).self_attn_layer_norm',
                 r'audio_encoder.encoder.blocks.layers.(\d+).input_layernorm'),
                (r'thinker.audio_tower.ln_post', r'audio_encoder.encoder.ln_post'),
                (r'thinker.audio_tower.proj', r'audio_encoder.encoder.proj'),
                (r'thinker.audio_tower.audio_bos_eos_token', r'audio_encoder.encoder.audio_bos_eos_token'),
                (r'thinker.audio_tower.conv1', r'audio_encoder.encoder.conv1'),
                (r'thinker.audio_tower.conv2', r'audio_encoder.encoder.conv2')
            )
        ),
    ]
    return ops


qwen2vl_tp_patterns = {
    r"text_decoder.output_layer.weight": RowWeightSplit,
    r"text_decoder.embedding.word_embeddings.weight": RowWeightSplit,
    r'text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight': GLUSplit,
    r'text_decoder.decoder.layers.(\d+).mlp.linear_fc2.weight': ColWeightSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight': RowWeightSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.bias': RowBiasSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.linear_proj.weight': ColWeightSplit,
    r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj.weight": ColWeightSplit,
    r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias": RowBiasSplit,
    r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight": RowBiasSplit,
    r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias": RowBiasSplit,
    r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight": RowWeightSplit,
    r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2.weight": ColWeightSplit,
    r"image_encoder.projector.encoder.linear_fc1.bias": RowBiasSplit,
    r"image_encoder.projector.encoder.linear_fc1.weight": RowWeightSplit,
    r"image_encoder.projector.encoder.linear_fc2.weight": ColWeightSplit
}

#  qwen2.5vl的tp切分在qwen2vl的tp切分基础上，修改了vit中mlp的tp切分逻辑，适应glu结构
qwen2_5_vl_tp_patterns = {**qwen2vl_tp_patterns,
                          **{r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias": GLUSplit,
                             r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight": GLUSplit}
                          }

qwen2_5_omni_tp_patterns = {
    **qwen2_5_vl_tp_patterns,
    **{r"audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias": RowBiasSplit,
       r"audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight": RowWeightSplit,
       r"audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj.weight": ColWeightSplit,
       r"audio_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias": RowBiasSplit,
       r"audio_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight": RowWeightSplit,
       r"audio_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2.weight": ColWeightSplit,
       }
}

# qwen3vl的tp切分在qwen2.5vl的tp切分基础上，增加了expert的tp切分逻辑
qwen3_vl_tp_patterns = {
    **qwen2_5_vl_tp_patterns,
    **{
        r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1.weight": GLUSplit,
        r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc2.weight": ColWeightSplit,
    }
}

deepseekvl_tp_patterns = {
    r"text_decoder.output_layer.weight": RowWeightSplit,
    r"text_decoder.embedding.word_embeddings.weight": RowWeightSplit,
    r'text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight': GLUSplit,
    r'text_decoder.decoder.layers.(\d+).mlp.linear_fc2.weight': ColWeightSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.linear_qb.weight': RowWeightSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.linear_kvb.weight': RowWeightSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.linear_kvb.bias': RowBiasSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.linear_proj.weight': ColWeightSplit,
    r'text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1.weight': GLUSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc2.weight": ColWeightSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc1.weight": GLUSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc2.weight": ColWeightSplit
}
