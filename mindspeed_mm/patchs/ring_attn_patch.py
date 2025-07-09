import torch 
from mindspeed.core.context_parallel.ring_context_parallel import AttentionWithCp    
from mindspeed.megatron_adaptor import get_mindspeed_args
from mindspeed.patch_utils import MindSpeedPatchesManager as pm


class AttentionWithCpPatch(AttentionWithCp):
    @classmethod
    def compute_mask(cls, actual_seq_qlen, actual_seq_kvlen, q_block_id, kv_block_id, attn_mask):
        from bisect import bisect_right
        from mindspeed.utils import batch_index

        if actual_seq_qlen:  
            seq_len = actual_seq_qlen[-1] // AttentionWithCp.batch_size
            actual_seq_qlen = batch_index(actual_seq_qlen, seq_len)
            actual_seq_kvlen = batch_index(actual_seq_kvlen, seq_len)
            block_size = cls.block_size
            actual_seq_qlen = [[0] + lst for lst in actual_seq_qlen]
            sub_seq_qlen = [torch.tensor(x[1:]) - torch.tensor(x[:-1]) for x in actual_seq_qlen]
            sub_seq_qid = torch.stack([torch.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_qlen]).npu() # B S

            this_ids = sub_seq_qid[:, q_block_id * block_size:(q_block_id + 1) * block_size].npu()
            this_tile = this_ids.unsqueeze(dim=2) # B S 1

            actual_seq_kvlen = [[0] + lst for lst in actual_seq_kvlen]
            sub_seq_kvlen = [torch.tensor(x[1:]) - torch.tensor(x[:-1]) for x in actual_seq_kvlen]
            sub_seq_kvid = torch.stack([torch.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_kvlen]).npu() # B S
            other_ids = sub_seq_kvid[:, kv_block_id * block_size:(kv_block_id + 1) * block_size].npu()
            other_tile = other_ids.unsqueeze(dim=1) # B 1 S

            mask = this_tile == other_tile # B S S
            
            return torch.logical_not(mask).unsqueeze(dim=1).npu()  # B 1 S S
        else:
            return attn_mask[kv_block_id] if isinstance(attn_mask, list) else None 
        

mindspeed_args = get_mindspeed_args()
if hasattr(mindspeed_args, 'context_parallel_algo') and hasattr(mindspeed_args, 'context_parallel_size'):
    if mindspeed_args.context_parallel_algo == "megatron_cp_algo" and int(mindspeed_args.context_parallel_size) > 1:
        pm.register_patch('mindspeed.core.context_parallel.ring_context_parallel.AttentionWithCp', 
                        AttentionWithCpPatch, force_patch=True)
        pm.apply_patches()