# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain BERT"""

from functools import partial

import math
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import ModernBertModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args

try:
    from flash_attn import flash_attn_func
except ImportError:
    print("Warning: flash-attn is not installed. It is required for HybridFlashAttention.")
    flash_attn_func = None
    

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building BERT model ...')

    args = get_args()
    config = core_transformer_config_from_args(args)
    num_tokentypes = 2 if args.bert_binary_head else 0
    model = ModernBertModel(
        config=config,
        num_tokentypes=num_tokentypes,
        parallel_output=True)

    return model


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['text'].long()
    types = data_b['types'].long()
    sentence_order = data_b['is_random'].long()
    loss_mask = data_b['loss_mask'].float()
    lm_labels = data_b['labels'].long()
    padding_mask = data_b['padding_mask'].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask

def data_post_process(data, data_sampler_state_dict):
    args = get_args()
    if args.data_efficiency_curriculum_learning:
        if 'seqlen_truncate' in data_sampler_state_dict['current_difficulties']:
            effective_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_truncate']
        else:
            effective_seqlen = torch.count_nonzero(data['padding_mask'], dim=1)
            effective_seqlen = torch.max(effective_seqlen).to(torch.cuda.current_device())
            torch.distributed.all_reduce(effective_seqlen,
                op=torch.distributed.ReduceOp.MAX,
                group=mpu.get_data_parallel_group())
            effective_seqlen = effective_seqlen.item()
        # Has to be multiple of 8 to enable Tensor Core acceleration
        if effective_seqlen % 8 != 0:
            effective_seqlen = math.ceil(effective_seqlen / 8) * 8
        if effective_seqlen < args.seq_length:
            data['text'] = data['text'][:, :effective_seqlen].contiguous()
            data['types'] = data['types'][:, :effective_seqlen].contiguous()
            data['loss_mask'] = data['loss_mask'][:, :effective_seqlen].contiguous()
            data['labels'] = data['labels'][:, :effective_seqlen].contiguous()
            data['padding_mask'] = data['padding_mask'][:, :effective_seqlen].contiguous()
    return data

# [修正] loss_func の実装を、loss_mask を使う形に書き換えます
def loss_func(loss_mask, output_tensor):
    # モデルが返すトークンごとのlossが output_tensor になります
    lm_loss_per_token = output_tensor.float()
    loss_mask = loss_mask.float()
    
    # loss_maskを使い、本当に学習したい部分のlossだけを合計します
    loss = torch.sum(
        lm_loss_per_token.view(-1) * loss_mask.view(-1)) / loss_mask.sum()
        
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}

def forward_step(data_iterator, model):
    """Forward step for ModernBertModel."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, types, _, loss_mask, lm_labels, padding_mask = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if args.data_efficiency_curriculum_learning:
        args.curriculum_seqlen = tokens.size()[1]
    
    # モデルはトークンごとのloss (output_tensor) を返す
    output_tensor = model(input_ids=tokens,
                          attention_mask=padding_mask, # この引数を追加！
                          tokentype_ids=types,
                          lm_labels=lm_labels)

    # 新しいloss_funcに、loss_maskとモデルの出力を渡す
    return output_tensor, partial(loss_func, loss_mask)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        binary_head=args.bert_binary_head)
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step, args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'},
             data_post_process=data_post_process)
