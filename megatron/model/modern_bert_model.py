# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""BERT model."""

import torch
import torch.nn as nn # Added for nn.ModuleList, nn.Parameter
import torch.nn.functional as F # Added for F.gelu, F.softmax
import math # Added for math.sqrt

from megatron import get_args
from megatron.core import tensor_parallel
from megatron.model.enums import AttnMaskType
from megatron.model.language_model import parallel_lm_logits
from megatron.model import LayerNorm
from megatron.model.utils import openai_gelu, erf_gelu
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from .module import MegatronModule

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb
    flash_attn_available = True
except ImportError:
    flash_attn_available = False
    RotaryEmbedding = None
    apply_rotary_emb = None
    print("WARNING: flash_attn is not available. Using PyTorch attention fallback. "
          "Rotary Embedding and Sliding Window Attention will NOT be efficiently applied if flash_attn is missing.")


def bert_extended_attention_mask(attention_mask):
    # We create a 3D attention mask from a 2D tensor mask.
    # [b, 1, s]
    attention_mask_b1s = attention_mask.unsqueeze(1)
    # [b, s, 1]
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    # [b, s, s]
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    # [b, 1, s, s]
    extended_attention_mask = attention_mask_bss.unsqueeze(1)

    # Convert attention mask to binary:
    # True indicates masked, False indicates unmasked (attend to)
    extended_attention_mask = (extended_attention_mask < 0.5)

    return extended_attention_mask

def bert_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids


class BertLMHead(MegatronModule):
    """Masked LM head for Bert

    Arguments:
        config: TransformerConfig object
        mpu_vocab_size: model parallel size of vocabulary.
        hidden_size: hidden size
        parallel_output: whether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, hidden_size, config, parallel_output):
        super().__init__(config=config)

        args = get_args()
        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        tensor_parallel.set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
        self.parallel_output = parallel_output

        self.dense = get_linear_layer(hidden_size, hidden_size, config.init_method, gather_params_on_init=args.zero_stage == 3)
        setattr(self.dense.weight, 'sequence_parallel', config.sequence_parallel)
        setattr(self.dense.bias, 'sequence_parallel', config.sequence_parallel)

        self.layernorm = LayerNorm(hidden_size,
                                   eps=config.layernorm_epsilon,
                                   sequence_parallel=config.sequence_parallel)
        self.gelu = torch.nn.functional.gelu
        if args.openai_gelu:
            self.gelu = openai_gelu
        elif args.onnx_safe:
            self.gelu = erf_gelu

    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        output = parallel_lm_logits(hidden_states,
                                    word_embeddings_weight,
                                    self.parallel_output,
                                    bias=self.bias)
        return output

def post_language_model_processing(lm_output, pooled_output,
                                   lm_head, binary_head,
                                   lm_labels,
                                   logit_weights,
                                   fp16_lm_cross_entropy):
    lm_logits = lm_head(
        lm_output, logit_weights)

    binary_logits = None
    if binary_head is not None:
        binary_logits = binary_head(pooled_output)

    if lm_labels is None:
        # [s b h] => [b s h]
        return lm_logits.transpose(0,1).contiguous(), binary_logits
    else:
        # [b s] => [s b]
        lm_labels = lm_labels.transpose(0,1).contiguous()
        # lm_logits : [s, b, h] and lm_labels: [s, b]
        if fp16_lm_cross_entropy:
            assert lm_logits.dtype == torch.half
            lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits, lm_labels)
        else:
            lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits.float(),
                                                         lm_labels)
        # [s, b] => [b s]
        lm_loss = lm_loss.transpose(0,1).contiguous()
        return lm_loss, binary_logits

class GeGLU(MegatronModule):
    def __init__(self, hidden_size, ffn_hidden_size, config):
        super().__init__(config=config)
        args = get_args()

        self.proj = ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size * 2,
            gather_output=False,
            init_method=config.init_method,
            bias=False,
            sequence_parallel=config.sequence_parallel
        )
        self.output_proj = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=config.output_layer_init_method,
            bias=False,
            sequence_parallel=config.sequence_parallel
        )

    def forward(self, x):
        x_proj = self.proj(x)
        x, gate = x_proj.chunk(2, dim=-1)
        return self.output_proj(x * F.gelu(gate))

class CustomSelfAttention(MegatronModule):
    def __init__(self, hidden_size, num_heads, config, rotary_emb_dim=None, rotary_theta=10000.0, attention_window_size=(-1, -1)):
        super().__init__(config=config)
        args = get_args()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_heads_parallel = num_heads // tensor_parallel.get_model_parallel_world_size()
        self.head_dim = hidden_size // num_heads

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            hidden_size * 3,
            gather_output=False,
            init_method=config.init_method,
            bias=False,
            sequence_parallel=config.sequence_parallel
        )
        self.out_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=config.output_layer_init_method,
            bias=False,
            sequence_parallel=config.sequence_parallel
        )

        self.rotary_emb = None
        if rotary_emb_dim is None:
            rotary_emb_dim = self.head_dim
        
        if args.use_rotary_position_embeddings:
            if flash_attn_available:
                self.rotary_emb = RotaryEmbedding(rotary_emb_dim, theta=rotary_theta)
                print(f"INFO: Rotary Embedding enabled with dimension {rotary_emb_dim} and theta {rotary_theta}.")
            else:
                print("WARNING: flash_attn not available. Rotary Embedding will not be applied.")
        else:
            print("INFO: Rotary Embedding is disabled by args.")

        self.attention_window_size = attention_window_size
        if self.attention_window_size == (-1, -1):
            print("INFO: CustomSelfAttention will use global attention (window_size=(-1,-1)).")
        else:
            print(f"INFO: CustomSelfAttention will use sliding window attention: {self.attention_window_size}")


    def forward(self, x, attention_mask=None):
        # x: [seq_len, batch_size, hidden_size]
        
        qkv = self.qkv_proj(x) # [seq_len, batch_size, 3 * hidden_size_parallel]
        
        # Flash Attentionの入力形式に合わせる: [batch_size, seq_len, 3, num_heads_parallel, head_dim]
        # x.shape[0] は seq_len, x.shape[1] は batch_size
        qkv_reshaped = qkv.view(x.shape[0], x.shape[1], 3, self.num_heads_parallel, self.head_dim)
        qkv_reshaped = qkv_reshaped.permute(1, 0, 2, 3, 4).contiguous() # [batch_size, seq_len, 3, num_heads_parallel, head_dim]

        # Rotary Embeddingの適用
        if self.rotary_emb is not None:
            q, k = qkv_reshaped[:, :, 0], qkv_reshaped[:, :, 1]
            cos, sin = self.rotary_emb(q)
            q, k = apply_rotary_emb(q, k, cos, sin)
            v = qkv_reshaped[:, :, 2]
            qkv_reshaped = torch.stack([q, k, v], dim=2)

        if flash_attn_available:
            if attention_mask is not None:
                # extended_attention_mask is [B, 1, S, S], True for masked.
                # Let's infer the original [B, S] mask where True means valid token (not masked)
                original_2d_attention_mask_bool = ~attention_mask[:, 0, :, 0] # [B, S] boolean, True for valid tokens
                
                seqlens = original_2d_attention_mask_bool.long().sum(dim=1) # [B]
                cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=x.device), seqlens.cumsum(0, dtype=torch.int32)])
                max_seqlen_in_batch = seq_len # Max sequence length in current batch (assuming fixed-size here)
            else:
                # If no attention_mask is provided (e.g., all sequences are full length and valid)
                batch_size = x.shape[1]
                seq_len = x.shape[0]
                cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=x.device)
                max_seqlen_in_batch = seq_len


            # qkv_reshaped is [batch_size, seq_len, 3, num_heads_parallel, head_dim]
            # flash_attn_varlen_qkvpacked_func expects [total_tokens, 3, num_heads_parallel, head_dim]
            qkv_flattened = qkv_reshaped.view(-1, 3, self.num_heads_parallel, self.head_dim)

            attn_output_flattened = flash_attn_varlen_qkvpacked_func(
                qkv_flattened,
                cu_seqlens,
                max_seqlen_in_batch,
                causal=getattr(self.config, 'causal_attention', False), # configからcausal設定を取得 (BERTは通常False)
                window_size=self.attention_window_size,
                return_attn_probs=False # アテンション確率は返さない
            )
            # Flatten された出力を元の形状に戻す [seq_len, batch_size, hidden_size_parallel]
            attn_output = attn_output_flattened.view(seq_len, batch_size, -1)
            
        else:
            q, k, v = qkv_reshaped[:,:,0], qkv_reshaped[:,:,1], qkv_reshaped[:,:,2]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                # extended_attention_mask は [b, 1, s, s] の形式で、True でマスク
                attn_scores = attn_scores.masked_fill(attention_mask.squeeze(1).unsqueeze(1).bool(), -10000.0)
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)

        return self.out_proj(attn_output)


class CustomTransformerLayer(MegatronModule):
    def __init__(self, config, layer_idx,
                 hidden_size, num_attention_heads, ffn_hidden_size):
        super().__init__(config=config)
        
        self.layer_idx = layer_idx # 0-indexed

        self.norm1 = LayerNorm(hidden_size,
                               eps=config.layernorm_epsilon,
                               sequence_parallel=config.sequence_parallel)
        self.norm2 = LayerNorm(hidden_size,
                               eps=config.layernorm_epsilon,
                               sequence_parallel=config.sequence_parallel)

        # アテンション層のタイプとRoPE thetaの決定ロジック
        current_attn_window_size = (-1, -1) # デフォルトはグローバルアテンション
        current_rope_theta = 10000.0 # デフォルトのtheta

        # config.global_attn_every_n_layers が設定されている場合
        if hasattr(config, 'global_attn_every_n_layers') and config.global_attn_every_n_layers > 0:
            if not hasattr(config, 'sliding_window') or config.sliding_window == -1:
                raise ValueError("`global_attn_every_n_layers` requires `sliding_window` to be set (e.g., to 4096).")
            
            # layer_id が global_attn_every_n_layers の倍数でない場合、ローカルアテンション (sliding window)
            if self.layer_idx % config.global_attn_every_n_layers != 0:
                # config.sliding_window は単一の整数を想定 (例: 4096)
                current_attn_window_size = (config.sliding_window // 2, config.sliding_window // 2)
                current_rope_theta = getattr(config, 'local_attn_rope_theta', 10000.0)
                print(f"INFO: Layer {layer_idx} (Local) uses sliding window attention: {current_attn_window_size} with theta {current_rope_theta}")
            else:
                # layer_id が global_attn_every_n_layers の倍数の場合、グローバルアテンション
                current_attn_window_size = (-1, -1) # FlashAttentionでグローバルアテンションを意味する
                current_rope_theta = getattr(config, 'global_attn_rope_theta', 10000.0)
                print(f"INFO: Layer {layer_idx} (Global) uses global attention with theta {current_rope_theta}")
        else:
            # global_attn_every_n_layers が設定されていない場合、すべての層でグローバルアテンション
            current_rope_theta = getattr(config, 'global_attn_rope_theta', 10000.0)
            print(f"INFO: Layer {layer_idx} (Default Global) uses global attention with theta {current_rope_theta}")

        self.attn = CustomSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            config=config,
            rotary_emb_dim=config.rotary_emb_dim if hasattr(config, 'rotary_emb_dim') else None,
            rotary_theta=current_rope_theta,
            attention_window_size=current_attn_window_size # 設定したウィンドウサイズを渡す
        )

        # GeGLU (FFN部分)
        self.ffn = GeGLU(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            config=config
        )

    def forward(self, x, attention_mask):
        # Pre-LN 適用
        attn_input = self.norm1(x)
        attention_output = self.attn(attn_input, attention_mask=attention_mask)
    
        x = x + attention_output
        ffn_input = self.norm2(x)
        ffn_output = self.ffn(ffn_input)
        
        # 残差接続
        x = x + ffn_output

        return x


class CustomBertEncoder(MegatronModule):
    def __init__(self, config, num_layers):
        super().__init__(config=config)
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(
                CustomTransformerLayer(
                    config=config,
                    layer_idx=i,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    ffn_hidden_size=config.ffn_hidden_size
                )
            )
        # Pooler for BERT's NSP head
        self.pooler = None
        if config.add_pooler: # This usually comes from get_language_model's add_pooler arg
             self.pooler = get_linear_layer(config.hidden_size, config.hidden_size, config.init_method,
                                            gather_params_on_init=get_args().zero_stage == 3)
             setattr(self.pooler.weight, 'sequence_parallel', config.sequence_parallel)
             setattr(self.pooler.bias, 'sequence_parallel', config.sequence_parallel)

    def forward(self, hidden_states, attention_mask):
        # Initial hidden_states comes from word_embeddings (shape: [seq_len, batch_size, hidden_size])
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        pooled_output = None
        if self.pooler is not None:
            # Take the first token's (CLS token) representation
            first_token_output = hidden_states[0, :, :] # [batch_size, hidden_size]
            pooled_output = self.pooler(first_token_output)
            pooled_output = torch.tanh(pooled_output)
        
        # Return pooled_output for NSP head, and hidden_states for LM head
        return hidden_states, pooled_output, 0.0 # Moe loss is 0.0 as there are no MoE layers here

    def set_input_tensor(self, input_tensor):
        # For pipeline parallelism
        self.input_tensor = input_tensor

class ModernBertModel(MegatronModule):
    """Bert Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=2,
                 add_binary_head=True,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 return_moe_loss=False):
        super().__init__(config=config)
        args = get_args()

        # TODO this option is not yet implemented in BERT
        assert args.untie_embeddings_and_output_weights is False

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.add_binary_head = add_binary_head
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.return_moe_loss = return_moe_loss

        self.return_embeddings = args.output_bert_embeddings
        if self.return_embeddings:
            assert self.post_process and self.add_binary_head

        config.add_pooler = self.add_binary_head 


        # Using CustomBertEncoder instead of get_language_model
        self.language_model = CustomBertEncoder(
            config=config,
            num_layers=config.num_hidden_layers
        )
        self._language_model_key = 'language_model'

        self.initialize_word_embeddings()
        if self.post_process:
            self.lm_head = BertLMHead(self.shared_embedding_or_output_weight().size(0), config.hidden_size,
                                      config, parallel_output)
            self._lm_head_key = 'lm_head'
            self.binary_head = None
            if self.add_binary_head:
                self.binary_head = get_linear_layer(config.hidden_size, 2,
                                                    config.init_method,
                                                    args.zero_stage == 3)
                self._binary_head_key = 'binary_head'
            self.final_layernorm = LayerNorm(config.hidden_size,
                                             eps=config.layernorm_epsilon,
                                             sequence_parallel=config.sequence_parallel)


    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        # This will be called by pipeline parallelism if enabled
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, bert_model_input, attention_mask,
                tokentype_ids=None, lm_labels=None):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)
        input_ids = bert_model_input
        position_ids = bert_position_ids(input_ids)

        # Word embeddings (managed by MegatronModule)
        word_embeddings = self.word_embeddings(input_ids, position_ids, tokentype_ids)

        # Pass word_embeddings to our custom encoder
        # CustomBertEncoder returns (hidden_states, pooled_output, moe_losses)
        lm_output, pooled_output, moe_losses = self.language_model(
            word_embeddings,
            attention_mask=extended_attention_mask # Pass the extended mask for CustomSelfAttention
        )

        # Apply final LayerNorm to LM output
        lm_output = self.final_layernorm(lm_output)

        if self.post_process:
            # pooled_output is now returned directly from CustomBertEncoder
            # The original BertModel had logic to compute it here based on lm_output[0, :, :]
            # This is now handled by CustomBertEncoder's self.pooler

            if self.return_embeddings:
                # Sum attention mask (original 2D mask, not extended_attention_mask)
                # Here, we assume original `attention_mask` is [B, S] binary (1 for valid, 0 for padded)
                # If you only have `extended_attention_mask` as input, you might need to infer it.
                # For this example, let's assume `attention_mask` still refers to the original [B,S] mask.
                embeddings = torch.transpose(lm_output, 0, 1) # [B, S, H]
                masks = torch.sum(attention_mask, dim=1) # [B] -> count of valid tokens

                # Collect masked embeddings.
                output = torch.zeros(
                    size=(embeddings.shape[0], embeddings.shape[2]),
                    dtype=torch.float32,
                    device=torch.cuda.current_device())
                for i, (embedding, mask) in enumerate(zip(embeddings, masks)):
                    # Average pooling excluding CLS token (index 0) and padding
                    # BERT typically averages all non-padding tokens or uses CLS.
                    # The original `return_embeddings` was mean of [1:mask-1] for CLS.
                    # Adjust if needed for your specific embedding pooling strategy.
                    output[i, :] = torch.mean(embedding[1: mask], dim=0) 

                return output # Only embedding output if return_embeddings is True

            # Process LM logits and binary logits
            lm_output_processed = post_language_model_processing(
                lm_output, pooled_output,
                self.lm_head, self.binary_head,
                lm_labels,
                self.shared_embedding_or_output_weight(),
                self.fp16_lm_cross_entropy
            )
            
            # Return moe_losses if required
            if self.return_moe_loss:
                return (*lm_output_processed, moe_losses) # Combine lm_loss/logits, binary_logits, moe_losses
            else:
                return lm_output_processed # Return lm_loss/logits, binary_logits
        else:
            # If not post_process, just return encoder output
            return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
        if self.post_process:
            state_dict_[self._lm_head_key] \
                = self.lm_head.state_dict_for_save_checkpoint(prefix=prefix,
                                                               keep_vars=keep_vars)
            if self.add_binary_head:
                state_dict_[self._binary_head_key] \
                    = self.binary_head.state_dict(prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] = self.word_embeddings.state_dict(prefix=prefix, keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if self.post_process:
            self.lm_head.load_state_dict(
                state_dict[self._lm_head_key], strict=strict)
        if self.post_process and self.add_binary_head:
            self.binary_head.load_state_dict(
                state_dict[self._binary_head_key], strict=strict)
        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)