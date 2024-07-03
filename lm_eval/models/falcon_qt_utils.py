import math
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F

from transformers.activations import get_activation
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_0
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers.models.falcon.configuration_falcon import FalconConfig
from transformers.models.falcon.modeling_falcon import *
from .quant_utils import *

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "Rocketknight1/falcon-rw-1b"
_CONFIG_FOR_DOC = "FalconConfig"

def pass_hadamard(X):
    import fast_hadamard_transform

    # refer to https://github.com/spcl/QuaRot/blob/6aa2da2c6961b753821820c97783fc10184ca55c/fake_quant/
    dtype = X.dtype
    device = X.device

    n = X.shape[-1]
    input = fast_hadamard_transform.hadamard_transform(X.to(dtype).contiguous(), scale=1/math.sqrt(n))
    
    return input.to(device=device, dtype=dtype).view(X.shape) 


class QuantFalconAttention(nn.Module):
    def __init__(self, config: FalconConfig, bit=2):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self._use_sdpa = config._attn_implementation == "sdpa"

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        if config.rotary:
            self._init_rope()

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        if config.new_decoder_architecture:
            qkv_out_dim = (config.num_kv_heads * 2 + config.num_attention_heads) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        self.query_key_value = FalconLinear(self.hidden_size, qkv_out_dim, bias=config.bias)
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        self.dense = FalconLinear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv_heads = config.num_kv_heads if (self.new_decoder_architecture or not self.multi_query) else 1
        self.bit = bit

    # Copied from transformers.models.llama.modeling_llama.LlamaAttention._init_rope with Llama->Falcon
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = FalconRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = FalconLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = FalconDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        if self.new_decoder_architecture:
            batch, seq_len, _ = fused_qkv.shape
            qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]
            key = torch.broadcast_to(key, query.shape)
            value = torch.broadcast_to(value, query.shape)

            query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
            return query, key, value
        elif not self.multi_query:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        else:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads + 2, self.head_dim)
            return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]

    # Copied from transformers.models.bloom.modeling_bloom.BloomAttention._merge_heads
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size, self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)

        kv_seq_len = key_layer.shape[-2]
        if layer_past is not None:
            kv_seq_len += layer_past[0].shape[-2]
        if alibi is None:
            cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size, self.num_heads, kv_length, head_dim]
            #  - value: [batch_size, self.num_heads, kv_length, head_dim]
            key_layer_trans = pass_hadamard(key_layer)
            value_layer_trans = pass_hadamard(value_layer)

            quantized_key_states = dequantize_per_head(quantize_per_head(key_layer_trans, self.bit))
            quantized_value_states = dequantize_per_head(quantize_per_head(value_layer_trans, self.bit))

            quantized_key_states = pass_hadamard(quantized_key_states)
            quantized_value_states = pass_hadamard(quantized_value_states)

            key_layer = torch.cat((past_key, key_layer), dim=-2)
            value_layer = torch.cat((past_value, value_layer), dim=-2)

            quantized_past_key = torch.concat([past_key, quantized_key_states], dim=2)
            quantized_past_value = torch.concat([past_value, quantized_value_states], dim=2)

            if use_cache:
                present = (quantized_past_key, quantized_past_value)
            else:
                present = None
        else:
            key_layer_trans = pass_hadamard(key_layer)
            value_layer_trans = pass_hadamard(value_layer)

            quantized_past_key = dequantize_per_head(quantize_per_head(key_layer_trans, self.bit))
            quantized_past_value = dequantize_per_head(quantize_per_head(value_layer_trans, self.bit))

            quantized_past_key = pass_hadamard(quantized_past_key)
            quantized_past_value = pass_hadamard(quantized_past_value)

            if use_cache:
                present = (quantized_past_key, quantized_past_value)
            else:
                present = None

        kv_length = key_layer.shape[-2]
        

        if self._use_sdpa and query_layer.device.type == "cuda" and attention_mask is not None:
            # For torch<=2.1.2, SDPA with memory-efficient backend is bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        if alibi is None:
            if self._use_sdpa and not output_attentions:
                # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
                # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
                # The query_length > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not
                # create a causal mask in case query_length == 1.
                is_causal = True if self.is_causal and attention_mask is None and query_length > 1 else False
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )
                attention_scores = None
            else:
                attention_scores = query_layer @ key_layer.transpose(-1, -2)
                attention_scores /= math.sqrt(self.head_dim)

                attention_scores = F.softmax(attention_scores + attention_mask, dim=-1, dtype=hidden_states.dtype)
                # It is unclear why neither dropout nor head_mask is applied here (while it is with alibi).
                attn_output = attention_scores @ value_layer

            attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

            attn_output = self.dense(attn_output)

            if output_attentions:
                return attn_output, present, attention_scores
            else:
                return attn_output, present

        else:
            if self._use_sdpa and not output_attentions and head_mask is None:
                # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
                # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
                is_causal = True if self.is_causal and attention_mask is None and query_length > 1 else False
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=self.attention_dropout.p if self.training else 0.0,
                    is_causal=is_causal,
                )
                attn_output = attn_output.transpose(1, 2)
                attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

                attn_output = self.dense(attn_output)
            else:
                matmul_result = query_layer @ key_layer.transpose(-1, -2)

                # change view to [batch_size, num_heads, q_length, kv_length]
                attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)

                # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
                input_dtype = attention_scores.dtype
                # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
                if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                    attention_scores = attention_scores.to(torch.float32)

                attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)
                attention_logits *= self.inv_norm_factor
                attention_probs = F.softmax(attention_logits + attention_mask, dim=-1, dtype=hidden_states.dtype)
                # [batch_size, num_heads, q_length, kv_length]
                attention_probs = self.attention_dropout(attention_probs)

                if head_mask is not None:
                    attention_probs = attention_probs * head_mask

                # change view [batch_size, num_heads, q_length, kv_length]
                attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)

                # matmul: [batch_size * num_heads, q_length, head_dim]
                attn_output = (attention_probs_reshaped @ value_layer).flatten(0, 1)

                # change view [batch_size, q_length, num_heads * head_dim]
                attn_output = self._merge_heads(attn_output)

                attn_output = self.dense(attn_output)

            if output_attentions:
                return attn_output, present, attention_probs
            else:
                return attn_output, present


class QuantFalconFlashAttention2(QuantFalconAttention):
    """
    Falcon flash attention module. This module inherits from `FalconAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size, self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)

        kv_seq_len = key_layer.shape[-2]
        if layer_past is not None:
            kv_seq_len += layer_past[0].shape[-2]
        if alibi is None:
            cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size, self.num_heads, kv_length, head_dim]
            #  - value: [batch_size, self.num_heads, kv_length, head_dim]
            key_layer_trans = pass_hadamard(key_layer)
            value_layer_trans = pass_hadamard(value_layer)

            quantized_key_states = dequantize_per_head(quantize_per_head(key_layer_trans, self.bit))
            quantized_value_states = dequantize_per_head(quantize_per_head(value_layer_trans, self.bit))

            quantized_key_states = pass_hadamard(quantized_key_states)
            quantized_value_states = pass_hadamard(quantized_value_states)

            key_layer = torch.cat((past_key, key_layer), dim=-2)
            value_layer = torch.cat((past_value, value_layer), dim=-2)

            quantized_past_key = torch.concat([past_key, quantized_key_states], dim=2)
            quantized_past_value = torch.concat([past_value, quantized_value_states], dim=2)

            if use_cache:
                past_key_value = (quantized_past_key, quantized_past_value)
            else:
                past_key_value = None
        else:
            key_layer_trans = pass_hadamard(key_layer)
            value_layer_trans = pass_hadamard(value_layer)

            quantized_past_key = dequantize_per_head(quantize_per_head(key_layer_trans, self.bit))
            quantized_past_value = dequantize_per_head(quantize_per_head(value_layer_trans, self.bit))

            quantized_past_key = pass_hadamard(quantized_past_key)
            quantized_past_value = pass_hadamard(quantized_past_value)

            if use_cache:
                past_key_value = (quantized_past_key, quantized_past_value)
            else:
                past_key_value = None

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_layer = query_layer.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)

        if alibi is not None:
            raise ValueError("`alibi` is not supported when `use_flash_attn` is True")

        attn_dropout = self.config.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_layer.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.query_key_value.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_layer = query_layer.to(target_dtype)
            key_layer = key_layer.to(target_dtype)
            value_layer = value_layer.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_layer, key_layer, value_layer, attention_mask, query_length, dropout=attn_dropout
        )

        attn_weights = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)
        attn_output = self.dense(attn_weights)

        if not output_attentions:
            attn_weights = None

        return attn_output, past_key_value, attn_weights

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class FalconMLP(nn.Module):
    def __init__(self, config: FalconConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.dense_h_to_4h = FalconLinear(hidden_size, config.ffn_hidden_size, bias=config.bias)
        self.act = get_activation(config.activation)
        self.dense_4h_to_h = FalconLinear(config.ffn_hidden_size, hidden_size, bias=config.bias)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


FALCON_ATTENTION_CLASSES = {
    "eager": QuantFalconAttention,
    "sdpa": QuantFalconAttention,  # FalconAttention originally implemented both a forward with & without SDPA
    "flash_attention_2": QuantFalconFlashAttention2,
}


class CustomedFalconDecoderLayer(nn.Module):
    def __init__(self, config: FalconConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.self_attention = FALCON_ATTENTION_CLASSES[config._attn_implementation](config)
        self.mlp = FalconMLP(config)
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        if config.num_ln_in_parallel_attn is None and config.new_decoder_architecture:
            config.num_ln_in_parallel_attn = 2

        if not config.parallel_attn:
            self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        else:
            if config.num_ln_in_parallel_attn == 2:
                # The layer norm before self-attention
                self.ln_attn = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
                # The layer norm before the MLP
                self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            else:
                self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states

        if self.config.new_decoder_architecture and self.config.num_ln_in_parallel_attn == 2:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        # Self attention.
        attn_outputs = self.self_attention(
            attention_layernorm_out,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    attention_output, residual, self.config.attention_dropout, training=self.training
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        if (
            self.config.new_decoder_architecture
            and self.config.parallel_attn
            and self.config.num_ln_in_parallel_attn == 1
        ):
            mlp_layernorm_out = attention_layernorm_out

        outputs = attn_outputs[1:]

        # MLP.
        mlp_output = self.mlp(mlp_layernorm_out)

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output

        output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class CustomedFalconModel(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_alibi = config.alibi

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        self.h = nn.ModuleList([CustomedFalconDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]

        if self.use_alibi:
            mask = (
                torch.ones(
                    (batch_size, seq_length + past_key_values_length), device=inputs_embeds.device, dtype=torch.long
                )
                if attention_mask is None
                else attention_mask
            )
            alibi = build_alibi_tensor(mask, self.num_heads, dtype=hidden_states.dtype)
        else:
            alibi = None
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            if alibi is None:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            elif head_mask is None:
                alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])

                # We don't call _prepare_4d_causal_attention_mask_for_sdpa as we need to mask alibi using the 4D attention_mask untouched.
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )

                # We take care to integrate alibi bias in the attention_mask here.
                min_dtype = torch.finfo(alibi.dtype).min
                attention_mask = torch.masked_fill(
                    alibi / math.sqrt(self.config.hidden_size // self.num_heads),
                    attention_mask < -1,
                    min_dtype,
                )

                # From PyTorch 2.1 onwards, F.scaled_dot_product_attention with the memory-efficient attention backend
                # produces nans if sequences are completely unattended in the attention mask. Details: https://github.com/pytorch/pytorch/issues/110213
                if seq_length > 1 and attention_mask.device.type == "cuda":
                    attention_mask = AttentionMaskConverter._unmask_unattended(attention_mask, min_dtype=min_dtype)
            else:
                # PyTorch SDPA does not support head_mask, we fall back on the eager implementation in this case.
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    layer_past,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@add_start_docstrings(
    "The Falcon Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).",
    FALCON_START_DOCSTRING,
)
class CustomedFalconForCausalLM(FalconPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.transformer = CustomedFalconModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # Note: versions of Falcon with alibi do not use position_ids. It is used with RoPE.
        if not self.transformer.use_alibi and attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in past
        )
        return reordered_past

