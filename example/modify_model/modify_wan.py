import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from diffusers.models import WanTransformer3DModel
from diffusers.models.transformers.transformer_wan import WanAttention, _get_qkv_projections, _get_added_kv_projections

from einops import rearrange
import torch.distributed as dist
try:
    from xfuser.core.distributed import get_ulysses_parallel_world_size
    from xfuser.model_executor.layers.usp import _ft_c_input_all_to_all, _ft_c_output_all_to_all
except:
    pass

class SageWanAttnProcessor:
    def __init__(self, attn_func):
        self.attn_func = attn_func
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )
        self.use_sp = False

        if dist.is_initialized() and get_ulysses_parallel_world_size() > 1:
            self.use_sp = True
        self.pad_len = 0

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # ---- transpose to (B, H, N, D) for sageattn/sdpa ----
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            #key_img = key_img.unflatten(2, (attn.heads, -1))
            #value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = self.attn_func(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            '''
            hidden_states_img = self.attn_func(
                query,
                key_img,
                value_img,
                causal=False,
            )
            '''
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            #hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if attn.cross_attention_dim_head is not None: # case for cross attention
            hidden_states = self.attn_func(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            '''
            hidden_states = self.attn_func(
                query,
                key,
                value,
                causal=False,
            )
            '''
        else:
            if self.use_sp:
                #query = rearrange(query, "b s h d" " -> b h s d").contiguous()
                #key = rearrange(key, "b s h d" " -> b h s d").contiguous()
                #value = rearrange(value, "b s h d" " -> b h s d").contiguous()
                query = _ft_c_input_all_to_all(query)
                key = _ft_c_input_all_to_all(key)
                value = _ft_c_input_all_to_all(value)
                #query = rearrange(query, "b h s d" " -> b s h d").contiguous()
                #key = rearrange(key, "b h s d" " -> b s h d").contiguous()
                #value = rearrange(value, "b h s d" " -> b s h d").contiguous()
            if self.pad_len > 0:
                query = query[:, :, : -self.pad_len].contiguous()
                key = key[:, :, : -self.pad_len].contiguous()
                value = value[:, :, : -self.pad_len].contiguous()
            #print(f"query shape = {query.shape}, pad_len = {self.pad_len}")
            hidden_states = self.attn_func(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            '''
            hidden_states = self.attn_func(
                query,
                key,
                value,
                causal=False,
            )
            '''
            if self.use_sp:
                #hidden_states = rearrange(hidden_states.contiguous(), "b s h d -> b h s d").contiguous()
                hidden_states = _ft_c_output_all_to_all(hidden_states)
                #hidden_states = rearrange(hidden_states, "b h s d -> b s h d").contiguous()

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        #hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def set_sage_attn_wan(
        model: WanTransformer3DModel,
        attn_func,
):
    for idx, block in enumerate(model.blocks):
        processor = SageWanAttnProcessor(attn_func)
        block.attn1.processor = processor
