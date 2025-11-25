# Copyright 2025 Jan Ploski and Pathway Technology, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, Union, List, Any

from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, CacheLayerMixin
from .configuration_bdh import BDHConfig


class BDHCacheLayer(CacheLayerMixin):
    """
    A single layer's cache for BDH.
    Instead of K/V tensors, it holds the Linear Attention Recurrent State.
    """

    def __init__(self):
        super().__init__()
        # We ignore self.keys and self.values from the parent class.
        # We use our own storage:
        self.recurrent_state: Optional[torch.Tensor] = None
        self.cumulative_length = 0  # Tracks tokens seen by this layer
        self.is_initialized = True  # Mark ready immediately

    def lazy_initialization(self, key_states: torch.Tensor):
        # Not needed for Recurrent state as we init on the fly,
        # but required by abstract base class.
        pass

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Called by Cache.update().
        We use this purely to track sequence length.
        The actual state update happens via explicit assignment later.
        """
        self.cumulative_length += key_states.shape[-2]

        # Return inputs as-is to satisfy interface (we don't store K/V)
        return key_states, value_states

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1  # Infinite context

    def get_mask_sizes(self, cache_position: torch.Tensor) -> Tuple[int, int]:
        # Standard logic for Dynamic/Infinite caches
        return cache_position.shape[0], 0

    # --- Overrides for State Management (Replacing K/V logic) ---

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders the recurrent state for beam search."""
        if self.recurrent_state is not None:
            # self.recurrent_state: [B, nh, N, D]
            self.recurrent_state = self.recurrent_state.index_select(
                0, beam_idx.to(self.recurrent_state.device)
            )

    def reset(self) -> None:
        """Resets the state."""
        self.recurrent_state = None
        self.cumulative_length = 0

    def offload(self):
        """Offload state to CPU."""
        if self.recurrent_state is not None:
            self.recurrent_state = self.recurrent_state.to("cpu", non_blocking=True)

    def prefetch(self):
        """Bring state back to GPU."""
        # We assume 'self.device' is set or we can infer it.
        # In many HF implementations, layers check keys.device.
        # Since we don't have keys, we might need to handle device tracking carefully.
        # For now, this is a placeholder as `device` isn't always robustly stored in Mixin.
        pass


class BDHCache(Cache):
    """
    The main Cache object passed to model.forward().
    It manages a list of BDHCacheLayer objects.
    """

    def __init__(
        self,
        config,
        max_batch_size=None,
        max_cache_len=None,
        device=None,
        dtype=torch.float32,
    ):
        # 1. Call super with our custom layer class
        # This automatically creates self.layers = [] and handles lazy instantiation
        super().__init__(layer_class_to_replicate=BDHCacheLayer)

        self.config = config
        self.dtype = dtype

        # Store params for property accessors
        self._max_batch_size = max_batch_size
        self._max_cache_len = max_cache_len

    def detach_(self):
        """Detaches all recurrent states from the computation graph."""
        for layer in self.layers:
            if layer.recurrent_state is not None:
                layer.recurrent_state = layer.recurrent_state.detach()

    def update_state(self, layer_idx: int, new_state: torch.Tensor):
        """
        Accesses the specific layer and updates its recurrent state.
        The 'self.layers' list is populated automatically by super().update()
        when the model runs, OR we force creation if needed.
        """
        # Ensure layer exists (rare edge case if update_state called before update)
        while len(self.layers) <= layer_idx:
            self.layers.append(BDHCacheLayer())

        self.layers[layer_idx].recurrent_state = new_state.to(torch.float32)

    def get_state(self, layer_idx: int) -> Optional[torch.Tensor]:
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].recurrent_state
        return None

    @property
    def max_batch_size(self):
        return self._max_batch_size

    @property
    def max_cache_len(self):
        return self._max_cache_len

    @property
    def is_sliding(self):
        return False


# Note (TODO?) the recurrent sums calculated here are potentially numerically unstable for big N and T
# as they presently lack a denominator/scaling term. Some further references:
# https://arxiv.org/pdf/2006.16236
# https://haileyschoelkopf.github.io/blog/2024/linear-attn/
#
class BDHRecurrentAttention(nn.Module):
    """
    Implementation of stateful linear(ized) attention, for "infinite" context length.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.num_attention_heads
        D = config.hidden_size
        N = config.mlp_internal_dim_multiplier * D // nh

        self.register_buffer(
            "freqs",
            self.get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N),
            persistent=False,
        )

    @staticmethod
    def get_freqs(n, theta, dtype):
        def quantize(t, q=2):
            return (t / q).floor() * q

        return (
            1.0
            / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
            / (2 * math.pi)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        # v is expected to be FP32 here
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = BDHRecurrentAttention.phases_cos_sin(phases)
        return (v * phases_cos) + (v_rot * phases_sin)

    def forward(
        self,
        Q_raw: torch.Tensor,
        V_raw: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[object] = None,
    ):
        target_dtype = Q_raw.dtype

        # Upcast to float32 for better numerical stability
        Q_raw = Q_raw.to(dtype=torch.float32)
        V_raw = V_raw.to(dtype=torch.float32)

        B, nh, T, N = Q_raw.size()
        D = V_raw.size(-1)
        assert T != 0

        # 1. RoPE
        if position_ids is not None:
            assert position_ids.shape[-1] != 0
            r_phases = (
                position_ids.unsqueeze(1).unsqueeze(-1).to(self.freqs.dtype)
                * self.freqs
            )
        else:
            r_phases = (
                torch.arange(T, device=Q_raw.device)
                .view(1, 1, -1, 1)
                .to(self.freqs.dtype)
                * self.freqs
            )

        QR = self.rope(r_phases, Q_raw)
        KR = QR

        # 2. V Expansion
        if V_raw.dim() == 3:
            V_to_cache = V_raw.unsqueeze(1).expand(-1, nh, -1, -1)
        else:
            V_to_cache = V_raw.expand(-1, nh, -1, -1)

        # 3. Update Cache Counter (critical for correct RoPE in future steps)
        # We pass KR, V_to_cache, but BDHCache ignores them and just increments a counter.
        if past_key_values is not None:
            past_key_values.update(KR, V_to_cache, layer_idx)

        # 4. Execute

        # A. PREFILL (Chunked Linear Attention)
        if T > 1:
            out, final_state = self._forward_prefill_chunked(
                QR, KR, V_to_cache, attention_mask, T, past_key_values, layer_idx
            )
            # Save the final state of the prompt so generation can continue from it
            if isinstance(past_key_values, BDHCache):
                past_key_values.update_state(layer_idx, final_state)

        # B. GENERATION (Recurrent Step)
        else:
            # Retrieve state
            prev_state = None
            if isinstance(past_key_values, BDHCache):
                prev_state = past_key_values.get_state(layer_idx)

            if prev_state is None:
                prev_state = torch.zeros(
                    (B, nh, N, D), device=QR.device, dtype=torch.float32
                )

            # 1. Output Calculation (Q * S)
            out = torch.matmul(QR, prev_state)

            # 2. State Update (S + K^T * V) - Background
            # Note: We must update the state for the NEXT token.
            # Unlike prefill, we do this token-by-token here.

            k_t = KR.transpose(-1, -2)  # [B, nh, N, 1]
            v_t = V_to_cache  # [B, nh, 1, D]

            if attention_mask is not None:
                # attention_mask is [B, T] or [B, 1, 1, T] -> extract last column
                # Ensure dimensions align for broadcasting: [B, 1, 1, 1]
                mask_gen = attention_mask[:, -1].view(B, 1, 1, 1).to(k_t.dtype)
                k_t = k_t * mask_gen

            state_update = torch.matmul(k_t, v_t)  # [B, nh, N, D]

            new_state = prev_state + state_update

            if isinstance(past_key_values, BDHCache):
                past_key_values.update_state(layer_idx, new_state)

        return out.to(target_dtype)

    def _forward_prefill_chunked(
        self, QR, KR, V, attention_mask, T, past_key_values, layer_idx
    ):
        """
        Efficient Chunked Linear Attention.
        Returns:
          1. Output [B, nh, T, D]
          2. Final State [B, nh, N, D] (to be saved to cache)
        """
        B, nh, S, N = KR.shape
        D = V.shape[-1]
        CHUNK_SIZE = 128

        # Retrieve existing state if we are continuing a sequence (unlikely for pure prefill, but good for safety)
        running_state = None
        if isinstance(past_key_values, BDHCache):
            running_state = past_key_values.get_state(layer_idx)

        if running_state is None:
            running_state = torch.zeros(
                (B, nh, N, D), device=QR.device, dtype=torch.float32
            )
        else:
            running_state = running_state.to(dtype=torch.float32)

        # Prepare Mask
        mask_map = None
        if attention_mask is not None:
            # Expand standard mask to 4D if needed
            if attention_mask.dim() == 2:
                mask_map = attention_mask[:, None, :, None]
            elif attention_mask.dim() == 4:
                # If causal mask [B, 1, T, T], we extract diagonal/validity
                mask_map = attention_mask[:, :, -1:, :].transpose(-1, -2)
            else:
                mask_map = attention_mask

            if mask_map.shape[2] >= S:
                mask_map = mask_map[:, :, -S:, :]

            mask_map = mask_map.to(dtype=torch.float32)

        output_chunks = []

        for i in range(0, S, CHUNK_SIZE):
            idx_end = min(i + CHUNK_SIZE, S)
            chunk_len = idx_end - i

            q_chunk = QR[..., i:idx_end, :]
            k_chunk = KR[..., i:idx_end, :]
            v_chunk = V[..., i:idx_end, :]

            # Apply Padding Mask to K/V only (Q is masked by causal logic implicitly via state)
            if mask_map is not None:
                m_chunk = mask_map[..., i:idx_end, :]
                k_chunk = k_chunk * m_chunk
                v_chunk = v_chunk * m_chunk

            # 1. Inter-chunk: Attention from History (Q * State_{t-1})
            out_inter = torch.matmul(q_chunk, running_state)

            # 2. Intra-chunk: Standard Causal Attention within the chunk
            #    (Uses local Q * K^T * V)
            attn_scores = torch.matmul(k_chunk, k_chunk.transpose(-1, -2))

            causal_mask = torch.tril(
                torch.ones((chunk_len, chunk_len), device=QR.device, dtype=torch.bool),
                diagonal=-1,  # Strict causal (diagonal is 0) matches recurrent Q*S_prev
            )
            attn_scores = attn_scores.masked_fill(~causal_mask, 0.0)

            out_intra = torch.matmul(attn_scores, v_chunk)

            # Sum components
            chunk_out = out_inter + out_intra
            output_chunks.append(chunk_out)

            # 3. Update State for next chunk
            k_t = k_chunk.transpose(-1, -2)
            state_update = torch.matmul(k_t, v_chunk)
            running_state = running_state + state_update

        return torch.cat(output_chunks, dim=2), running_state


class BDHParallelAttention(nn.Module):
    """
    Implementation of stateless fixed-length linear(ized) attention,
    supporting "as much context length as fits into GPU memory".
    This can be utilized together with config.sliding_window > 0 for even longer context,
    but the attention won't be able to process tokens that slide out. Therefore, it is included
    as reference implementation only, superseded by BDHRecurrentAttention.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.num_attention_heads
        D = config.hidden_size
        # N corresponds to the Key/Query dimension (sparse)
        N = config.mlp_internal_dim_multiplier * D // nh

        self.register_buffer(
            "freqs",
            self.get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N),
            persistent=False,
        )

    @staticmethod
    def get_freqs(n, theta, dtype):
        def quantize(t, q=2):
            return (t / q).floor() * q

        return (
            1.0
            / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
            / (2 * math.pi)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        # v is expected to be FP32 here
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = BDHParallelAttention.phases_cos_sin(phases)
        return (v * phases_cos) + (v_rot * phases_sin)

    def forward(
        self,
        Q_raw: torch.Tensor,
        V_raw: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[object] = None,
    ):
        target_dtype = Q_raw.dtype  # Save original dtype (e.g. bfloat16)

        Q_raw = Q_raw.to(torch.float32)
        V_raw = V_raw.to(torch.float32)

        B, nh, T, N = Q_raw.size()

        # 1. RoPE Setup
        if position_ids is not None:
            r_phases = (
                position_ids.unsqueeze(1).unsqueeze(-1).to(self.freqs.dtype)
                * self.freqs
            )
        else:
            r_phases = (
                torch.arange(T, device=Q_raw.device)
                .view(1, 1, -1, 1)
                .to(self.freqs.dtype)
                * self.freqs
            )

        # 2. Apply RoPE (in FP32)
        QR = self.rope(r_phases, Q_raw)
        KR = QR

        # 3. Update KV Cache & handle dimensions
        # V_raw is [B, T, D] or [B, 1, T, D].
        if V_raw.dim() == 3:
            V_to_cache = V_raw.unsqueeze(1).expand(-1, nh, -1, -1)
        else:
            V_to_cache = V_raw.expand(-1, nh, -1, -1)

        # Update cache with FP32 tensors
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(KR, V_to_cache, layer_idx)
        else:
            key_states = KR
            value_states = V_to_cache

        # 4. Hybrid Execution Strategy
        if T <= 2048:
            out = self._forward_standard(
                QR, key_states, value_states, attention_mask, T
            )
        else:
            out = self._forward_prefill_chunked(
                QR, key_states, value_states, attention_mask, T
            )

        return out.to(target_dtype)

    def _forward_standard(self, QR, key_states, value_states, attention_mask, T):
        """
        Standard quadratic attention running in FP32.
        Faster for small T. Memory complexity: O(T^2) = quadratic, time complexity: O(T^2 * N) = quadratic.
        """
        # [B, nh, T, N] @ [B, nh, N, S] -> [B, nh, T, S]
        # Calculations here happen in FP32 automatically because inputs are FP32
        scores = torch.matmul(QR, key_states.transpose(-1, -2))

        if T > 1:
            total_len = key_states.size(2)
            mask = torch.tril(
                torch.ones((T, total_len), device=scores.device, dtype=torch.bool),
                diagonal=total_len - T - 1,
            )
            scores = scores.masked_fill(~mask, 0.0)
        else:
            # Inference (T=1): Mask the last position (itself)
            scores[..., -1:] = 0.0

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_expanded = attention_mask[:, None, None, :]
            else:
                mask_expanded = attention_mask
            if mask_expanded.shape[-1] > key_states.shape[2]:
                mask_expanded = mask_expanded[..., -key_states.shape[2] :]
            scores = scores.masked_fill(mask_expanded == 0, 0.0)

        output = torch.matmul(scores, value_states)
        return output

    def _forward_prefill_chunked(self, QR, key_states, value_states, attention_mask, T):
        """
        Chunked linear attention.
        Slower for small T. Memory complexity: O(N * D) = const, time complexity: O(T * N * D) = linear scaling.
        """
        B, nh, S, N = key_states.shape
        D = value_states.shape[-1]
        CHUNK_SIZE = 128

        mask_map = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_map = attention_mask[:, None, :, None]
            else:
                mask_map = attention_mask
            if mask_map.shape[2] > S:
                mask_map = mask_map[:, :, -S:, :]
            elif mask_map.shape[2] < S:
                pass
            # Ensure mask is FP32
            mask_map = mask_map.to(dtype=torch.float32)

        running_state = torch.zeros(
            (B, nh, N, D), device=QR.device, dtype=torch.float32
        )
        output_chunks = []

        query_start_offset = S - T

        for i in range(0, S, CHUNK_SIZE):
            idx_end = min(i + CHUNK_SIZE, S)
            chunk_len = idx_end - i

            k_chunk = key_states[..., i:idx_end, :]
            v_chunk = value_states[..., i:idx_end, :]

            if mask_map is not None:
                m_chunk = mask_map[..., i:idx_end, :]
                k_chunk = k_chunk * m_chunk
                v_chunk = v_chunk * m_chunk

            q_start_in_seq = max(i, query_start_offset)

            if q_start_in_seq < idx_end:
                q_local_start = q_start_in_seq - query_start_offset
                q_local_end = idx_end - query_start_offset
                q_chunk = QR[..., q_local_start:q_local_end, :]

                # 1. Inter-chunk (FP32)
                out_inter = torch.matmul(q_chunk, running_state)

                # 2. Intra-chunk (Now strictly FP32)
                attn_scores = torch.matmul(k_chunk, k_chunk.transpose(-1, -2))

                causal_mask = torch.tril(
                    torch.ones(
                        (chunk_len, chunk_len), device=QR.device, dtype=torch.bool
                    ),
                    diagonal=-1,
                )
                attn_scores = attn_scores.masked_fill(~causal_mask, 0.0)
                out_intra = torch.matmul(attn_scores, v_chunk)

                rel_start = q_start_in_seq - i
                out_intra_sliced = out_intra[..., rel_start:, :]

                chunk_out = out_inter + out_intra_sliced
                # Output chunk is FP32
                output_chunks.append(chunk_out)

            # 3. Update State (FP32)
            k_t = k_chunk.transpose(-1, -2)
            v_t = v_chunk

            state_update = torch.matmul(k_t, v_t)
            running_state = running_state + state_update

            del k_chunk, v_chunk, state_update, k_t, v_t

        return torch.cat(output_chunks, dim=2)


class BDHPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading
    and loading pretrained models.
    """

    config_class = BDHConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class BDHModel(BDHPreTrainedModel):
    """
    The bare BDH model outputting raw hidden-states without any specific head on top.
    """

    config_class = BDHConfig

    def __init__(self, config: BDHConfig):
        super().__init__(config)
        self.config = config
        nh = config.num_attention_heads
        D = config.hidden_size
        N = config.mlp_internal_dim_multiplier * D // nh

        # Shared weights for the recurrent-style architecture
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        if config._attn_implementation_internal == "bdh_recurrent":
            self.attn = BDHRecurrentAttention(config)
        elif config._attn_implementation_internal == "bdh_parallel":
            self.attn = BDHParallelAttention(config)
        else:
            raise ValueError(
                f"Unsupported attn_implementation {config._attn_implementation_internal}"
            )

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)

        B, T, D = inputs_embeds.size()

        if use_cache and past_key_values is None:
            past_key_values = BDHCache(
                config=self.config,
                max_batch_size=B,
                max_cache_len=0,  # Infinite
                device=inputs_embeds.device,
                dtype=torch.float32,
            )

        if position_ids is None:
            if attention_mask is not None:
                # Infer from mask
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if position_ids.shape[-1] > T:
                    position_ids = position_ids[:, -T:]
            else:
                # Infer from Cache global counter
                # DynamicSlidingWindowLayer.get_seq_length() returns cumulative_length (Total Seen)
                # This ensures RoPE continues counting up even if cache is shifted and truncated.
                if past_key_values is not None:
                    # We use the length of the first layer as proxy for global state
                    start_pos = past_key_values.get_seq_length(0)
                    position_ids = (
                        torch.arange(
                            start_pos, start_pos + T, device=inputs_embeds.device
                        )
                        .unsqueeze(0)
                        .expand(B, -1)
                    )
                else:
                    position_ids = (
                        torch.arange(T, device=inputs_embeds.device)
                        .unsqueeze(0)
                        .expand(B, -1)
                    )

        x = inputs_embeds.unsqueeze(1)
        x_seq = self.ln(x)

        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = (x_seq.view(B, T, D),)

        for level in range(self.config.num_hidden_layers):
            x_latent = torch.einsum("btd,hdn->bhtn", x_seq.squeeze(1), self.encoder)
            x_sparse = F.relu(x_latent)

            # Pass layer_idx so DynamicCache knows which layer to update
            yKV = self.attn(
                Q_raw=x_sparse,
                V_raw=x_seq,
                layer_idx=level,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            nh = self.config.num_attention_heads
            N = self.config.mlp_internal_dim_multiplier * D // nh

            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x_seq = self.ln(x_seq + y)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x_seq.view(B, T, D),)

        last_hidden_state = x_seq.view(B, T, D)

        if not return_dict:
            output = (last_hidden_state,)
            if use_cache:
                output += (past_key_values,)
            if output_hidden_states:
                output += (all_hidden_states,)
            return output

        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class BDHForCausalLM(BDHPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    """
    The BDH model with a language head on top.
    """

    def __init__(self, config: BDHConfig):
        super().__init__(config)
        self.backbone = BDHModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing.
        # Note: following HF conventions, by default it ties word embeddings to lm_head;
        # this can be disabled by setting config.tie_word_embeddings = False
        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if kwargs.get("use_cache", True) and isinstance(
            self.backbone.attn, BDHRecurrentAttention
        ):
            if past_key_values is None or not isinstance(past_key_values, BDHCache):
                # Hack: if we're in stateful mode, force-replace the default DynamicCache passed down to us
                # with our own; I didn't find a better method of convincing HF to create the correct cache type.
                past_key_values = BDHCache(
                    self.config,
                    max_batch_size=input_ids.shape[0],
                    max_cache_len=0,  # Infinite context
                    device=input_ids.device
                    if input_ids is not None
                    else inputs_embeds.device,
                    dtype=torch.float32,
                )

        if past_key_values is not None:
            # BDHCache returns cumulative_length, which is the total number of tokens processed.
            # Use this to slice input_ids correctly.
            cache_length = past_key_values.get_seq_length()

            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:]
            elif input_ids.shape[1] > cache_length:
                input_ids = input_ids[:, cache_length:]
            else:
                input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def get_input_embeddings(self):
        return self.backbone.embed

    def set_input_embeddings(self, value):
        self.backbone.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
