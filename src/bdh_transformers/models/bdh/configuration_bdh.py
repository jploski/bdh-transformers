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

from transformers.configuration_utils import PretrainedConfig


class BDHConfig(PretrainedConfig):
    model_type = "bdh"

    def __init__(
        self,
        num_hidden_layers: int = 6,
        hidden_size: int = 256,
        dropout: float = 0.1,
        num_attention_heads: int = 4,
        mlp_internal_dim_multiplier: int = 128,
        vocab_size: int = 256,
        sliding_window: int = None,  # DynamicCache checks for this attribute
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=1,
        use_cache=True,
        attn_implementation=None,  # "bdh_recurrent" or "bdh_parallel"
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads
        self.mlp_internal_dim_multiplier = mlp_internal_dim_multiplier
        self.vocab_size = vocab_size
        self.sliding_window = sliding_window
        self.use_cache = use_cache

        # Use _attn_implementation from parameter, config.json, or default "bdh_recurrent",
        # in that order of preference:
        if attn_implementation is None:
            attn_implementation = kwargs.pop("_attn_implementation", "bdh_recurrent")
        else:
            kwargs.pop("_attn_implementation", None)  # override it with parameter

        kwargs["attn_implementation"] = attn_implementation

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
