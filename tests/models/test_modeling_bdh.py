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

import unittest
import torch

from transformers.testing_utils import require_torch, torch_device

from bdh_transformers.models.bdh.configuration_bdh import BDHConfig
from bdh_transformers.models.bdh.modeling_bdh import BDHForCausalLM, BDHParallelAttention, BDHRecurrentAttention

class BDHStandaloneTest(unittest.TestCase):
    """
    Unit tests for the BDHForCausalLM model.
    """
    
    def _get_config(self, **kwargs):
        """
        Creates a larger, more realistic model configuration for testing.
        Numerical errors tend to accumulate more with a higher number of parameters.
        """
        config = BDHConfig(
            vocab_size=100,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            mlp_internal_dim_multiplier=256,
            pad_token_id=0,
            eos_token_id=1,
            max_position_embeddings=4096,
            attn_implementation=kwargs.get("attn_implementation", None),
        )
        # Allows overriding of default config values for specific tests
        config.update(kwargs)
        return config

    @require_torch
    def test_model_creation(self):
        """Checks if the model initializes correctly and can run a forward pass."""
        config = self._get_config()
        model = BDHForCausalLM(config).to(torch_device)
        model.eval()
        
        # Create random input tensor
        input_ids = torch.randint(2, config.vocab_size, (1, 5), device=torch_device)
        outputs = model(input_ids)
        
        self.assertEqual(outputs.logits.shape, (1, 5, config.vocab_size))

    @require_torch
    def test_use_cache_consistency(self):
        """
        Ensures that token-by-token generation with the KV cache produces the
        exact same output sequence as generating the full sequence at once. This
        is tested over a longer generation to check for potential divergence.
        """
        config = self._get_config()
        model = BDHForCausalLM(config).to(torch_device)
        model.eval()

        input_ids = torch.randint(2, config.vocab_size, (1, 10), device=torch_device)
        
        gen_kwargs = {
            "max_new_tokens": 50,
            "do_sample": False,
            "pad_token_id": config.pad_token_id,
            "eos_token_id": config.eos_token_id,
        }

        # 1. Generate the full sequence without cache to serve as the ground truth.
        out_no_cache = model.generate(input_ids, use_cache=False, **gen_kwargs)

        # 2. Generate token-by-token with the cache enabled.
        next_input_ids = input_ids
        past_key_values = None
        generated_tokens = []

        for _ in range(gen_kwargs["max_new_tokens"]):
            with torch.no_grad():
                outputs = model(
                    input_ids=next_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_tokens.append(next_token.item())
            
            next_input_ids = next_token.unsqueeze(-1)
            past_key_values = outputs.past_key_values

        out_with_cache_manual = torch.cat(
            [input_ids, torch.tensor([generated_tokens], device=torch_device)], 
            dim=-1
        )
        
        # 3. The results from both methods should be identical.
        torch.testing.assert_close(
            out_no_cache, 
            out_with_cache_manual, 
            msg="Mismatch between single-shot and manual token-by-token generation."
        )

    @require_torch
    def test_attention_implementation_consistency(self):
        """
        Checks if the 'bdh_parallel' and 'bdh_recurrent' attention 
        implementations produce numerically close results.
        """
        # Use a fixed seed to ensure both models are initialized with the same weights.
        torch.manual_seed(0)
        model_parallel = BDHForCausalLM(
            self._get_config(attn_implementation="bdh_parallel")
        ).to(torch_device)
        model_parallel.eval()
        self.assertTrue(isinstance(model_parallel.backbone.attn, BDHParallelAttention))

        torch.manual_seed(0)
        model_recurrent = BDHForCausalLM(
            self._get_config(attn_implementation="bdh_recurrent")
        ).to(torch_device)
        model_recurrent.eval()
        self.assertTrue(isinstance(model_recurrent.backbone.attn, BDHRecurrentAttention))

        input_ids = torch.randint(2, self._get_config().vocab_size, (2, 40), device=torch_device)

        with torch.no_grad():
            outputs_parallel = model_parallel(input_ids)
            outputs_recurrent = model_recurrent(input_ids)
        
        # The logits should be nearly identical, within a small tolerance for FP32 variations.
        torch.testing.assert_close(
            outputs_parallel.logits, 
            outputs_recurrent.logits,
            atol=1e-5,
            rtol=1e-4,
            msg="Logits from 'bdh_parallel' and 'bdh_recurrent' implementations do not match."
        )

    @require_torch
    def test_long_sequence_chunked_prefill_consistency(self):
        """
        Tests consistency between a single prefill for a sequence > 2048 tokens
        (which triggers a special chunked attention path) and a sequential prefill
        that builds the cache in multiple steps.
        """
        config = self._get_config()
        model = BDHForCausalLM(config).to(torch_device)
        model.eval()

        seq_len = 2100
        prefill_chunk_len = 2048 # The threshold for the chunked attention path.
        
        long_input_ids = torch.randint(2, config.vocab_size, (1, seq_len), device=torch_device)

        # 1. Full prefill path (T > 2048), which should trigger chunked attention.
        with torch.no_grad():
            outputs_full_prefill = model(long_input_ids)
        logits_full_prefill = outputs_full_prefill.logits

        # 2. Sequential prefill path. First, process a chunk <= 2048 tokens.
        with torch.no_grad():
            outputs_chunk1 = model(
                long_input_ids[:, :prefill_chunk_len], 
                use_cache=True
            )
        
        # Then, process the remaining chunk using the cache from the first pass.
        with torch.no_grad():
            outputs_chunk2 = model(
                long_input_ids[:, prefill_chunk_len:],
                past_key_values=outputs_chunk1.past_key_values,
                use_cache=True
            )
        
        # The concatenated logits from the sequential run should match the single run.
        logits_sequential_prefill = torch.cat(
            [outputs_chunk1.logits, outputs_chunk2.logits], 
            dim=1
        )

        # 3. Compare the logits from both paths.
        torch.testing.assert_close(
            logits_full_prefill,
            logits_sequential_prefill,
            atol=1e-5,
            rtol=1e-4,
            msg="Mismatch between single chunked prefill (>2048) and sequential prefill."
        )

    @require_torch
    def test_beam_search_generation_with_cache(self):
        """
        Tests if generation with beam search works correctly, as this
        triggers the `BDHCache.reorder_cache` method.
        """
        config = self._get_config()
        model = BDHForCausalLM(config).to(torch_device)
        model.eval()

        input_ids = torch.randint(2, config.vocab_size, (1, 5), device=torch_device)

        # Generate with beam search
        with torch.no_grad():
            output_beam = model.generate(
                input_ids,
                max_new_tokens=5,
                num_beams=4,
                do_sample=False,
                num_return_sequences=1,
            )
        
        # 1. Check that it runs without error and the shape is correct
        self.assertEqual(output_beam.shape, (1, 10))

        # 2. Check that the output is different from greedy search (highly likely with beams)
        with torch.no_grad():
            output_greedy = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
            )
        
        self.assertFalse(
            torch.equal(output_beam, output_greedy),
            "Beam search output should differ from greedy search output."
        )


    @require_torch
    def test_model_can_be_trained(self):
        """
        Verifies that the model can perform a forward and backward pass to compute
        gradients, ensuring it is trainable.
        """
        config = self._get_config()
        model = BDHForCausalLM(config).to(torch_device)
        model.train() # Set model to training mode

        # Create dummy inputs and labels
        input_ids = torch.randint(2, config.vocab_size, (2, 10), device=torch_device)
        labels = torch.randint(2, config.vocab_size, (2, 10), device=torch_device)

        # Forward pass to get loss
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        self.assertIsNotNone(loss)
        
        # Backward pass
        loss.backward()

        # Check if a key parameter has gradients
        # The 'decoder' is a core, unique parameter in your architecture
        self.assertIsNotNone(model.backbone.decoder.grad)
        self.assertGreater(
            torch.abs(model.backbone.decoder.grad).sum(), 
            0,
            "Gradients for the decoder parameter should not be zero."
        )

    @require_torch
    def test_long_generation_numerical_stability(self):
        """
        Stress-tests the recurrent attention for numerical stability (NaN/inf values)
        over a very long generation sequence.
        """
        config = self._get_config(attn_implementation="bdh_recurrent")
        model = BDHForCausalLM(config).to(torch_device)
        model.eval()

        input_ids = torch.randint(2, config.vocab_size, (1, 5), device=torch_device)

        with torch.no_grad():
            # Generate a long sequence to check for exploding states
            outputs = model.generate(
                input_ids, 
                max_new_tokens=512, 
                use_cache=True,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True # Needed to access logits of all steps
            )

        # Check the logits at the final step for any non-finite values
        final_logits = outputs.scores[-1]
        self.assertTrue(
            torch.all(torch.isfinite(final_logits)),
            "Found NaN or Inf in logits after long generation, indicating numerical instability."
        )

    @require_torch
    def test_custom_position_ids(self):
        """
        Ensures the model respects manually provided `position_ids` instead of
        always relying on its internal counter.
        """
        config = self._get_config()
        model = BDHForCausalLM(config).to(torch_device)
        model.eval()

        input_ids = torch.randint(2, config.vocab_size, (1, 10), device=torch_device)

        # 1. Forward pass with default, sequential position_ids
        with torch.no_grad():
            outputs_default = model(input_ids)

        # 2. Forward pass with custom, non-sequential position_ids
        custom_position_ids = torch.tensor([[0, 1, 2, 3, 10, 11, 12, 30, 31, 32]], device=torch_device)
        with torch.no_grad():
            outputs_custom = model(input_ids, position_ids=custom_position_ids)

        # 3. The logits should be different, as RoPE is sensitive to position
        self.assertFalse(
            torch.allclose(outputs_default.logits, outputs_custom.logits),
            "Model did not produce different logits for custom position_ids."
        )