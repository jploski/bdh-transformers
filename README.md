# bdh-transformers

A [Hugging Face Transformers](https://github.com/huggingface/transformers) compatible implementation of the [Baby Dragon Hatchling (BDH-GPU)](https://github.com/pathwaycom/bdh) language model.

This library provides the `BDHForCausalLM` architecture wrapped in the standard `transformers` API, allowing you to use it with `AutoModel`, `AutoConfig`, and the generation pipeline.

Note that this experimental model is *not* based on the standard Transformer architecture. It uses a recurrent structure for each layer, making it more similar to models like the [Universal Transformer](https://arxiv.org/abs/1807.03819). Currently the intended audience for this model are ML researchers / developers. Model weights are not included (train your own bebe dragon).

![Baby Dragon Hatchling](images/bdh.png)

## Architectural Highlights

According to the [research paper](https://arxiv.org/abs/2509.26507) the BDH model was derived by mimicking low-level biological brain structures through a [Hebbian learning](https://en.wikipedia.org/wiki/Hebbian_theory) framework, which was subsequently mapped to a novel language model architecture to support efficient execution on current hardware (BDH-GPU).
The implementation found provided by this repository, apart from integrating with the HF transformers ecosystem, focuses on long-context sequence modeling. Its key features are:

*   **Recurrent State Cache:** Instead of a traditional KV cache that grows with the sequence length, BDH-GPU (similar to Mamba) can utilize a fixed-size incrementally updated state. This allows it to process sequences of "infinite" length with constant memory usage during generation (subject to numerical precision and information-theoretic limitations).
*   **Linear Attention:** Attention is computed in linear time `O(T)` relative to sequence length, compared to the quadratic complexity `O(T^2)` of standard attention, making it more suitable for long documents or continuous streams of data.
*   **Dual Attention Implementations:** The model provides two attention mechanisms that are mathematically equivalent but optimized for different use cases:
    *   `attn_implementation="bdh_recurrent"`: The default, stateful implementation ideal for efficient, token-by-token generation.
    *   `attn_implementation="bdh_parallel"`: A parallel implementation optimized for fast prefilling or training on long sequences.

## Installation

### Prerequisites
*   Python 3.10+
*   PyTorch
*   Hugging Face Transformers

### From Source (Recommended for Development)
If you have cloned the repository locally:

```bash
git clone https://github.com/jploski/bdh
cd bdh
pip install -e .
```

### From PyPI

An installable package is not currently provided.

---

## Usage

Once installed, you can import the package and use standard Hugging Face classes. The package automatically registers the `bdh` model type upon import.

### Training

See the included example script `train.py`.

### Text Generation
```python
import torch
import bdh_transformers  # Importing this registers 'bdh' with AutoClasses
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer from the Hugging Face Hub or a local path
model_id = "./bdh"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=False, # change to True if you pull the model from HF Hub
    attn_implementation="bdh_recurrent", # or "bdh_parallel" for KV cache
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

prompt = "The small smooth-skinned dragon"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20, use_cache=True)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

---

## Features & Compatibility

This implementation is designed to fit seamlessly into the Hugging Face ecosystem:

*   **Auto-Registration:** Automatically registers `BDHConfig` and `BDHForCausalLM` with the `transformers` library when imported.
    **Recurrent State Caching:** Full support for `use_cache=True` using a constant-memory recurrent state, enabling "infinite" context generation.
*   **Chunked Linear Attention:** Efficiently processes long prompts (prefill) with linear time complexity and constant memory usage within the attention block.
*   **RoPE:** Implements Rotary Positional Embeddings for positional encoding.
*   **Generation:** Fully compatible with `.generate()` methods, including greedy search, sampling, and beam search.
*   **Trainable:** Supports standard training and fine-tuning workflows within the Hugging Face ecosystem.

---

## Development

### Running Tests
This repository includes a comprehensive unit test suite to verify model consistency and integration.

1.  Install test dependencies:
    ```bash
    pip install pytest
    ```

2.  Run the test suite:
    ```bash
    pytest tests/models/test_modeling_bdh.py
    ```

---

## Credits & Acknowledgements

*   **Original Code & Architecture:** The core BDH-GPU architecture was [researched](https://arxiv.org/abs/2509.26507) and [developed](https://github.com/pathwaycom/bdh) by **Pathway Technology, Inc.**
*   **Hugging Face Implementation:** Packaged and adapted for the `transformers` library by **Jan Ploski**. The author of this HF integration is not affiliated with the original creators of the model. 

---

## License

This project is licensed under the **Apache License 2.0**.

See the [LICENSE](LICENSE) file for details.

*   Copyright (c) 2025 Jan Ploski
*   Copyright (c) 2025 Pathway Technology, Inc.
