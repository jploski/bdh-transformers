import argparse
import logging
import math
import os

from datasets import DatasetDict
import requests
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    ByT5Tokenizer,
)

from bdh_transformers.models.bdh.configuration_bdh import BDHConfig
from bdh_transformers.models.bdh.modeling_bdh import BDHForCausalLM

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")


# Fetch the tiny Shakespeare dataset
def fetch_data():
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


def main(args):
    """
    Main training and evaluation script for the BDH-GPU model.
    """
    fetch_data()

    # --- 1. Load Local Dataset and Manually Split the Text ---
    logger.info(f"Loading local dataset from: {args.dataset_path}")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {args.dataset_path}")

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Manually split the raw text string before creating datasets.
    split_percentage = args.validation_split_percentage / 100.0
    split_index = int(len(full_text) * (1.0 - split_percentage))

    train_text = full_text[:split_index]
    val_text = full_text[split_index:]

    from datasets import Dataset

    # Create two separate Dataset objects.
    train_dataset = Dataset.from_dict({"text": [train_text]})
    val_dataset = Dataset.from_dict({"text": [val_text]})

    # Combine them into the DatasetDict the Trainer expects.
    raw_datasets = DatasetDict({"train": train_dataset, "validation": val_dataset})
    logger.info(
        f"Data successfully split into {len(train_text):,} train characters and {len(val_text):,} validation characters."
    )

    # --- 2. Initialize Byte-Level Tokenizer ---
    # (This section is correct and remains the same)
    logger.info("Initializing byte-level tokenizer")
    tokenizer = ByT5Tokenizer()

    # --- 3. Preprocess Data ---
    # (This section is also correct and remains the same)
    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    logger.info("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    block_size = args.block_size

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info(f"Grouping texts into chunks of {block_size} tokens...")
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    # --- 4. Configure and Initialize Model ---
    logger.info("Initializing model from scratch with a new config")
    config = BDHConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        dropout=0.1,
        mlp_internal_dim_multiplier=128,  # A common ratio is 4x hidden_size
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_position_embeddings=block_size,
        attn_implementation=args.attn_implementation,
    )

    model = BDHForCausalLM(config=config)
    logger.info(f"Model created with {model.num_parameters():,} parameters.")

    # --- 5. Set Up Trainer ---
    # Data collator for causal language modeling
    # It will automatically create `labels` from `input_ids` by shifting
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.model_name),
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        load_best_model_at_end=True,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",  # Disable wandb/tensorboard integration for simplicity
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # --- 6. Train and Evaluate ---
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Training complete. Evaluating on the validation set...")
    eval_results = trainer.evaluate()

    # --- 7. Calculate and Report Perplexity ---
    try:
        eval_loss = eval_results["eval_loss"]
        perplexity = math.exp(eval_loss)
        logger.info(f"Validation Loss: {eval_loss:.4f}")
        logger.info(f"Validation Perplexity: {perplexity:.4f}")
    except KeyError:
        logger.error(
            "Evaluation did not produce 'eval_loss'. Cannot calculate perplexity."
        )

    # --- 8. Generate a Sample (Cherry on Top) ---
    logger.info("Generating a sample text from the trained model...")

    # The trainer loads the best model at the end, so we can use it directly.
    model = trainer.model
    model.eval()
    device = trainer.args.device

    prompt = "ROMEO:\n"
    logger.info(f"Using prompt: '{prompt}'")

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_k=4,
            use_cache=True,
        )

    # Decode and print the result
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Use standard print for a clean, highlighted output block
    print("-" * 80)
    print(">>>>> Generated Sample <<<<<")
    print(generated_text)
    print("-" * 80)

    # # # --- 9. Save Final Model ---
    final_model_path = os.path.join(args.output_dir, f"{args.model_name}-final")
    trainer.save_model(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the BDH-GPU model."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="bdh-shakespeare-byte-level",
        help="Name of the model run for output directories.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./input.txt",
        help="Path to the local dataset to use.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=int,
        default=10,
        help="Percentage of data to use for validation.",
    )
    parser.add_argument(
        "--block_size", type=int, default=512, help="Context size for training."
    )
    parser.add_argument(
        "--max_steps", type=int, default=800, help="Number of training steps."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="Training and evaluation batch size per device.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./training_output",
        help="Directory to save checkpoints and final model.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="bdh_parallel",
        choices=["bdh_parallel", "bdh_recurrent"],
        help="Attention implementation to use. 'bdh_parallel' is recommended for training.",
    )

    args = parser.parse_args()
    main(args)
