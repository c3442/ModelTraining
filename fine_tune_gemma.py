#!/usr/bin/env python
"""
Train Gemma 3 Vision on Nutrition5k-style data using QLoRA + FlashAttention.

Data format (JSONL):

Each line is a JSON array of messages, e.g.:

[
  {
    "role": "user",
    "content": [
      {"type": "image", "image": "dish_1561662216/rgb.png"},
      {"type": "text",  "text": "Extract foods from the image and estimate their portion size in grams."}
    ]
  },
  {
    "role": "assistant",
    "content": [
      {"type": "text", "text": "{\"foods\": [...]}"}  # JSON string with foods + grams
    ]
  }
]

Images live in S3 at:
  s3://8up-model-training/images_nutrition5k/<relative_path_from_json>
"""

import argparse
import json
from typing import Dict, List, Sequence, Union

import fsspec
import torch
from PIL import Image
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Defaults – tweak if needed
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "google/gemma-3n-E4B-it"     # vision-capable base
DEFAULT_PROCESSOR_ID = "google/gemma-3n-E4B-it"

# DEFAULT_TRAIN_JSONL = ["s3://8up-model-training/training_nutrition5k/test.jsonl"]
DEFAULT_TRAIN_JSONL = [
    "s3://8up-model-training/training_nutrition5k/batch_0.jsonl",
    "s3://8up-model-training/training_nutrition5k/batch_1.jsonl",
    "s3://8up-model-training/training_nutrition5k/batch_2.jsonl",
]
DEFAULT_IMAGE_BASE = "s3://8up-model-training/images_nutrition5k"
DEFAULT_OUTPUT_DIR = "gemma3-nutrition5k-vision-qlora"


# ---------------------------------------------------------------------------
# Data loading from JSONL + S3 paths
# ---------------------------------------------------------------------------

def load_nutrition5k_jsonl(jsonl_paths: Union[str, Sequence[str]]) -> Dataset:
    """
    Read one or more JSONL files where each line is a JSON *array* of messages
    and wrap them into a Dataset with a single column: "messages".
    """
    if isinstance(jsonl_paths, str):
        paths: List[str] = [jsonl_paths]
    else:
        paths = list(jsonl_paths)

    records: List[Dict] = []

    for jsonl_path in paths:
        fs, _, _ = fsspec.get_fs_token_paths(jsonl_path)
        with fs.open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                messages = json.loads(line)
                if not isinstance(messages, list):
                    raise ValueError(f"Expected a list of messages per line, got: {type(messages)}")
                records.append({"messages": messages})

    if not records:
        raise RuntimeError(f"No records found in {paths}")

    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Vision helper: load PIL images from S3 or local base dir
# ---------------------------------------------------------------------------

def process_vision_info(messages: List[Dict], image_base_dir: str) -> List[Image.Image]:
    """
    Extract a list of PIL.Image.Image objects from message content.

    - Expects elements with: {"type": "image", "image": "<relative_path>.png"}
    - Loads images from: image_base_dir / <relative_path>
      where image_base_dir can be an s3:// or local path.
    """
    image_inputs: List[Image.Image] = []

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if not isinstance(element, dict):
                continue

            if element.get("type") != "image":
                continue

            image_field = element.get("image")
            if image_field is None:
                continue

            # Build full path (works for s3:// and local)
            if isinstance(image_field, str):
                if image_field.startswith("s3://"):
                    full_path = image_field
                else:
                    full_path = image_base_dir.rstrip("/") + "/" + image_field.lstrip("/")
            else:
                raise TypeError(
                    f"Expected image path string in 'image' field, got {type(image_field)}"
                )

            # Open via fsspec (supports S3, local, etc.)
            fs, _, _ = fsspec.get_fs_token_paths(full_path)
            try:
                with fs.open(full_path, "rb") as f:
                    pil = Image.open(f).convert("RGB")
                image_inputs.append(pil)
            except Exception as e:
                raise RuntimeError(f"Failed to load image from {full_path}: {e}") from e

    if len(image_inputs) == 0:
        raise RuntimeError("process_vision_info: No valid image found after filtering")

    return image_inputs


# ---------------------------------------------------------------------------
# Collate fn – uses processor.apply_chat_template + vision loader
# ---------------------------------------------------------------------------

def make_collate_fn(processor, image_base_dir: str):
    """
    Returns a collate_fn compatible with SFTTrainer.

    - Uses processor.apply_chat_template to build text from "messages"
    - Uses process_vision_info to load images from S3/local
    - Masks padding + image tokens in labels
    """

    def collate_fn(examples):
        texts: List[str] = []
        images: List[List[Image.Image]] = []

        for example in examples:
            msgs = example["messages"]

            # Load all image(s) for this example
            image_inputs = process_vision_info(msgs, image_base_dir)

            # Build chat text (no generation prompt during training)
            text = processor.apply_chat_template(
                msgs,
                add_generation_prompt=False,
                tokenize=False,
            )
            texts.append(text.strip())
            images.append(image_inputs)

        # Tokenize text and process images into tensors
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Labels = input_ids, but with padding + image tokens masked to -100
        labels = batch["input_ids"].clone()

        # Image token id(s)
        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
        ]

        # Mask padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # Mask image tokens
        labels[torch.isin(labels, torch.tensor(image_token_id))] = -100
        # Extra special token used by the Gemma guide
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Base Gemma model id (vision-capable).",
    )
    parser.add_argument(
        "--processor_id",
        type=str,
        default=DEFAULT_PROCESSOR_ID,
        help="Processor / tokenizer id.",
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        nargs="+",
        default=DEFAULT_TRAIN_JSONL,
        help="Path to JSONL training data (can be s3://...).",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default=DEFAULT_IMAGE_BASE,
        help="Base directory for images (e.g., s3://8up-model-training/images_nutrition5k).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for LoRA adapter + processor.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Steps to accumulate gradients before optimizer step.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (QLoRA default from the paper).",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set, will try to push to Hugging Face Hub.",
    )

    args_cli = parser.parse_args()

    # Ensure CUDA + bf16-friendly GPU (e.g., A100)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this script.")

    major_cc = torch.cuda.get_device_capability()[0]
    if major_cc < 8:
        raise ValueError(
            f"GPU compute capability {major_cc} may not support bfloat16/flash-attn well. "
            "Use a GPU like A100, L4, or later."
        )

    # ----------------------------------------------------------------------
    # Load dataset from JSONL
    # ----------------------------------------------------------------------
    print(f"Loading dataset from {args_cli.train_jsonl} ...")
    dataset = load_nutrition5k_jsonl(args_cli.train_jsonl)
    print("Example sample[0].keys():", dataset[0].keys())
    print("First example message roles:", [m["role"] for m in dataset[0]["messages"]])

    # ----------------------------------------------------------------------
    # Load Gemma model + processor with QLoRA + FlashAttention2
    # ----------------------------------------------------------------------
    print("Loading model + processor with QLoRA + FlashAttention2...")

    model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )
    model_kwargs["quantization_config"] = quant_config

    model = AutoModelForImageTextToText.from_pretrained(
        args_cli.model_id,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(args_cli.processor_id)

    # ----------------------------------------------------------------------
    # LoRA / QLoRA configuration – from official vision QLoRA guide
    # ----------------------------------------------------------------------
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )

    # ----------------------------------------------------------------------
    # Training hyperparameters (SFTConfig)
    # ----------------------------------------------------------------------
    sft_args = SFTConfig(
        output_dir=args_cli.output_dir,
        num_train_epochs=args_cli.num_train_epochs,
        per_device_train_batch_size=args_cli.per_device_train_batch_size,
        gradient_accumulation_steps=args_cli.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=args_cli.learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        push_to_hub=args_cli.push_to_hub,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",  # we use a custom collator; no single text field
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    sft_args.remove_unused_columns = False

    collate_fn = make_collate_fn(processor, args_cli.image_base_dir)

    # ----------------------------------------------------------------------
    # Build SFTTrainer
    # ----------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )

    # ----------------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------------
    print("Starting training...")
    trainer.train()

    # Save final adapter + processor
    print(f"Saving final model + processor to {args_cli.output_dir} ...")
    trainer.save_model()
    processor.save_pretrained(args_cli.output_dir)

    print("Training complete.")


if __name__ == "__main__":
    main()
