#!/usr/bin/env python
"""
Evaluate Gemma 3 Vision on Nutrition5k-style chat data.

Supports:
  - Baseline model (Gemma 3 vision-capable base)
  - Fine-tuned LoRA adapter produced by `fine_tune_gemma.py`

Writes per-example generations to JSONL and a small metrics JSON.
"""

import argparse
import datetime as _dt
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import difflib
import fsspec
import torch
from PIL import Image
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EVAL_JSONL = [
    "s3://8up-model-training/training_nutrition5k/test.jsonl",
]
DEFAULT_IMAGE_BASE = "s3://8up-model-training/images_nutrition5k"
DEFAULT_MODEL_DIR = "gemma3-nutrition5k-vision-qlora"  # fine-tuned adapter dir
DEFAULT_OUTPUT_DIR = "eval-results"
DEFAULT_BASE_MODEL_ID = "google/gemma-3-4b-pt"
DEFAULT_PROCESSOR_ID = "google/gemma-3-4b-it"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_nutrition5k_jsonl(jsonl_paths: Union[str, Sequence[str]]) -> Dataset:
    """
    Read one or more JSONL files where each line is a JSON *array* of messages.
    Returns a Dataset with a single column: "messages".
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
                    raise ValueError(f"Expected list of messages per line, got {type(messages)}")
                records.append({"messages": messages})

    if not records:
        raise RuntimeError(f"No records found in {paths}")

    return Dataset.from_list(records)


def load_images_from_messages(messages: List[Dict], image_base_dir: str) -> List[Image.Image]:
    """
    Extract all image payloads from a message list and return PIL images.

    Images are resolved relative to `image_base_dir` unless they already carry
    an s3:// or absolute path. Raises if no images are found.
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

            if isinstance(image_field, str):
                if image_field.startswith("s3://"):
                    full_path = image_field
                else:
                    full_path = image_base_dir.rstrip("/") + "/" + image_field.lstrip("/")
            else:
                raise TypeError(f"Expected image path string in 'image' field, got {type(image_field)}")

            fs, _, _ = fsspec.get_fs_token_paths(full_path)
            try:
                with fs.open(full_path, "rb") as f:
                    pil = Image.open(f).convert("RGB")
                image_inputs.append(pil)
            except Exception as exc:
                raise RuntimeError(f"Failed to load image from {full_path}: {exc}") from exc

    if not image_inputs:
        raise RuntimeError("No valid image found in messages")

    return image_inputs


def extract_reference_text(messages: List[Dict]) -> str:
    """
    Pull the assistant text (ground truth) from a conversation, if present.
    Concatenates multiple assistant text chunks.
    """
    parts: List[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and element.get("type") == "text":
                text = element.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    model: AutoModelForImageTextToText
    processor: AutoProcessor


def load_model_and_processor(
    adapter_dir: Union[str, None],
    base_model_id: str,
    processor_id: Union[str, None],
    attn_impl: str,
    load_in_4bit: bool,
) -> ModelBundle:
    """
    Load either the baseline model or the baseline + LoRA adapter.
    """
    processor_to_use = processor_id or adapter_dir or DEFAULT_PROCESSOR_ID

    model_kwargs = dict(
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
            bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
        )
        model_kwargs["quantization_config"] = quant_config

    # Base model first
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        **model_kwargs,
    )

    # Attach adapter if provided
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)

    processor = AutoProcessor.from_pretrained(processor_to_use)
    return ModelBundle(model=model, processor=processor)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def to_device(batch: Dict, device: torch.device) -> Dict:
    """Move tensor values to the target device."""
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def generate_one(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    messages: List[Dict],
    image_base_dir: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """
    Run a single forward generation for one conversation with images.
    """
    image_inputs = load_images_from_messages(messages, image_base_dir)
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    batch = processor(
        text=[prompt],
        images=[image_inputs],
        return_tensors="pt",
        padding=True,
    )
    batch = to_device(batch, model.device)

    do_sample = temperature > 0.0
    with torch.inference_mode():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
    return decoded[0].strip()


def try_json_parse(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def sequence_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    dataset = load_nutrition5k_jsonl(args.eval_jsonl)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    adapter_dir = args.adapter_dir if args.adapter_dir else None
    bundle = load_model_and_processor(
        adapter_dir=adapter_dir,
        base_model_id=args.base_model_id,
        processor_id=args.processor_id,
        attn_impl=args.attn_impl,
        load_in_4bit=args.load_in_4bit,
    )
    model, processor = bundle.model, bundle.processor

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    pred_path = out_dir / f"predictions_{timestamp}.jsonl"
    metrics_path = out_dir / f"metrics_{timestamp}.json"

    print(f"Evaluating {len(dataset)} samples...")
    print(f"Writing predictions to {pred_path}")

    total = len(dataset)
    exact = 0
    json_ok = 0
    sim_sum = 0.0

    with pred_path.open("w") as f_out:
        for idx, example in enumerate(dataset):
            messages = example["messages"]
            prediction = generate_one(
                model=model,
                processor=processor,
                messages=messages,
                image_base_dir=args.image_base_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            reference = extract_reference_text(messages)
            is_json = try_json_parse(prediction)
            similarity = sequence_similarity(reference, prediction) if reference else 0.0
            is_exact = bool(reference and prediction.strip() == reference.strip())

            total += 0  # no-op; keeps intent clear
            exact += int(is_exact)
            json_ok += int(is_json)
            sim_sum += similarity

            record = {
                "id": idx,
                "prediction": prediction,
                "reference": reference,
                "messages": messages,
                "prediction_is_json": is_json,
                "exact_match": is_exact,
                "similarity": similarity,
            }
            f_out.write(json.dumps(record) + "\n")

            if (idx + 1) % 10 == 0:
                print(f"  {idx + 1} / {len(dataset)} complete...")

    summary = {
        "timestamp": timestamp,
        "num_samples": len(dataset),
        "exact_match": exact,
        "exact_match_rate": exact / len(dataset) if dataset else 0.0,
        "valid_json": json_ok,
        "valid_json_rate": json_ok / len(dataset) if dataset else 0.0,
        "avg_similarity": sim_sum / len(dataset) if dataset else 0.0,
        "adapter_dir": adapter_dir,
        "base_model_id": args.base_model_id,
        "processor_id": args.processor_id or adapter_dir or DEFAULT_PROCESSOR_ID,
        "eval_jsonl": args.eval_jsonl,
        "image_base_dir": args.image_base_dir,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "load_in_4bit": args.load_in_4bit,
        "attn_impl": args.attn_impl,
    }

    with metrics_path.open("w") as f_metrics:
        json.dump(summary, f_metrics, indent=2)

    print("Evaluation complete.")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval_jsonl",
        type=str,
        nargs="+",
        default=DEFAULT_EVAL_JSONL,
        help="Path(s) to JSONL eval data.",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default=DEFAULT_IMAGE_BASE,
        help="Base directory for images (s3:// or local).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save predictions + metrics.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Path to fine-tuned adapter. Leave empty to use baseline only.",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default=DEFAULT_BASE_MODEL_ID,
        help="Base Gemma vision model id (used for both baseline + adapter).",
    )
    parser.add_argument(
        "--processor_id",
        type=str,
        default=None,
        help="Processor/tokenizer id. Defaults to adapter_dir if provided, else Gemma IT.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of eval samples (useful for quick sanity checks).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens during generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 => greedy).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit (saves memory; requires supported GPU).",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "eager"],
        help="Attention implementation passed to the model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
