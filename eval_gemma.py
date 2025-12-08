#!/usr/bin/env python
"""
Evaluate Gemma 3 Vision on Nutrition5k data.

Generates predictions and calculates:
1. Food detection accuracy (Precision, Recall, F1)
2. Portion estimation error (MAE in grams)
"""

import argparse
import json
import os
import re
import torch
import fsspec
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List, Dict, Any, Tuple
from peft import PeftModel
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

# Defaults
DEFAULT_EVAL_JSONL = ["s3://8up-model-training/training_nutrition5k/test.jsonl"]
DEFAULT_IMAGE_BASE = "s3://8up-model-training/images_nutrition5k"
DEFAULT_MODEL_DIR = "gemma3-nutrition5k-vision-qlora"
DEFAULT_BASE_MODEL_ID = "google/gemma-3-4b-it"
DEFAULT_PROCESSOR_ID = "google/gemma-3-4b-it"
DEFAULT_OUTPUT_DIR = "eval-results"


# ---------------------------------------------------------------------------
# Metrics Helper Class
# ---------------------------------------------------------------------------

class NutritionMetrics:
    def __init__(self):
        self.tp = 0  # True Positives (Correct food identified)
        self.fp = 0  # False Positives (Hallucinated food)
        self.fn = 0  # False Negatives (Missed food)
        self.mass_errors = []  # Absolute error in grams for correctly matched items
        self.total_mass_errors = []  # Absolute error of total dish weight

    def parse_output(self, text: str) -> Dict[str, Any]:
        """
        Robustly parses the model output to find valid JSON.
        Expected format: {"foods": [{"name": "x", "grams": 10.0}, ...]}
        """
        # Strip markdown code blocks if present
        clean_text = re.sub(r'```json\s*|\s*```', '', text).strip()

        # Try to find the first '{' and last '}'
        start = clean_text.find('{')
        end = clean_text.rfind('}')

        if start != -1 and end != -1:
            json_str = clean_text[start: end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Return empty structure if parsing fails
        return {"foods": []}

    def update(self, pred_data: Dict, ref_data: Dict):
        """
        Compare prediction vs reference for a single example.
        """
        # Normalize names for matching (lowercase, strip)
        pred_foods = {f["name"].lower().strip(): f.get("grams", 0.0) for f in pred_data.get("foods", [])}
        ref_foods = {f["name"].lower().strip(): f.get("grams", 0.0) for f in ref_data.get("foods", [])}

        pred_names = set(pred_foods.keys())
        ref_names = set(ref_foods.keys())

        # 1. Detection Metrics
        common = pred_names.intersection(ref_names)
        self.tp += len(common)
        self.fp += len(pred_names - ref_names)
        self.fn += len(ref_names - pred_names)

        # 2. Mass Error (only for matched items)
        for name in common:
            p_mass = float(pred_foods[name])
            r_mass = float(ref_foods[name])
            self.mass_errors.append(abs(p_mass - r_mass))

        # 3. Total Dish Mass Error
        total_pred = sum(float(x) for x in pred_foods.values())
        total_ref = sum(float(x) for x in ref_foods.values())
        self.total_mass_errors.append(abs(total_pred - total_ref))

    def compute(self) -> Dict[str, float]:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        mean_mass_mae = np.mean(self.mass_errors) if self.mass_errors else 0.0
        mean_total_mae = np.mean(self.total_mass_errors) if self.total_mass_errors else 0.0

        return {
            "food_detection_precision": round(precision, 4),
            "food_detection_recall": round(recall, 4),
            "food_detection_f1": round(f1, 4),
            "ingredient_mass_mae_grams": round(mean_mass_mae, 2),
            "total_dish_mass_mae_grams": round(mean_total_mae, 2)
        }


# ---------------------------------------------------------------------------
# Data Loading & Processing
# ---------------------------------------------------------------------------

def load_eval_data(jsonl_paths: List[str]) -> List[List[Dict]]:
    records = []
    for path in jsonl_paths:
        fs, _, _ = fsspec.get_fs_token_paths(path)
        with fs.open(path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def process_vision_info(messages: List[Dict], image_base_dir: str) -> List[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list): content = [content]
        for element in content:
            if isinstance(element, dict) and element.get("type") == "image":
                image_field = element.get("image")
                full_path = f"{image_base_dir.rstrip('/')}/{image_field.lstrip('/')}"
                fs, _, _ = fsspec.get_fs_token_paths(full_path)
                try:
                    with fs.open(full_path, "rb") as f:
                        pil = Image.open(f).convert("RGB")
                    image_inputs.append(pil)
                except Exception as e:
                    print(f"Error loading image {full_path}: {e}")
    return image_inputs


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base_model_id", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--eval_jsonl", type=str, nargs="+", default=DEFAULT_EVAL_JSONL)
    parser.add_argument("--image_base", type=str, default=DEFAULT_IMAGE_BASE)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_calc = NutritionMetrics()

    # 1. Load Model
    print(f"Loading base model: {args.base_model_id}...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_id,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if os.path.exists(args.adapter_path):
        print(f"Loading LoRA adapter: {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        print("No adapter found. Using base model.")

    model.eval()

    # Load processor from adapter dir if present, else base
    proc_path = args.adapter_path if os.path.exists(args.adapter_path) else DEFAULT_PROCESSOR_ID
    print(f"Loading processor from {proc_path}...")
    processor = AutoProcessor.from_pretrained(proc_path)

    # 2. Load Data
    print("Loading test data...")
    dataset = load_eval_data(args.eval_jsonl)
    print(f"Found {len(dataset)} examples.")

    # 3. Inference Loop
    results = []

    print("Starting evaluation...")
    for idx, messages in enumerate(tqdm(dataset)):
        # Extract inputs and ground truth
        user_msgs = [m for m in messages if m["role"] == "user"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]

        # Get Reference JSON
        if not assistant_msgs:
            continue
        # The reference content is a stringified JSON inside "text" field
        ref_text_raw = assistant_msgs[0]["content"][0]["text"]
        ref_json = metrics_calc.parse_output(ref_text_raw)

        # Prepare Inputs
        images = process_vision_info(user_msgs, args.image_base)
        if not images:
            print(f"Skipping index {idx}: No images found.")
            continue

        prompt = processor.apply_chat_template(
            user_msgs,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = processor(text=prompt, images=[images], return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

        # Decode
        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True)

        # Parse Prediction
        pred_json = metrics_calc.parse_output(output_text)

        # Update Metrics
        metrics_calc.update(pred_json, ref_json)

        # Save record
        results.append({
            "id": idx,
            "ground_truth": ref_json,
            "prediction": pred_json,
            "raw_output": output_text
        })

    # 4. Save Final Results
    metrics = metrics_calc.compute()

    # Save Metrics Summary
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save Predictions
    preds_path = os.path.join(args.output_dir, "predictions.jsonl")
    with open(preds_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    print("-" * 40)
    print("Evaluation Complete.")
    print(f"F1 Score (Food Detection): {metrics['food_detection_f1']}")
    print(f"MAE (Mass Estimation):     {metrics['ingredient_mass_mae_grams']} g")
    print(f"Results saved to:          {args.output_dir}")
    print("-" * 40)


if __name__ == "__main__":
    main()