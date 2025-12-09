#!/usr/bin/env python
"""
Evaluate Gemma 3 Vision on Nutrition5k data using a FIXED Text Prompt.

Generates predictions by overriding the dataset's text prompt with a custom
instruction set, then calculates accuracy metrics against the ground truth.
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
from typing import List, Dict, Any
from huggingface_hub import login
from peft import PeftModel
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

TEXT_PROMPT = """
Extract foods from the image and apply the following rules strictly.
**Naming**
- Use succinct, standardized, and consistent names.
**Duplicates**
- Return exactly one canonical entry per unique food or ingredient.
- Combine duplicates into a single item and indicate quantity or repetition in the portion field.
**Portion Size**
- Estimate portion sizes as accurately as possible from the image, using common household or metric units (e.g., “1 cup,” “200 g”).
- For real foods shown multiple times, aggregate them into the total portion.
**Output Format**
- Return the final results strictly as a valid **JSON array** of objects, one object per identified food item.
- Each object must include:
{
  "name": str type, food name,
  "grams": float type, estimated portion in gram,
}
"""

# DEFAULT_EVAL_JSONL = ["s3://8up-model-training/training_nutrition5k/test.jsonl"]
# DEFAULT_IMAGE_BASE = "s3://8up-model-training/images_nutrition5k"
DEFAULT_EVAL_JSONL = ["/Users/linliao/Dev/8up/data/training_nutrition5k/test.jsonl"]
DEFAULT_IMAGE_BASE = "/Users/linliao/Dev/8up/data/nutrition5k/images/realsense_overhead"
DEFAULT_MODEL_DIR = "gemma3-nutrition5k-vision-qlora"
DEFAULT_BASE_MODEL_ID = "google/gemma-3n-E4B-it"
DEFAULT_PROCESSOR_ID = "google/gemma-3n-E4B-it"
DEFAULT_OUTPUT_DIR = "eval-results"


# ---------------------------------------------------------------------------
# Metrics Helper Class
# ---------------------------------------------------------------------------

class NutritionMetrics:
    def __init__(self):
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.mass_errors = []  # Error for correctly identified items
        self.total_mass_errors = []  # Error for total dish weight

    def parse_output(self, text: str) -> Dict[str, Any]:
        """
        Parses model output to find valid JSON.
        """
        # Remove markdown fencing
        clean_text = re.sub(r'```json\s*|\s*```', '', text).strip()

        # extract matching brackets
        start = clean_text.find('[')
        end = clean_text.rfind(']')

        # If the model returned a bare list [...] instead of {"foods": [...]}, handle it
        if start != -1 and end != -1:
            json_str = clean_text[start: end + 1]
            try:
                data = json.loads(json_str)
                # Standardize to dict format for easier comparison
                if isinstance(data, list):
                    return {"foods": data}
                return data
            except json.JSONDecodeError:
                pass

        # Fallback: try finding curly braces if the list parse failed
        start = clean_text.find('{')
        end = clean_text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(clean_text[start: end + 1])
            except:
                pass

        return {"foods": []}

    def update(self, pred_data: Dict, ref_data: Dict):
        # Normalize names
        pred_foods = {f.get("name", "unknown").lower().strip(): f.get("grams", 0.0) for f in pred_data.get("foods", [])}
        ref_foods = {f.get("name", "unknown").lower().strip(): f.get("grams", 0.0) for f in ref_data.get("foods", [])}

        pred_names = set(pred_foods.keys())
        ref_names = set(ref_foods.keys())

        # Detection stats
        common = pred_names.intersection(ref_names)
        self.tp += len(common)
        self.fp += len(pred_names - ref_names)
        self.fn += len(ref_names - pred_names)

        # Mass Error (for matched items only)
        for name in common:
            try:
                p = float(pred_foods[name])
                r = float(ref_foods[name])
                self.mass_errors.append(abs(p - r))
            except:
                pass

        # Total Dish Mass Error
        try:
            total_pred = sum(float(x) for x in pred_foods.values())
            total_ref = sum(float(x) for x in ref_foods.values())
            self.total_mass_errors.append(abs(total_pred - total_ref))
        except:
            pass

    def compute(self) -> Dict[str, float]:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "mae_ingredients_grams": round(np.mean(self.mass_errors) if self.mass_errors else 0.0, 2),
            "mae_total_dish_grams": round(np.mean(self.total_mass_errors) if self.total_mass_errors else 0.0, 2)
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
    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        # Log in using the retrieved token
        # Note: If you ran 'huggingface-cli login' previously, this line is optional.
        login(token=hf_token)
    else:
        # Provide a warning if the token is not set
        print("Warning: HF_TOKEN environment variable not found. ")
        print("Please set it or use 'huggingface-cli login' for authentication.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base_model_id", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--eval_jsonl", type=str, nargs="+", default=DEFAULT_EVAL_JSONL)
    parser.add_argument("--image_base", type=str, default=DEFAULT_IMAGE_BASE)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_calc = NutritionMetrics()

    # 1. Load Model (Base + Adapter)
    print(f"Loading base model: {args.base_model_id}...")
    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    # model = AutoModelForImageTextToText.from_pretrained(
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        # quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # torch.float16,
        low_cpu_mem_usage=True,
    )

    if os.path.exists(args.adapter_path):
        print(f"Loading LoRA adapter: {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        print("No adapter found. Using base model.")

    model.eval()

    proc_path = args.adapter_path if os.path.exists(args.adapter_path) else DEFAULT_PROCESSOR_ID
    processor = AutoProcessor.from_pretrained(proc_path)

    # 2. Load Data
    print("Loading test data...")
    dataset = load_eval_data(args.eval_jsonl)
    print(f"Found {len(dataset)} examples. Starting evaluation...")

    results = []

    for idx, messages in enumerate(tqdm(dataset)):
        # Extract Original Messages
        original_user_msgs = [m for m in messages if m["role"] == "user"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]

        if not assistant_msgs or not original_user_msgs:
            continue

        # 1. Prepare Ground Truth
        # Reference is usually inside a "text" field containing JSON string
        ref_text_raw = assistant_msgs[0]["content"][0]["text"]
        ref_json = metrics_calc.parse_output(ref_text_raw)

        # 2. Prepare Images
        # We still need to load images based on the paths in the *original* message
        images = process_vision_info(original_user_msgs, args.image_base)
        if not images:
            continue

        # 3. Construct NEW Prompt (Overriding the file's text prompt)
        # We keep the "image" content block so the processor knows where to put image tokens
        # but we replace the "text" content block with our fixed TEXT_PROMPT.

        # Find the content item that holds the image path/placeholder
        original_content = original_user_msgs[0]["content"]
        if not isinstance(original_content, list): original_content = [original_content]

        image_content_items = [x for x in original_content if x.get("type") == "image"]

        # Create the new content list: [Image_Item, New_Text_Prompt]
        new_content = image_content_items + [{"type": "text", "text": TEXT_PROMPT}]

        override_user_msgs = [{"role": "user", "content": new_content}]

        # 4. Generate Prompt
        prompt = processor.apply_chat_template(
            override_user_msgs,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = processor(text=prompt, images=[images], return_tensors="pt").to(model.device)

        # 5. Run Inference
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                # use_cache=True
            )

        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True)

        # 6. Parse and Calculate
        pred_json = metrics_calc.parse_output(output_text)
        metrics_calc.update(pred_json, ref_json)

        results.append({
            "id": idx,
            "prompt_text_used": TEXT_PROMPT,
            "ground_truth": ref_json,
            "prediction": pred_json,
            "raw_output": output_text
        })

    # Save outputs
    metrics = metrics_calc.compute()

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"F1 Score (Food Detection):  {metrics['f1_score']}")
    print(f"MAE (Ingredients Mass):     {metrics['mae_ingredients_grams']} g")
    print(f"MAE (Total Dish Mass):      {metrics['mae_total_dish_grams']} g")
    print(f"Detailed results saved to:  {args.output_dir}")
    print("=" * 40)


if __name__ == "__main__":
    main()