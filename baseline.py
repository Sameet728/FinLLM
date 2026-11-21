"""
FinLLM Baseline Evaluation
============================
Evaluates the pretrained Qwen2.5-0.5B-Instruct model BEFORE fine-tuning.
This establishes the baseline performance for comparison.

Usage:
    python baseline.py

Student: Sameet Pisal | PRN: 202401120018 | Roll: 41
"""

import json
import time
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

from config import (
    TEST_CSV, BASELINE_METRICS_JSON, RESULTS_DIR, LABEL_NAMES, ensure_dirs
)
from inference import load_base_model, predict


def evaluate_baseline():
    ensure_dirs()
    print("=" * 60)
    print("FinLLM — Baseline Evaluation (Pretrained Model)")
    print("=" * 60)
    
    # Load test set
    if not TEST_CSV.exists():
        print(f"ERROR: Test CSV not found: {TEST_CSV}")
        print("Run prepare_dataset.py first.")
        sys.exit(1)
    
    test_df = pd.read_csv(TEST_CSV)
    print(f"\nTest set: {len(test_df)} examples")
    
    # Load pretrained model
    print("\nLoading pretrained model (no adapter)...")
    model, tokenizer = load_base_model()
    print("Model loaded successfully.")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Evaluate on test set
    print(f"\nRunning baseline inference on {len(test_df)} test examples...")
    print("-" * 60)
    
    predictions = []
    true_labels = []
    raw_outputs = []
    inference_times = []
    invalid_count = 0
    
    for i, row in test_df.iterrows():
        result = predict(
            sentence=row["sentence"],
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            return_confidence=False  # skip confidence for speed
        )
        
        pred_label = result["label"]
        true_label = row["label"].lower() if isinstance(row["label"], str) else str(row["label"]).lower()
        
        predictions.append(pred_label if pred_label else "invalid")
        true_labels.append(true_label)
        raw_outputs.append(result["raw_output"])
        inference_times.append(result["inference_time_s"])
        
        if not result["is_valid"]:
            invalid_count += 1
        
        # Progress every 10 examples
        if (i + 1) % 10 == 0 or i == 0:
            valid_preds = [p for p in predictions if p != "invalid"]
            current_acc = sum(1 for p, t in zip(valid_preds, true_labels[:len(valid_preds)]) if p == t) / max(len(valid_preds), 1)
            print(f"  [{i+1:3d}/{len(test_df)}] {row['sentence'][:50]}...")
            print(f"           True: {true_label:10s} | Pred: {pred_label or 'INVALID':10s} | {result['inference_time_s']:.2f}s")
    
    print("-" * 60)
    
    # Filter out invalid predictions for metric calculation
    valid_mask = [p != "invalid" for p in predictions]
    valid_predictions = [predictions[i] for i in range(len(predictions)) if valid_mask[i]]
    valid_true = [true_labels[i] for i in range(len(true_labels)) if valid_mask[i]]
    
    # Compute metrics
    if valid_predictions:
        accuracy = accuracy_score(valid_true, valid_predictions)
        macro_precision = precision_score(valid_true, valid_predictions, average="macro", labels=LABEL_NAMES, zero_division=0)
        macro_recall = recall_score(valid_true, valid_predictions, average="macro", labels=LABEL_NAMES, zero_division=0)
        macro_f1 = f1_score(valid_true, valid_predictions, average="macro", labels=LABEL_NAMES, zero_division=0)
        weighted_f1 = f1_score(valid_true, valid_predictions, average="weighted", labels=LABEL_NAMES, zero_division=0)
    else:
        accuracy = macro_precision = macro_recall = macro_f1 = weighted_f1 = 0.0
    
    avg_inference_time = float(np.mean(inference_times))
    
    print(f"\n{'='*60}")
    print("BASELINE EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Total test examples : {len(test_df)}")
    print(f"  Valid predictions   : {len(valid_predictions)}")
    print(f"  Invalid predictions : {invalid_count}")
    print(f"  Accuracy            : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro Precision     : {macro_precision:.4f}")
    print(f"  Macro Recall        : {macro_recall:.4f}")
    print(f"  Macro F1-score      : {macro_f1:.4f}")
    print(f"  Weighted F1-score   : {weighted_f1:.4f}")
    print(f"  Avg inference time  : {avg_inference_time:.3f}s")
    print(f"{'='*60}")
    
    # Classification report
    if valid_predictions:
        report = classification_report(valid_true, valid_predictions, labels=LABEL_NAMES, zero_division=0)
        print(f"\nClassification Report:\n{report}")
    
    # Save results
    metrics = {
        "model": "Qwen2.5-0.5B-Instruct (pretrained, no adapter)",
        "total_test_examples": len(test_df),
        "valid_predictions": len(valid_predictions),
        "invalid_predictions": int(invalid_count),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "avg_inference_time_s": float(avg_inference_time),
        "total_parameters": int(total_params),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(BASELINE_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nBaseline metrics saved: {BASELINE_METRICS_JSON}")
    
    # Save individual predictions
    pred_df = pd.DataFrame({
        "sentence": test_df["sentence"].values,
        "true_label": true_labels,
        "baseline_prediction": predictions,
        "baseline_raw_output": raw_outputs,
        "baseline_inference_time_s": inference_times,
        "baseline_is_valid": valid_mask,
    })
    baseline_preds_path = RESULTS_DIR / "baseline_predictions.csv"
    pred_df.to_csv(baseline_preds_path, index=False)
    print(f"Individual predictions saved: {baseline_preds_path}")
    
    return metrics, pred_df


if __name__ == "__main__":
    metrics, pred_df = evaluate_baseline()
    print("\nBaseline evaluation complete.")
