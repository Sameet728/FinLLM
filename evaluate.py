"""
FinLLM Fine-Tuned Model Evaluation
=====================================
Evaluates the LoRA fine-tuned model on the same test set as baseline.
Generates comparison metrics and all required charts.

Usage:
    python evaluate.py

Student: Sameet Pisal | PRN: 202401120018 | Roll: 41
"""

import json
import time
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

from config import (
    TEST_CSV, RESULTS_DIR, CHARTS_DIR,
    BASELINE_METRICS_JSON, FINETUNED_METRICS_JSON,
    PREDICTIONS_CSV, CLASSIFICATION_REPORT_CSV,
    TRAINING_LOGS_CSV, TRAINING_CONFIG_JSON,
    CHART_CONFUSION_MATRIX, CHART_BASELINE_COMPARE,
    CHART_CLASSWISE, CHART_TRAINABLE_PARAMS,
    CHART_TRAINING_LOSS, CHART_VALIDATION_LOSS,
    LABEL_NAMES, ensure_dirs
)
from inference import load_finetuned_model, predict


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_finetuned():
    ensure_dirs()
    print("=" * 60)
    print("FinLLM — Fine-Tuned Model Evaluation")
    print("=" * 60)
    
    # Load test data
    test_df = pd.read_csv(TEST_CSV)
    print(f"\nTest set: {len(test_df)} examples")
    
    # Load fine-tuned model
    print("\nLoading fine-tuned model (base + LoRA adapter)...")
    model, tokenizer = load_finetuned_model()
    
    # Evaluate
    print(f"\nRunning inference on {len(test_df)} test examples...")
    print("-" * 60)
    
    predictions = []
    true_labels = []
    raw_outputs = []
    inference_times = []
    confidences = []
    invalid_count = 0
    
    for i, row in test_df.iterrows():
        result = predict(
            sentence=row["sentence"],
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            return_confidence=True,
        )
        
        pred_label = result["label"]
        true_label = row["label"].lower() if isinstance(row["label"], str) else str(row["label"]).lower()
        
        predictions.append(pred_label if pred_label else "invalid")
        true_labels.append(true_label)
        raw_outputs.append(result["raw_output"])
        inference_times.append(result["inference_time_s"])
        confidences.append(result.get("confidence"))
        
        if not result["is_valid"]:
            invalid_count += 1
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(test_df)}] True: {true_label:10s} | Pred: {pred_label or 'INVALID':10s} | {result['inference_time_s']:.2f}s")
    
    print("-" * 60)
    
    # Compute metrics
    valid_mask = [p != "invalid" for p in predictions]
    valid_predictions = [predictions[i] for i in range(len(predictions)) if valid_mask[i]]
    valid_true = [true_labels[i] for i in range(len(true_labels)) if valid_mask[i]]
    
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
    print("FINE-TUNED MODEL EVALUATION RESULTS")
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
    
    # Per-class metrics
    per_class = {}
    if valid_predictions:
        for label in LABEL_NAMES:
            p = precision_score(valid_true, valid_predictions, labels=[label], average="macro", zero_division=0)
            r = recall_score(valid_true, valid_predictions, labels=[label], average="macro", zero_division=0)
            f = f1_score(valid_true, valid_predictions, labels=[label], average="macro", zero_division=0)
            per_class[label] = {"precision": float(p), "recall": float(r), "f1": float(f)}
        
        report = classification_report(valid_true, valid_predictions, labels=LABEL_NAMES, zero_division=0)
        print(f"\nClassification Report:\n{report}")
    
    # Save finetuned metrics
    metrics = {
        "model": "Qwen2.5-0.5B-Instruct + LoRA (FinLLM)",
        "total_test_examples": len(test_df),
        "valid_predictions": len(valid_predictions),
        "invalid_predictions": int(invalid_count),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "avg_inference_time_s": float(avg_inference_time),
        "per_class_metrics": per_class,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(FINETUNED_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nFine-tuned metrics saved: {FINETUNED_METRICS_JSON}")
    
    # Save predictions
    pred_df = pd.DataFrame({
        "sentence": test_df["sentence"].values,
        "true_label": true_labels,
        "finetuned_prediction": predictions,
        "finetuned_raw_output": raw_outputs,
        "finetuned_inference_time_s": inference_times,
        "finetuned_is_valid": valid_mask,
    })
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    
    # Save classification report as CSV
    if valid_predictions:
        report_dict = classification_report(
            valid_true, valid_predictions, labels=LABEL_NAMES,
            output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(CLASSIFICATION_REPORT_CSV)
        print(f"Classification report saved: {CLASSIFICATION_REPORT_CSV}")
    
    return metrics, pred_df, valid_true, valid_predictions


# ─── Chart Generation ─────────────────────────────────────────────────────────

def generate_charts(finetuned_metrics, valid_true, valid_predictions):
    """Generate all required evaluation charts."""
    ensure_dirs()
    
    # Load baseline metrics for comparison
    baseline_metrics = {}
    if BASELINE_METRICS_JSON.exists():
        with open(BASELINE_METRICS_JSON) as f:
            baseline_metrics = json.load(f)
    
    # 1. Confusion Matrix
    print("\nGenerating confusion matrix...")
    if valid_predictions:
        cm = confusion_matrix(valid_true, valid_predictions, labels=LABEL_NAMES)
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#FDEBD0", "#E74C3C"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(LABEL_NAMES)))
        ax.set_yticks(range(len(LABEL_NAMES)))
        ax.set_xticklabels([l.capitalize() for l in LABEL_NAMES], fontsize=11)
        ax.set_yticklabels([l.capitalize() for l in LABEL_NAMES], fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_title("FinLLM Confusion Matrix (Fine-Tuned)", fontsize=13, fontweight="bold")
        
        for i in range(len(LABEL_NAMES)):
            for j in range(len(LABEL_NAMES)):
                text = ax.text(j, i, str(cm[i, j]),
                               ha="center", va="center",
                               fontsize=14, fontweight="bold",
                               color="white" if cm[i, j] > cm.max() * 0.5 else "black")
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(CHART_CONFUSION_MATRIX, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {CHART_CONFUSION_MATRIX}")
    
    # 2. Baseline vs Fine-Tuned Comparison
    print("Generating baseline comparison chart...")
    if baseline_metrics:
        metrics_to_plot = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
        metric_labels = ["Accuracy", "Macro\nPrecision", "Macro\nRecall", "Macro\nF1", "Weighted\nF1"]
        
        baseline_vals = [baseline_metrics.get(m, 0) for m in metrics_to_plot]
        finetuned_vals = [finetuned_metrics.get(m, 0) for m in metrics_to_plot]
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, baseline_vals, width, label="Pretrained (Baseline)",
                       color="#95A5A6", edgecolor="white", linewidth=0.8)
        bars2 = ax.bar(x + width/2, finetuned_vals, width, label="FinLLM (Fine-Tuned)",
                       color="#1A3C6E", edgecolor="white", linewidth=0.8)
        
        for bar, val in zip(bars1, baseline_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#666")
        for bar, val in zip(bars2, finetuned_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1A3C6E")
        
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Pretrained vs Fine-Tuned Model Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(CHART_BASELINE_COMPARE, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {CHART_BASELINE_COMPARE}")
    
    # 3. Class-wise performance
    print("Generating class-wise performance chart...")
    if valid_predictions:
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        class_prec = [precision_score(valid_true, valid_predictions, labels=[l], average="macro", zero_division=0) for l in LABEL_NAMES]
        class_rec = [recall_score(valid_true, valid_predictions, labels=[l], average="macro", zero_division=0) for l in LABEL_NAMES]
        class_f1 = [f1_score(valid_true, valid_predictions, labels=[l], average="macro", zero_division=0) for l in LABEL_NAMES]
        
        x = np.arange(len(LABEL_NAMES))
        width = 0.25
        colors_class = ["#1A3C6E", "#2980B9", "#7FB3D3"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, class_prec, width, label="Precision", color=colors_class[0])
        ax.bar(x, class_rec, width, label="Recall", color=colors_class[1])
        ax.bar(x + width, class_f1, width, label="F1-Score", color=colors_class[2])
        
        ax.set_xticks(x)
        ax.set_xticklabels([l.capitalize() for l in LABEL_NAMES], fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("FinLLM — Per-Class Classification Performance", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(CHART_CLASSWISE, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {CHART_CLASSWISE}")
    
    # 4. Trainable vs Frozen Parameters
    print("Generating trainable parameters chart...")
    if TRAINING_CONFIG_JSON.exists():
        with open(TRAINING_CONFIG_JSON) as f:
            train_config = json.load(f)
        
        total = train_config.get("total_parameters", 1)
        trainable = train_config.get("trainable_parameters", 0)
        frozen = total - trainable
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        sizes = [frozen, trainable]
        labels = [f"Frozen\n{frozen/1e6:.1f}M", f"Trainable\n{trainable/1e3:.0f}K"]
        colors = ["#BDC3C7", "#1A3C6E"]
        ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.2f%%",
                startangle=90, textprops={"fontsize": 11})
        ax1.set_title("Parameter Distribution", fontsize=13, fontweight="bold")
        
        # Bar comparison
        categories = ["Total", "Frozen", "Trainable"]
        values = [total/1e6, frozen/1e6, trainable/1e3]
        bar_colors = ["#7F8C8D", "#BDC3C7", "#1A3C6E"]
        
        ax2.bar(["Total\n(M)", "Frozen\n(M)", "Trainable\n(K)"],
                [total/1e6, frozen/1e6, trainable/1e3],
                color=bar_colors, edgecolor="white")
        ax2.set_ylabel("Count (M = Millions, K = Thousands)", fontsize=10)
        ax2.set_title("LoRA Parameter Efficiency", fontsize=13, fontweight="bold")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        
        for i, (cat, val) in enumerate(zip(["Total", "Frozen", "Trainable"],
                                            [total/1e6, frozen/1e6, trainable/1e3])):
            ax2.text(i, val + 0.5, f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
        
        fig.suptitle(f"LoRA Fine-Tuning — Parameter Counts\n"
                     f"(rank={train_config.get('lora_rank',8)}, "
                     f"alpha={train_config.get('lora_alpha',16)}, "
                     f"target={train_config.get('target_modules',[])})",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(CHART_TRAINABLE_PARAMS, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {CHART_TRAINABLE_PARAMS}")
    
    # 5. Training loss (already generated by train.py, but regenerate if logs exist)
    if TRAINING_LOGS_CSV.exists():
        logs_df = pd.read_csv(TRAINING_LOGS_CSV)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(logs_df["step"], logs_df["loss"], alpha=0.4, color="#3498DB", linewidth=0.8, label="Step Loss")
        ax.plot(logs_df["step"], logs_df["avg_loss"], color="#1A3C6E", linewidth=2.0, label="Avg Loss")
        
        if "val_loss" in logs_df.columns:
            val_rows = logs_df[logs_df["val_loss"].notna()]
            if len(val_rows) > 0:
                ax.scatter(val_rows["step"], val_rows["val_loss"],
                           color="#E74C3C", s=80, zorder=5, label="Validation Loss")
                # Also save validation loss chart
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.scatter(val_rows["step"], val_rows["val_loss"], color="#E74C3C", s=80)
                ax2.plot(val_rows["step"], val_rows["val_loss"], color="#E74C3C", linewidth=2)
                ax2.set_xlabel("Training Step")
                ax2.set_ylabel("Validation Loss")
                ax2.set_title("FinLLM Validation Loss")
                plt.tight_layout()
                plt.savefig(CHART_VALIDATION_LOSS, dpi=150, bbox_inches="tight")
                plt.close(fig2)
        
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("FinLLM Training Loss Curve", fontsize=14, fontweight="bold")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(CHART_TRAINING_LOSS, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {CHART_TRAINING_LOSS}")
    
    # 6. Print comparison table
    print_comparison_table(baseline_metrics, finetuned_metrics)


def print_comparison_table(baseline, finetuned):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("PRETRAINED vs FINE-TUNED MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'Pretrained':>15} {'FinLLM':>15} {'Difference':>15}")
    print("-" * 70)
    
    metrics = [
        ("Accuracy", "accuracy"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Macro F1-score", "macro_f1"),
        ("Weighted F1-score", "weighted_f1"),
        ("Invalid Predictions", "invalid_predictions"),
        ("Avg Inference (s)", "avg_inference_time_s"),
    ]
    
    for display_name, key in metrics:
        b_val = baseline.get(key, 0)
        f_val = finetuned.get(key, 0)
        diff = f_val - b_val if isinstance(b_val, (int, float)) else "N/A"
        
        if isinstance(b_val, float):
            print(f"  {display_name:<23} {b_val:>15.4f} {f_val:>15.4f} {diff:>+15.4f}")
        else:
            print(f"  {display_name:<23} {str(b_val):>15} {str(f_val):>15} {str(diff):>15}")
    
    print("=" * 70)
    
    # Save comparison to JSON
    comparison = {}
    for display_name, key in metrics:
        b_val = baseline.get(key, 0)
        f_val = finetuned.get(key, 0)
        comparison[key] = {
            "pretrained": b_val,
            "finetuned": f_val,
            "difference": f_val - b_val if isinstance(b_val, (int, float)) else None,
        }
    
    with open(RESULTS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    ensure_dirs()
    
    # Run evaluation
    finetuned_metrics, pred_df, valid_true, valid_predictions = evaluate_finetuned()
    
    # Generate charts
    generate_charts(finetuned_metrics, valid_true, valid_predictions)
    
    print("\nEvaluation complete!")
    print(f"Results: {RESULTS_DIR}")
    print(f"Charts:  {CHARTS_DIR}")
