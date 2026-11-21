"""
FinLLM Dataset Preparation
===========================
Processes the official Financial PhraseBank dataset (Sentences_50Agree.txt, 4846 sentences).
Creates stratified train (300), validation (80), and test (180) splits.
Saves CSV files and generates dataset statistics and class distribution charts.

Student: Sameet Pisal | PRN: 202401120018 | Roll: 41
Institute: MIT Academy of Engineering
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATASET_NAME, DATASET_CONFIG, LABEL_MAP, LABEL_NAMES,
    SEED, SYSTEM_PROMPT, TRAIN_CSV, VALIDATION_CSV, TEST_CSV,
    CHART_DATASET_DIST, RESULTS_DIR, RAW_DIR, PROCESSED_DIR,
    DATA_DIR, SYSTEM_INFO_JSON, ensure_dirs
)

import platform
import psutil


def collect_system_info():
    """Collect hardware and software configuration."""
    import torch
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "cpu": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": None,
        "gpu_name": "Intel Arc Graphics (Integrated)",
        "gpu_vram_gb": "~2 GB shared",
        "storage_free_gb": round(psutil.disk_usage("C:\\").free / (1024**3), 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    return info


def load_raw_phrasebank():
    """Load the official Sentences_50Agree.txt file."""
    txt_path = RAW_DIR / "FinancialPhraseBank-v1.0" / "Sentences_50Agree.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing {txt_path}. Check raw data extraction.")
    
    rows = []
    with open(txt_path, "r", encoding="iso-8859-1") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("@", 1)
            if len(parts) == 2:
                sentence, label = parts[0].strip(), parts[1].strip().lower()
                rows.append({"id": idx, "sentence": sentence, "label": label})
    
    df = pd.DataFrame(rows)
    # Verified label mapping
    label_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    df["label_id"] = df["label"].map(label_to_id)
    return df, label_to_id


def clean_dataset(df):
    """Deduplicate and clean sentences."""
    n_before = len(df)
    # Drop empty or null
    df = df.dropna(subset=["sentence", "label"])
    # Strip whitespace
    df["sentence"] = df["sentence"].str.strip()
    # Remove exact duplicate sentences (keep first)
    df = df.drop_duplicates(subset=["sentence"], keep="first")
    # Filter out sentences shorter than 10 characters
    df = df[df["sentence"].str.len() >= 10]
    n_after = len(df)
    print(f"      Initial: {n_before} -> After deduplication & cleaning: {n_after} (removed {n_before - n_after})")
    return df.reset_index(drop=True)


def create_stratified_splits(df):
    """
    Create stratified splits adhering to assignment parameters:
    - Train: 300 samples (stratified)
    - Validation: 80 samples (stratified)
    - Test: 180 samples (stratified held-out)
    """
    print("\n[4/8] Creating stratified splits (300 Train, 80 Validation, 180 Test, seed=42)...")
    
    # 1. First sample 180 test samples
    train_val_df, test_df = train_test_split(
        df,
        test_size=180,
        stratify=df["label"],
        random_state=SEED
    )
    
    # 2. Next sample 80 validation samples
    train_rest_df, val_df = train_test_split(
        train_val_df,
        test_size=80,
        stratify=train_val_df["label"],
        random_state=SEED
    )
    
    # 3. From train_rest_df, sample exactly 300 for the primary training run
    train_300_df, remaining_train_df = train_test_split(
        train_rest_df,
        train_size=300,
        stratify=train_rest_df["label"],
        random_state=SEED
    )
    
    # Format instruction-response
    for split_name, s_df in [("Train (300)", train_300_df), ("Validation (80)", val_df), ("Test (180)", test_df)]:
        dist = s_df["label"].value_counts()
        print(f"      {split_name}: {len(s_df)} sentences")
        for lbl in ["positive", "negative", "neutral"]:
            cnt = dist.get(lbl, 0)
            print(f"        {lbl:10s}: {cnt:3d} ({cnt/len(s_df)*100:.1f}%)")
            
    return (
        train_300_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        train_rest_df.reset_index(drop=True)
    )


def format_instruction(sentence, label_text):
    return {
        "sentence": sentence,
        "label": label_text,
        "instruction": SYSTEM_PROMPT,
        "user_input": sentence,
        "assistant_response": label_text.lower(),
    }


def add_instruction_format(df):
    records = []
    for _, row in df.iterrows():
        rec = format_instruction(row["sentence"], row["label"])
        rec["label_id"] = row["label_id"]
        records.append(rec)
    return pd.DataFrame(records)


def plot_dataset_distribution(train_df, val_df, test_df):
    """Generate 3-class distribution chart."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Financial PhraseBank — Class Distribution Across Splits", fontsize=14, fontweight="bold")
    
    colors = {"negative": "#E74C3C", "neutral": "#F39C12", "positive": "#27AE60"}
    
    for ax, (split_name, df) in zip(axes, [("Training (Primary)", train_df), ("Validation", val_df), ("Test (Held-out)", test_df)]):
        counts = df["label"].value_counts()
        ordered = {k: counts.get(k, 0) for k in ["negative", "neutral", "positive"]}
        bars = ax.bar(
            [k.capitalize() for k in ordered.keys()],
            list(ordered.values()),
            color=[colors[k] for k in ordered.keys()],
            edgecolor="white",
            linewidth=1.0
        )
        ax.set_title(f"{split_name}\n({len(df)} examples)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Sentence Count", fontsize=10)
        ax.set_ylim(0, max(ordered.values()) * 1.25)
        for bar, count in zip(bars, ordered.values()):
            pct = count / len(df) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(CHART_DATASET_DIST, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[6/8] Chart saved: {CHART_DATASET_DIST}")


def main():
    ensure_dirs()
    print("=" * 65)
    print("FinLLM — Dataset Preparation (Financial PhraseBank 50% Agreement)")
    print("=" * 65)
    
    # 0. System info
    print("\n[0/8] Collecting system information...")
    sys_info = collect_system_info()
    for k, v in sys_info.items():
        print(f"      {k:20s}: {v}")
    
    # 1. Load official dataset
    print("\n[1/8] Loading raw Sentences_50Agree.txt (4,846 sentences)...")
    df_raw, label_to_id = load_raw_phrasebank()
    print(f"      Total raw sentences: {len(df_raw)}")
    
    # 2. Inspect
    print("\n[2/8] Dataset inspection:")
    print(f"      Columns: {list(df_raw.columns)}")
    print(f"      Sample rows:\n{df_raw[['sentence', 'label']].head(3).to_string()}")
    print("\n      Raw class distribution:")
    for lbl, cnt in df_raw["label"].value_counts().items():
        print(f"        {lbl:10s}: {cnt:4d} ({cnt/len(df_raw)*100:.1f}%)")
    
    # 3. Clean
    print("\n[3/8] Cleaning and deduplication...")
    df_clean = clean_dataset(df_raw)
    
    # 4. Splits (300 train, 80 val, 180 test)
    train_df, val_df, test_df, full_train_df = create_stratified_splits(df_clean)
    
    # 5. Format instruction-response
    print("\n[5/8] Formatting instruction-response pairs...")
    train_inst = add_instruction_format(train_df)
    val_inst = add_instruction_format(val_df)
    test_inst = add_instruction_format(test_df)
    full_train_inst = add_instruction_format(full_train_df)
    
    # 6. Plot
    plot_dataset_distribution(train_inst, val_inst, test_inst)
    
    # 7. Save CSVs
    print("\n[7/8] Saving CSV datasets...")
    train_inst.to_csv(TRAIN_CSV, index=False)
    val_inst.to_csv(VALIDATION_CSV, index=False)
    test_inst.to_csv(TEST_CSV, index=False)
    full_train_inst.to_csv(DATA_DIR / "full_train.csv", index=False)
    print(f"      Train:      {TRAIN_CSV} ({len(train_inst)} rows)")
    print(f"      Validation: {VALIDATION_CSV} ({len(val_inst)} rows)")
    print(f"      Test:       {TEST_CSV} ({len(test_inst)} rows)")
    print(f"      Full Train: {DATA_DIR / 'full_train.csv'} ({len(full_train_inst)} rows)")
    
    # 8. Save statistics
    print("\n[8/8] Saving statistics and system information...")
    stats = {
        "dataset_name": "Financial PhraseBank",
        "dataset_subset": "sentences_50agree",
        "raw_total": len(df_raw),
        "clean_total": len(df_clean),
        "train_samples": len(train_inst),
        "val_samples": len(val_inst),
        "test_samples": len(test_inst),
        "full_available_train": len(full_train_inst),
        "label_mapping": label_to_id,
        "class_distribution_train": {k: int(v) for k, v in train_inst["label"].value_counts().items()},
        "class_distribution_val": {k: int(v) for k, v in val_inst["label"].value_counts().items()},
        "class_distribution_test": {k: int(v) for k, v in test_inst["label"].value_counts().items()},
        "random_seed": SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(RESULTS_DIR / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    sys_info["dataset_statistics"] = stats
    with open(SYSTEM_INFO_JSON, "w") as f:
        json.dump(sys_info, f, indent=2)
    
    print("=" * 65)
    print("Dataset preparation complete!")
    print(f"  Train:      {len(train_inst)} examples")
    print(f"  Validation: {len(val_inst)} examples")
    print(f"  Test:       {len(test_inst)} examples")
    print("=" * 65)


if __name__ == "__main__":
    main()
