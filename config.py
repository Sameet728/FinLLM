"""
FinLLM Configuration
====================
Central configuration for all project parameters, paths, and hyperparameters.
Modify this file to adjust training settings.

Student: Sameet Pisal | PRN: 202401120018 | Roll: 41
Institute: MIT Academy of Engineering
"""

import os
from pathlib import Path

# ─── Project Root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()

# ─── Directory Paths ──────────────────────────────────────────────────────────
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
MODELS_DIR      = PROJECT_ROOT / "models"
ADAPTER_DIR     = MODELS_DIR / "finllm_adapter"
RESULTS_DIR     = PROJECT_ROOT / "results"
CHARTS_DIR      = PROJECT_ROOT / "charts"
DIAGRAMS_DIR    = PROJECT_ROOT / "diagrams"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
NOTEBOOKS_DIR   = PROJECT_ROOT / "notebooks"
REPORT_DIR      = PROJECT_ROOT / "report"
TESTS_DIR       = PROJECT_ROOT / "tests"

# ─── Data File Paths ──────────────────────────────────────────────────────────
TRAIN_CSV      = DATA_DIR / "train.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
TEST_CSV       = DATA_DIR / "test.csv"

# ─── Results File Paths ───────────────────────────────────────────────────────
BASELINE_METRICS_JSON     = RESULTS_DIR / "baseline_metrics.json"
FINETUNED_METRICS_JSON    = RESULTS_DIR / "finetuned_metrics.json"
PREDICTIONS_CSV           = RESULTS_DIR / "predictions.csv"
TRAINING_LOGS_CSV         = RESULTS_DIR / "training_logs.csv"
CLASSIFICATION_REPORT_CSV = RESULTS_DIR / "classification_report.csv"
SYSTEM_INFO_JSON          = RESULTS_DIR / "system_information.json"
TRAINING_CONFIG_JSON      = RESULTS_DIR / "training_configuration.json"

# ─── Chart Paths ──────────────────────────────────────────────────────────────
CHART_DATASET_DIST     = CHARTS_DIR / "dataset_distribution.png"
CHART_TRAINING_LOSS    = CHARTS_DIR / "training_loss.png"
CHART_VALIDATION_LOSS  = CHARTS_DIR / "validation_loss.png"
CHART_CONFUSION_MATRIX = CHARTS_DIR / "confusion_matrix.png"
CHART_BASELINE_COMPARE = CHARTS_DIR / "baseline_comparison.png"
CHART_CLASSWISE        = CHARTS_DIR / "classwise_performance.png"
CHART_TRAINABLE_PARAMS = CHARTS_DIR / "trainable_parameters.png"

# ─── Diagram Paths ────────────────────────────────────────────────────────────
DIAG_SYSTEM_ARCH     = DIAGRAMS_DIR / "system_architecture.png"
DIAG_TRANSFORMER     = DIAGRAMS_DIR / "transformer_architecture.png"
DIAG_LORA            = DIAGRAMS_DIR / "lora_architecture.png"
DIAG_TRAINING        = DIAGRAMS_DIR / "training_pipeline.png"
DIAG_INFERENCE       = DIAGRAMS_DIR / "inference_pipeline.png"

# ─── Report Paths ─────────────────────────────────────────────────────────────
REPORT_DOCX = REPORT_DIR / "Assignment_2_FinLLM_Sameet_Pisal_202401120018.docx"
REPORT_PDF  = REPORT_DIR / "Assignment_2_FinLLM_Sameet_Pisal_202401120018.pdf"

# ─── Model & Dataset ──────────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME    = "takala/financial_phrasebank"
DATASET_CONFIG  = "sentences_50agree"

# ─── Label Mapping ────────────────────────────────────────────────────────────
# Verified from dataset features (ClassLabel: 0=negative, 1=neutral, 2=positive)
LABEL_MAP    = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_NAMES  = ["negative", "neutral", "positive"]
NUM_LABELS   = 3

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a financial sentiment analysis model. "
    "Classify the provided financial text based on its implications for the "
    "company's financial performance. "
    "Respond with exactly one label: positive, negative or neutral."
)

# ─── Data Splits ──────────────────────────────────────────────────────────────
TRAIN_RATIO      = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO       = 0.10
SEED             = 42

# ─── LoRA Hyperparameters ─────────────────────────────────────────────────────
LORA_RANK       = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["q_proj", "v_proj"]

# ─── Training Hyperparameters ─────────────────────────────────────────────────
# CPU-optimised configuration (no CUDA detected)
LEARNING_RATE   = 2e-4
NUM_EPOCHS      = 1
BATCH_SIZE      = 1          # small batch for CPU memory
GRAD_ACCUM      = 4          # effective batch size = 4
MAX_SEQ_LEN     = 128        # shorter = faster on CPU
TRAIN_SUBSET    = 300        # stratified subset from full training set
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1
LR_SCHEDULER    = "cosine"   # cosine annealing
OPTIMIZER       = "adamw_torch"

# ─── Inference Settings ───────────────────────────────────────────────────────
MAX_NEW_TOKENS  = 10         # only need one token (positive/negative/neutral)
DO_SAMPLE       = False      # greedy decoding for reproducibility
TEMPERATURE     = 1.0
REPETITION_PENALTY = 1.1

# ─── Assignment Information ───────────────────────────────────────────────────
STUDENT_NAME    = "Sameet Pisal"
PRN             = "202401120018"
ROLL_NUMBER     = "41"
INSTITUTE       = "MIT Academy of Engineering"
ASSIGNMENT_NO   = "2"
SUBMISSION_DATE = "21 November 2026"
PROJECT_TITLE   = "FinLLM — Development and Fine-Tuning of an Open-Source Large Language Model for Financial News Sentiment Analysis"


def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, ADAPTER_DIR,
              RESULTS_DIR, CHARTS_DIR, DIAGRAMS_DIR, SCREENSHOTS_DIR,
              NOTEBOOKS_DIR, REPORT_DIR, TESTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Configuration loaded successfully.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME} ({DATASET_CONFIG})")
    print(f"LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print(f"Training subset: {TRAIN_SUBSET} examples")
