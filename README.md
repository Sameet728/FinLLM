# FinLLM — Financial News Sentiment Analysis

> **Assignment 2: Individual Development of a Large Language Model (LLM)**  
> Student: Sameet Pisal | PRN: 202401120018 | Roll: 41  
> Department: Department of Data Science | Institute: MIT Academy of Engineering  
> Submission Date: 21 November 2026  
> **Live Public Web Dashboard:** [https://sameetfinllm.streamlit.app](https://sameetfinllm.streamlit.app)  
> **GitHub Repository:** [https://github.com/Sameet728/FinLLM](https://github.com/Sameet728/FinLLM)  
> **Local Dashboard:** `http://localhost:8501`

---

## Project Overview

**FinLLM** is a domain-specific Large Language Model fine-tuned for financial news sentiment analysis. It uses **Qwen2.5-0.5B-Instruct** as the pretrained base model and applies **LoRA (Low-Rank Adaptation)** parameter-efficient fine-tuning on the **Financial PhraseBank** dataset to classify financial news into three categories:

| Sentiment | Example |
|---|---|
| ✅ **Positive** | "The company reported a 30% increase in quarterly revenue." |
| ❌ **Negative** | "Net losses widened significantly due to declining market demand." |
| ➖ **Neutral** | "The board appointed a new independent director on Monday." |

---

## Features

- **Genuine LoRA Fine-Tuning** — real parameter-efficient fine-tuning, not a chatbot wrapper
- **Financial Sentiment Classification** — understands financial terminology and domain context
- **Pretrained vs Fine-Tuned Comparison** — demonstrates measurable model improvement
- **Professional Streamlit Dashboard** — financial-grade analytics UI
- **Batch Analysis** — process multiple news items via CSV upload
- **Confidence Scoring** — softmax over label token logits (documented method)
- **Google Colab Notebook** — GPU training alternative

---

## Technology Stack

| Component | Technology |
|---|---|
| Pretrained LLM | Qwen/Qwen2.5-0.5B-Instruct |
| Fine-tuning | LoRA via PEFT |
| Dataset | Financial PhraseBank (Malo et al. 2014) |
| Framework | PyTorch + Hugging Face Transformers |
| Frontend | Streamlit |
| Evaluation | scikit-learn |
| Report | python-docx + docx2pdf |

---

## Quick Start (Windows)

### 1. Prerequisites
- Python 3.10–3.13
- 8+ GB RAM
- 10+ GB free disk space
- Internet connection (for initial model download)

### 2. Setup
```powershell
# Create project directory
cd C:\Users\samee\Desktop\LLM\FinLLM

# Install dependencies
python -m pip install -r requirements.txt
```

### 3. Run Pipeline
```powershell
# Step 1: Prepare dataset
python prepare_dataset.py

# Step 2: Evaluate pretrained baseline
python baseline.py

# Step 3: Fine-tune with LoRA
python train.py

# Step 4: Evaluate fine-tuned model
python evaluate.py

# Step 5: Generate architecture diagrams
python generate_diagrams.py

# Step 6: Launch web application
streamlit run app.py

# Step 7: Generate academic report
python generate_report.py
```

---

## LoRA Fine-Tuning Configuration

```
Model:          Qwen/Qwen2.5-0.5B-Instruct
LoRA rank (r):  8
LoRA alpha:     16
LoRA dropout:   0.05
Target modules: q_proj, v_proj
Learning rate:  2e-4
Epochs:         1
Batch size:     1 (grad accum: 4 → effective: 4)
Max seq len:    128
Training data:  256 examples (stratified, seed=42)
Device:         CPU (Intel Core Ultra 5 125H)
```

### Mathematical Foundation

LoRA adapts pretrained weights using:

```
W' = W₀ + (α/r) × B × A
```

Where:
- `W₀` = frozen pretrained weight matrix
- `B` ∈ ℝ^(d×r), `A` ∈ ℝ^(r×d) = trainable low-rank matrices
- `r = 8` = LoRA rank
- `α = 16` = scaling factor

This reduces trainable parameters from ~494M (full fine-tuning) to ~786K (0.16%) — a 630× reduction.

---

## Hardware Requirements

| Component | Minimum | Used in Project |
|---|---|---|
| CPU | 4 cores | Intel Core Ultra 5 125H (14 cores) |
| RAM | 8 GB | 15.71 GB |
| Disk | 10 GB | 22 GB free |
| GPU | Optional | Intel Arc (integrated, not used) |
| CUDA | Not required | Not available |

---

## Project Structure

```
FinLLM/
├── app.py                      # Streamlit web application
├── config.py                   # Central configuration
├── prepare_dataset.py          # Dataset download and preprocessing
├── build_dataset.py            # Local dataset builder (offline fallback)
├── baseline.py                 # Pretrained model evaluation
├── train.py                    # LoRA fine-tuning
├── evaluate.py                 # Fine-tuned model evaluation
├── inference.py                # Shared inference utilities
├── generate_diagrams.py        # Architecture diagram generation
├── generate_report.py          # Academic report generation (DOCX + PDF)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/                    # Raw downloaded dataset
│   ├── processed/
│   ├── train.csv               # 256 training examples
│   ├── validation.csv          # 32 validation examples
│   └── test.csv                # 33 held-out test examples
│
├── models/
│   └── finllm_adapter/         # Saved LoRA adapter weights
│
├── results/                    # JSON and CSV evaluation results
├── charts/                     # Performance visualization PNGs
├── diagrams/                   # Architecture diagram PNGs
├── screenshots/                # Implementation screenshots
├── notebooks/
│   └── FinLLM_Training.ipynb   # Google Colab GPU training notebook
└── report/                     # Final DOCX and PDF report
```

---

## Evaluation Results

*Results populated after running the full pipeline. See `results/` directory for actual values.*

| Metric | Pretrained | FinLLM (Fine-Tuned) |
|---|---|---|
| Accuracy | — | — |
| Macro F1 | — | — |
| Macro Precision | — | — |
| Macro Recall | — | — |
| Avg Inference | — | — |

---

## Dataset Information

| Property | Value |
|---|---|
| Source | Financial PhraseBank (Malo et al. 2014) |
| Configuration | sentences_50agree |
| Total examples | 321 (local representative dataset) |
| Training split | 256 (80%) |
| Validation split | 32 (10%) |
| Test split | 33 (10%) |
| Labels | negative (0), neutral (1), positive (2) |
| Annotation | Human expert agreement (≥50% annotators) |

---

## Limitations

1. **Small model** — Qwen2.5-0.5B has limited capacity vs larger LLMs
2. **CPU training** — training on CPU limits training data size and epochs
3. **Dataset size** — 321 sentences is a representative but small training set
4. **Financial language ambiguity** — mixed-sentiment news may be misclassified
5. **Historical data** — no real-time market context
6. **Not trading advice** — sentiment classification ≠ stock price prediction

---

## References

1. Malo, P. et al. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *JASIST*, 65(4), 782-796.
2. Qwen Team. (2024). Qwen2.5 Technical Report. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
3. Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*
4. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*
5. Yang, Y. et al. (2020). FinBERT: A Pretrained Language Model for Financial Communications. *arXiv:2006.08097*
6. Wolf, T. et al. (2020). HuggingFace's Transformers. *EMNLP*

---

## License

- **Model (Qwen2.5-0.5B-Instruct):** Apache 2.0
- **Dataset (Financial PhraseBank):** Research use
- **This project:** Educational and non-commercial use only
