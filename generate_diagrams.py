"""
FinLLM Architecture Diagrams
==============================
Generates all required architecture diagrams as high-resolution PNGs.
Uses matplotlib only (no external graphviz dependency required).

Usage:
    python generate_diagrams.py

Student: Sameet Pisal | PRN: 202401120018 | Roll: 41
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from config import DIAGRAMS_DIR, ensure_dirs


NAVY    = "#1A3C6E"
BLUE    = "#2980B9"
LBLUE   = "#AED6F1"
GREEN   = "#27AE60"
RED     = "#E74C3C"
AMBER   = "#F39C12"
GRAY    = "#95A5A6"
WHITE   = "#FFFFFF"
LGRAY   = "#F2F3F4"


def add_box(ax, x, y, w, h, text, fc=LBLUE, ec=NAVY, fontsize=9, bold=False):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.02",
                          facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color=NAVY, wrap=True)


def add_arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))


# ─── 1. System Architecture ───────────────────────────────────────────────────

def draw_system_architecture():
    fig, ax = plt.subplots(figsize=(8, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis("off")
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)
    
    ax.text(5, 17.3, "FinLLM — System Architecture", ha="center", va="center",
            fontsize=14, fontweight="bold", color=NAVY)
    
    boxes = [
        (5, 16.0, 7, 0.7, "Financial PhraseBank Dataset\n(4,846 sentences, 50% agreement)", LBLUE),
        (5, 14.7, 7, 0.7, "Data Cleaning & Deduplication", LBLUE),
        (5, 13.4, 7, 0.7, "Stratified Train / Validation / Test Split\n(80% / 10% / 10%)", LBLUE),
        (5, 12.1, 7, 0.7, "Instruction Formatting & Tokenization\n(Qwen Chat Template)", LBLUE),
        (5, 10.8, 7, 0.7, "Pretrained Qwen2.5-0.5B-Instruct\n(494M parameters, frozen)", "#D6EAF8"),
        (5,  9.5, 7, 0.7, "LoRA Adapter Configuration\n(rank=8, alpha=16, q_proj + v_proj)", "#D5F5E3"),
        (5,  8.2, 7, 0.7, "Supervised Fine-Tuning\n(CPU Training, 300 examples, 1 epoch)", "#D5F5E3"),
        (5,  6.9, 7, 0.7, "Trained FinLLM Adapter\n(~786K trainable parameters)", "#A9DFBF"),
        (5,  5.6, 7, 0.7, "Model Evaluation\n(Accuracy, F1, Confusion Matrix)", LBLUE),
        (5,  4.3, 7, 0.7, "Results & Visualizations\n(Charts, CSVs, JSONs)", LBLUE),
        (5,  3.0, 7, 0.7, "Streamlit Web Application\n(FinLLM Dashboard)", "#FADBD8"),
    ]
    
    for (x, y, w, h, text, color) in boxes:
        add_box(ax, x, y, w, h, text, fc=color, fontsize=9)
    
    ys = [b[1] for b in boxes]
    for i in range(len(ys) - 1):
        add_arrow(ax, 5, ys[i] - 0.35, 5, ys[i+1] + 0.35)
    
    plt.tight_layout()
    out = DIAGRAMS_DIR / "system_architecture.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved: {out}")


# ─── 2. Transformer Architecture ─────────────────────────────────────────────

def draw_transformer_architecture():
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)
    
    ax.text(5, 13.5, "Qwen2.5-0.5B-Instruct — Decoder Architecture",
            ha="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(5, 13.0, "(24 Transformer Decoder Layers)", ha="center", fontsize=10, color=GRAY)
    
    # Input
    add_box(ax, 5, 12.3, 6, 0.55, "Input Tokens (Financial News Text)", LBLUE, fontsize=9)
    
    # Embedding
    add_box(ax, 5, 11.4, 6, 0.55, "Token Embedding Layer\n(vocab_size=151,936 → d_model=896)", LBLUE, fontsize=9)
    
    # Rotary PE
    add_box(ax, 5, 10.5, 6, 0.55, "Rotary Positional Embedding (RoPE)", "#E8F4FD", fontsize=9)
    
    # Decoder block (repeated)
    block_colors = ["#D6EAF8", "#EBF5FB", "#D6EAF8"]
    block_labels = [
        "RMS Normalization",
        "Grouped-Query Self-Attention\n(14 KV heads, 7 Q heads)",
        "RMS Normalization\n+ SwiGLU Feed-Forward Network\n(hidden_dim=4864)",
    ]
    
    ax.text(5, 9.7, "Decoder Layer × 24", ha="center", fontsize=10,
            fontweight="bold", color=NAVY)
    ax.add_patch(mpatches.FancyBboxPatch(
        (1.5, 7.3), 7, 2.3, boxstyle="round,pad=0.1",
        facecolor="#F0F8FF", edgecolor=NAVY, linewidth=2, linestyle="--"
    ))
    
    add_box(ax, 5, 9.3, 5.5, 0.45, block_labels[0], block_colors[0], fontsize=8.5)
    add_box(ax, 5, 8.6, 5.5, 0.55, block_labels[1], block_colors[1], fontsize=8.5)
    add_box(ax, 5, 7.65, 5.5, 0.65, block_labels[2], block_colors[2], fontsize=8.5)
    
    # Residual connection annotation
    ax.annotate("Residual\nConnection", xy=(8.8, 8.8), fontsize=7.5,
                color=GRAY, ha="center")
    ax.annotate("", xy=(8.8, 7.3), xytext=(8.8, 9.3),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0,
                                connectionstyle="arc3,rad=-0.5"))
    
    add_arrow(ax, 5, 11.7, 5, 11.65)
    add_arrow(ax, 5, 11.1, 5, 10.75)
    add_arrow(ax, 5, 10.2, 5, 9.55)
    add_arrow(ax, 5, 9.0, 5, 8.85)
    add_arrow(ax, 5, 8.3, 5, 7.95)
    
    # Output
    add_box(ax, 5, 6.8, 6, 0.55, "RMS Layer Normalization (output)", LBLUE, fontsize=9)
    add_box(ax, 5, 6.0, 6, 0.55, "Linear Projection (Language Model Head)\nd_model=896 → vocab_size=151,936", LBLUE, fontsize=9)
    add_box(ax, 5, 5.2, 6, 0.55, "Softmax → Next Token Probabilities", "#D5F5E3", fontsize=9)
    add_box(ax, 5, 4.4, 6, 0.55, "Autoregressive Decoding (Greedy)\n→ Sentiment Label Token", "#A9DFBF", fontsize=9)
    
    for y_pair in [(7.1, 6.8), (6.3, 5.5), (4.9, 4.7)]:
        add_arrow(ax, 5, y_pair[0], 5, y_pair[1])
    add_arrow(ax, 5, 7.3, 5, 7.1)
    
    # Key stats
    ax.text(0.3, 1.5, "Key Architecture Details:", fontsize=10, fontweight="bold", color=NAVY)
    stats = [
        "• Parameters: ~494M total",
        "• Layers: 24 decoder blocks",
        "• Hidden dim: 896",
        "• Attention heads: 14 (Q), 7 (KV) — Grouped-Query Attention",
        "• FFN dim: 4864 (SwiGLU activation)",
        "• Context length: 128K tokens",
        "• Vocabulary: 151,936 tokens (Qwen tokenizer)",
    ]
    for i, stat in enumerate(stats):
        ax.text(0.3, 1.1 - i * 0.22, stat, fontsize=8, color=NAVY)
    
    plt.tight_layout()
    out = DIAGRAMS_DIR / "transformer_architecture.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved: {out}")


# ─── 3. LoRA Architecture ─────────────────────────────────────────────────────

def draw_lora_architecture():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)
    
    ax.text(6, 8.5, "LoRA Architecture — W' = W₀ + (α/r) × B × A",
            ha="center", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(6, 8.0, "W₀: Frozen pretrained weights  |  B, A: Trainable LoRA matrices  |  r=8, α=16",
            ha="center", fontsize=10, color=GRAY)
    
    # Input
    add_box(ax, 6, 7.0, 3, 0.55, "Input Activation (x)", LBLUE, fontsize=10, bold=True)
    
    # Main path (frozen)
    add_box(ax, 3.0, 5.0, 4.0, 1.4, "W₀  (Frozen)\nPretrained Weight Matrix\n(d × d, e.g. 896 × 896)", GRAY, fontsize=9)
    ax.text(3.0, 3.8, "🔒 FROZEN — no gradient update", ha="center", fontsize=8.5, color=GRAY)
    
    # LoRA path (trainable)
    add_box(ax, 9.0, 5.8, 3.5, 0.9, "A  (Trainable)\nLow-Rank Down-Projection\n(d × r = 896 × 8)", "#D5F5E3", fontsize=8.5)
    add_box(ax, 9.0, 4.5, 3.5, 0.9, "B  (Trainable)\nLow-Rank Up-Projection\n(r × d = 8 × 896)", "#A9DFBF", fontsize=8.5)
    ax.text(9.0, 3.8, "✅ TRAINABLE — gradient flows", ha="center", fontsize=8.5, color=GREEN)
    
    # Scaling
    add_box(ax, 9.0, 3.3, 3.5, 0.6, "Scale by α/r = 16/8 = 2.0", "#FEF9E7", AMBER, fontsize=9)
    
    # Addition
    add_box(ax, 6, 2.2, 3, 0.65, "⊕  Addition\n W' × x = W₀x + (α/r)BAx", "#FADBD8", RED, fontsize=9, bold=True)
    
    # Output
    add_box(ax, 6, 1.0, 3, 0.55, "Output: W'x", "#D5F5E3", GREEN, fontsize=10, bold=True)
    
    # Arrows
    # Input → W0
    add_arrow(ax, 4.5, 6.72, 3.5, 6.0)
    # Input → A
    add_arrow(ax, 7.5, 6.72, 8.0, 6.3)
    # A → B
    add_arrow(ax, 9.0, 5.35, 9.0, 4.95)
    # B → scale
    add_arrow(ax, 9.0, 4.05, 9.0, 3.6)
    # W0 → addition
    add_arrow(ax, 3.0, 3.3, 4.8, 2.5)
    # scale → addition
    add_arrow(ax, 7.5, 3.3, 7.3, 2.5)
    # addition → output
    add_arrow(ax, 6, 1.87, 6, 1.28)
    
    # Parameter savings annotation
    ax.text(0.3, 1.5,
            "Parameter Efficiency:\n"
            "Full fine-tuning:  d×d = 896×896 = 802,816 per matrix\n"
            "LoRA fine-tuning:  d×r + r×d = 896×8 + 8×896 = 14,336 per matrix\n"
            "Reduction factor:  56× fewer trainable parameters per layer",
            fontsize=9, color=NAVY,
            bbox=dict(boxstyle="round", facecolor="#EBF5FB", edgecolor=NAVY, alpha=0.7))
    
    plt.tight_layout()
    out = DIAGRAMS_DIR / "lora_architecture.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved: {out}")


# ─── 4. Training Pipeline ─────────────────────────────────────────────────────

def draw_training_pipeline():
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)
    
    ax.text(5, 12.5, "FinLLM Training Pipeline", ha="center",
            fontsize=14, fontweight="bold", color=NAVY)
    
    steps = [
        (5, 11.6, 7.5, 0.65, "Raw Financial PhraseBank\n(4,846 labeled sentences)", LBLUE),
        (5, 10.5, 7.5, 0.65, "Data Cleaning\n(remove duplicates, NaN, short texts)", LBLUE),
        (5,  9.4, 7.5, 0.65, "Stratified Split (seed=42)\nTrain: 80% | Val: 10% | Test: 10%", LBLUE),
        (5,  8.3, 7.5, 0.65, "Training Subset\n(300 stratified examples for CPU)", LBLUE),
        (5,  7.2, 7.5, 0.65, "Instruction Formatting → Tokenization\n(Qwen Chat Template, max_len=128)", LBLUE),
        (5,  6.1, 7.5, 0.65, "Answer-Only Loss Masking\n(labels=-100 for instruction tokens)", "#D6EAF8"),
        (5,  5.0, 7.5, 0.65, "LoRA Fine-Tuning\n(1 epoch, lr=2e-4, AdamW + cosine scheduler)", "#D5F5E3"),
        (5,  3.9, 7.5, 0.65, "Backpropagation\n(gradients flow only through A and B matrices)", "#D5F5E3"),
        (5,  2.8, 7.5, 0.65, "Adapter Saved\n(adapter_model.safetensors + tokenizer)", "#A9DFBF"),
        (5,  1.7, 7.5, 0.65, "Model Evaluation on Held-Out Test Set\n(same 10% test set as baseline)", LBLUE),
    ]
    
    for (x, y, w, h, text, color) in steps:
        add_box(ax, x, y, w, h, text, fc=color, fontsize=9)
    
    ys = [s[1] for s in steps]
    for i in range(len(ys) - 1):
        add_arrow(ax, 5, ys[i] - 0.33, 5, ys[i+1] + 0.33)
    
    plt.tight_layout()
    out = DIAGRAMS_DIR / "training_pipeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved: {out}")


# ─── 5. Inference Pipeline ────────────────────────────────────────────────────

def draw_inference_pipeline():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)
    
    ax.text(5, 11.5, "FinLLM Inference Pipeline", ha="center",
            fontsize=14, fontweight="bold", color=NAVY)
    
    steps = [
        (5, 10.5, 7, 0.65, "User Financial News Input\n(e.g. 'Revenue increased by 25%…')", "#FADBD8"),
        (5,  9.4, 7, 0.65, "Chat Template Formatting\n(System + User messages → Qwen format)", LBLUE),
        (5,  8.3, 7, 0.65, "Tokenizer\n(Qwen tokenizer → token IDs)", LBLUE),
        (5,  7.2, 7, 0.65, "Pretrained Qwen2.5-0.5B\n(frozen weights W₀)", "#D6EAF8"),
        (5,  6.1, 7, 0.65, "LoRA Adapter Applied\n(W' = W₀ + (α/r)BA for q_proj, v_proj)", "#D5F5E3"),
        (5,  5.0, 7, 0.65, "Autoregressive Generation\n(greedy decoding, max_new_tokens=10)", LBLUE),
        (5,  3.9, 7, 0.65, "Output Token Parsing\n(extract: positive / negative / neutral)", LBLUE),
        (5,  2.8, 7, 0.65, "Confidence Score\n(softmax over 3 label token logits)", "#FEF9E7"),
        (5,  1.7, 7, 0.65, "Sentiment Result + Visualization\n(displayed in Streamlit dashboard)", "#A9DFBF"),
    ]
    
    for (x, y, w, h, text, color) in steps:
        add_box(ax, x, y, w, h, text, fc=color, fontsize=9)
    
    ys = [s[1] for s in steps]
    for i in range(len(ys) - 1):
        add_arrow(ax, 5, ys[i] - 0.33, 5, ys[i+1] + 0.33)
    
    plt.tight_layout()
    out = DIAGRAMS_DIR / "inference_pipeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    print("=" * 60)
    print("FinLLM — Generating Architecture Diagrams")
    print("=" * 60)
    
    draw_system_architecture()
    draw_transformer_architecture()
    draw_lora_architecture()
    draw_training_pipeline()
    draw_inference_pipeline()
    
    print("\nAll diagrams generated successfully!")
    for f in DIAGRAMS_DIR.glob("*.png"):
        print(f"  {f}")


if __name__ == "__main__":
    main()
