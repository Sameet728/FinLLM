"""
FinLLM LoRA Fine-Tuning
========================
Fine-tunes Qwen2.5-0.5B-Instruct using LoRA (Parameter-Efficient Fine-Tuning).
Implements answer-only loss masking for supervised fine-tuning.

Mathematical foundation:
    W' = W₀ + (α/r) * B * A
    where W₀ = frozen pretrained weights,
          B, A = trainable low-rank matrices,
          r = LoRA rank,
          α = scaling factor

Usage:
    python train.py

Student: Sameet Pisal | PRN: 202401120018 | Roll: 41
"""

import copy
import json
import math
import os
import random
import time
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from config import (
    MODEL_NAME, ADAPTER_DIR, TRAIN_CSV, VALIDATION_CSV,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM,
    MAX_SEQ_LEN, TRAIN_SUBSET, WEIGHT_DECAY, WARMUP_RATIO,
    SEED, RESULTS_DIR, CHARTS_DIR, TRAINING_LOGS_CSV,
    TRAINING_CONFIG_JSON, CHART_TRAINING_LOSS, LABEL_NAMES,
    ensure_dirs
)
from inference import build_training_text


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FinancialSFTDataset(Dataset):
    """
    Supervised Fine-Tuning dataset with answer-only loss masking.
    
    For each example:
    - Tokenize the full instruction-response text
    - Set labels = -100 for all tokens except the assistant response
    - Only the label tokens (positive/negative/neutral) contribute to loss
    """
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = MAX_SEQ_LEN):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        for _, row in df.iterrows():
            self._process_row(row)
    
    def _process_row(self, row):
        sentence = str(row["sentence"])
        label = str(row["label"]).lower()
        
        # Build the full training text using chat template
        full_text = build_training_text(sentence, label, self.tokenizer)
        
        # Tokenize full text
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            return_tensors=None,
            add_special_tokens=False,
        )
        input_ids = full_enc["input_ids"]
        
        # Build labels with -100 masking (only assistant answer contributes to loss)
        # Find the assistant response prefix to know where labels start
        # Qwen format: <|im_start|>assistant\n{label}<|im_end|>
        assistant_prefix = "<|im_start|>assistant\n"
        prefix_ids = self.tokenizer.encode(assistant_prefix, add_special_tokens=False)
        
        labels = [-100] * len(input_ids)
        
        # Find where assistant tokens begin in input_ids
        prefix_len = len(prefix_ids)
        for start_idx in range(len(input_ids) - prefix_len + 1):
            if input_ids[start_idx:start_idx + prefix_len] == prefix_ids:
                # Unmask tokens after the prefix
                for j in range(start_idx + prefix_len, len(input_ids)):
                    labels[j] = input_ids[j]
                break
        
        if all(l == -100 for l in labels):
            # Fallback: unmask last few tokens (the label word)
            for j in range(max(0, len(input_ids) - 5), len(input_ids)):
                labels[j] = input_ids[j]
        
        self.examples.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_token_id: int):
    """Left-pad sequences in a batch to the same length."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    
    input_ids_list = []
    labels_list = []
    attention_masks = []
    
    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len
        
        padded_input = torch.cat([
            torch.full((pad_len,), pad_token_id, dtype=torch.long),
            item["input_ids"]
        ])
        padded_labels = torch.cat([
            torch.full((pad_len,), -100, dtype=torch.long),
            item["labels"]
        ])
        attn_mask = torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            torch.ones(seq_len, dtype=torch.long)
        ])
        
        input_ids_list.append(padded_input)
        labels_list.append(padded_labels)
        attention_masks.append(attn_mask)
    
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_masks),
    }


# ─── Parameter Counting ───────────────────────────────────────────────────────

def count_parameters(model) -> dict:
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    pct = (trainable / total) * 100 if total > 0 else 0.0
    
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(frozen),
        "trainable_percentage": round(pct, 4),
    }


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    set_seed(SEED)
    ensure_dirs()
    
    print("=" * 60)
    print("FinLLM — LoRA Fine-Tuning")
    print("=" * 60)
    
    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading training data...")
    if not TRAIN_CSV.exists():
        print("ERROR: Run prepare_dataset.py first.")
        sys.exit(1)
    
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VALIDATION_CSV)
    
    # Create stratified training subset for CPU feasibility
    if TRAIN_SUBSET and len(train_df) > TRAIN_SUBSET:
        from sklearn.model_selection import train_test_split
        train_df, _ = train_test_split(
            train_df,
            train_size=TRAIN_SUBSET,
            stratify=train_df["label"],
            random_state=SEED
        )
        train_df = train_df.reset_index(drop=True)
        print(f"      Stratified subset: {TRAIN_SUBSET} examples (from full training set)")
    
    print(f"      Train: {len(train_df)}, Validation: {len(val_df)}")
    
    # ── Load model & tokenizer ───────────────────────────────────────────────
    print(f"\n[2/7] Loading model: {MODEL_NAME}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # ── Capture initial adapter weights for change verification ─────────────
    # We'll compare these after training to prove weights actually changed
    
    # ── LoRA configuration ───────────────────────────────────────────────────
    print(f"\n[3/7] Configuring LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Save initial weights for change verification
    initial_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_weights[name] = param.data.clone()
    
    # ── Parameter counts ─────────────────────────────────────────────────────
    param_info = count_parameters(model)
    print(f"\n      LoRA Configuration:")
    print(f"        rank (r)         : {LORA_RANK}")
    print(f"        alpha            : {LORA_ALPHA}")
    print(f"        dropout          : {LORA_DROPOUT}")
    print(f"        target modules   : {TARGET_MODULES}")
    print(f"\n      Parameter Counts:")
    print(f"        Total parameters     : {param_info['total_parameters']:,}")
    print(f"        Trainable parameters : {param_info['trainable_parameters']:,}")
    print(f"        Frozen parameters    : {param_info['frozen_parameters']:,}")
    print(f"        Trainable %          : {param_info['trainable_percentage']:.4f}%")
    model.print_trainable_parameters()
    
    # ── Prepare datasets ─────────────────────────────────────────────────────
    print(f"\n[4/7] Preparing tokenized datasets...")
    train_dataset = FinancialSFTDataset(train_df, tokenizer, max_len=MAX_SEQ_LEN)
    val_dataset = FinancialSFTDataset(val_df, tokenizer, max_len=MAX_SEQ_LEN)
    
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    
    print(f"      Training batches: {len(train_loader)}")
    print(f"      Validation batches: {len(val_loader)}")
    
    # ── Optimizer & scheduler ────────────────────────────────────────────────
    total_steps = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    
    # Cosine scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
    
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    print(f"\n      Training configuration:")
    print(f"        Epochs         : {NUM_EPOCHS}")
    print(f"        Train examples : {len(train_dataset)}")
    print(f"        Batch size     : {BATCH_SIZE}")
    print(f"        Grad accum     : {GRAD_ACCUM} (effective batch = {BATCH_SIZE * GRAD_ACCUM})")
    print(f"        Total steps    : {total_steps}")
    print(f"        Warmup steps   : {warmup_steps}")
    print(f"        Learning rate  : {LEARNING_RATE}")
    print(f"        Weight decay   : {WEIGHT_DECAY}")
    print(f"        Max seq len    : {MAX_SEQ_LEN}")
    print(f"        Device         : CPU")
    
    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n[5/7] Training...")
    print("-" * 60)
    
    training_logs = []
    global_step = 0
    start_time = time.time()
    
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRAD_ACCUM
            
            # Backward pass
            loss.backward()
            epoch_loss += outputs.loss.item()
            
            # Gradient accumulation step
            if (batch_idx + 1) % GRAD_ACCUM == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / (batch_idx + 1)
                lr_now = scheduler.get_last_lr()[0]
                
                log_entry = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "batch": batch_idx + 1,
                    "loss": round(outputs.loss.item(), 6),
                    "avg_loss": round(avg_loss, 6),
                    "lr": round(lr_now, 8),
                    "elapsed_s": round(elapsed, 1),
                }
                training_logs.append(log_entry)
                
                # Checkpoint saving every 25 steps (Requirement #3)
                if global_step in [25, 50] or global_step % 25 == 0:
                    ckpt_dir = ADAPTER_DIR / f"checkpoint-step-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))
                    print(f"      [Checkpoint saved]: step {global_step} -> {ckpt_dir}", flush=True)
                
                # Print progress
                if global_step % 5 == 0 or global_step <= 5:
                    pct = (batch_idx + 1) / len(train_loader) * 100
                    eta = elapsed / max(global_step, 1) * (total_steps - global_step)
                    print(
                        f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                        f"Step {global_step:4d}/{total_steps} | "
                        f"Loss: {outputs.loss.item():.4f} | "
                        f"Avg: {avg_loss:.4f} | "
                        f"LR: {lr_now:.2e} | "
                        f"ETA: {eta/60:.1f}m",
                        flush=True
                    )
        
        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_out = model(
                    input_ids=val_batch["input_ids"],
                    attention_mask=val_batch["attention_mask"],
                    labels=val_batch["labels"],
                )
                val_loss += val_out.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = epoch_loss / len(train_loader)
        
        print(f"\n  === Epoch {epoch+1} complete ===")
        print(f"      Train loss     : {avg_train_loss:.4f}")
        print(f"      Validation loss: {avg_val_loss:.4f}")
        print(f"      Time elapsed   : {(time.time()-start_time)/60:.1f} min")
        print()
        
        # Add validation loss to last log entry
        if training_logs:
            training_logs[-1]["val_loss"] = round(avg_val_loss, 6)
        
        model.train()
    
    training_duration = time.time() - start_time
    final_loss = training_logs[-1]["loss"] if training_logs else 0.0
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Total time     : {training_duration/60:.1f} minutes")
    print(f"  Final loss     : {final_loss:.4f}")
    print(f"  Total steps    : {global_step}")
    print(f"{'='*60}")
    
    # ── Verify weights changed ───────────────────────────────────────────────
    print("\n[6/7] Verifying adapter weight changes...")
    model.eval()
    changed = 0
    unchanged = 0
    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_weights:
            diff = (param.data - initial_weights[name]).abs().max().item()
            if diff > 1e-7:
                changed += 1
            else:
                unchanged += 1
    print(f"      Changed adapter matrices: {changed}")
    print(f"      Unchanged matrices: {unchanged}")
    if changed > 0:
        print("      ✓ Adapter weights genuinely updated during training.")
    else:
        print("      ⚠ No weight changes detected — check training configuration.")
    
    # ── Save adapter ─────────────────────────────────────────────────────────
    print(f"\n[7/7] Saving LoRA adapter...")
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"      Adapter saved: {ADAPTER_DIR}")
    
    # Save training logs
    logs_df = pd.DataFrame(training_logs)
    logs_df.to_csv(TRAINING_LOGS_CSV, index=False)
    print(f"      Training logs: {TRAINING_LOGS_CSV}")
    
    # Save training configuration
    config_data = {
        "model_name": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "max_seq_len": MAX_SEQ_LEN,
        "train_subset": len(train_dataset),
        "total_steps": int(global_step),
        "warmup_steps": int(warmup_steps),
        "optimizer": "AdamW",
        "lr_scheduler": "cosine_with_warmup",
        "weight_decay": WEIGHT_DECAY,
        "device": "cpu",
        "training_duration_minutes": round(training_duration / 60, 2),
        "final_train_loss": float(final_loss),
        "seed": SEED,
        **param_info,
        "adapter_changed_matrices": int(changed),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(TRAINING_CONFIG_JSON, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"      Training config: {TRAINING_CONFIG_JSON}")
    
    # ── Training loss chart ───────────────────────────────────────────────────
    if training_logs:
        fig, ax = plt.subplots(figsize=(10, 5))
        steps = [entry["step"] for entry in training_logs]
        losses = [entry["loss"] for entry in training_logs]
        avg_losses = [entry["avg_loss"] for entry in training_logs]
        
        ax.plot(steps, losses, alpha=0.4, color="#3498DB", linewidth=0.8, label="Step Loss")
        ax.plot(steps, avg_losses, color="#1A3C6E", linewidth=2.0, label="Average Loss")
        
        # Plot validation loss points
        val_entries = [(e["step"], e.get("val_loss")) for e in training_logs if "val_loss" in e]
        if val_entries:
            val_steps, val_losses = zip(*val_entries)
            ax.scatter(val_steps, val_losses, color="#E74C3C", s=80, zorder=5, label="Validation Loss")
        
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("FinLLM Training Loss", fontsize=14, fontweight="bold")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(CHART_TRAINING_LOSS, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"      Training loss chart: {CHART_TRAINING_LOSS}")
    
    return config_data, logs_df


if __name__ == "__main__":
    config_data, logs_df = train()
    print("\nFine-tuning complete!")
    print(f"Adapter saved to: {ADAPTER_DIR}")
