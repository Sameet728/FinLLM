"""
FinLLM CPU Training Benchmark
===============================
Runs a short 10-step benchmark on CPU with Qwen2.5-0.5B-Instruct + LoRA.
Measures forward pass time, backward pass time, step latency, memory usage,
and estimates total training time for full runs.

Student: Sameet Pisal | PRN: 202401120018 | Roll: 41
Institute: MIT Academy of Engineering
"""

import json
import time
import sys
import psutil
import torch
from pathlib import Path
from torch.optim import AdamW

from config import (
    MODEL_NAME, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
    RESULTS_DIR, ensure_dirs
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


def run_benchmark(num_benchmark_steps=10):
    ensure_dirs()
    print("=" * 65)
    print("FinLLM — CPU Training Benchmark & Parameter Count Verification")
    print("=" * 65)
    
    print("\n[1/5] Loading tokenizer and model in float32 for CPU...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    load_time = time.time() - t0
    print(f"      Model loaded in {load_time:.2f}s")
    
    # Base parameter count
    base_total = sum(p.numel() for p in model.parameters())
    print(f"      Base model total parameters: {base_total:,}")
    
    print("\n[2/5] Applying LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        inference_mode=False
    )
    model = get_peft_model(model, lora_config)
    
    print("\n[3/5] Programmatically counting parameters (Requirement #4)...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params) * 100
    
    print(f"      Total parameters:     {total_params:>12,}")
    print(f"      Trainable parameters: {trainable_params:>12,}")
    print(f"      Frozen parameters:    {frozen_params:>12,}")
    print(f"      Trainable ratio:      {trainable_pct:>11.4f}%")
    
    print("\n[4/5] Running short training benchmark (10 forward + backward steps)...")
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    
    # Synthetic batch of length 64 (representative of average financial news sentence)
    dummy_input = torch.randint(100, 1000, (1, 64), dtype=torch.long)
    dummy_labels = dummy_input.clone()
    dummy_labels[:, :-5] = -100  # loss on last 5 tokens only
    dummy_mask = torch.ones_like(dummy_input)
    
    step_times = []
    losses = []
    
    for step in range(num_benchmark_steps):
        s_start = time.time()
        optimizer.zero_grad()
        out = model(input_ids=dummy_input, attention_mask=dummy_mask, labels=dummy_labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        s_time = time.time() - s_start
        step_times.append(s_time)
        losses.append(loss.item())
        print(f"      Benchmark Step {step+1:2d}/{num_benchmark_steps}: Latency = {s_time:.3f}s | Loss = {loss.item():.4f}")
    
    avg_step_time = float(np.mean(step_times))
    std_step_time = float(np.std(step_times))
    
    # Training projections
    steps_300 = 300  # 300 examples, batch_size=1
    est_total_seconds = avg_step_time * steps_300
    est_total_minutes = est_total_seconds / 60
    
    print("\n[5/5] Benchmark Summary & Training Feasibility Analysis:")
    print(f"      Average step latency:   {avg_step_time:.3f}s ± {std_step_time:.3f}s")
    print(f"      Throughput:             {1.0/avg_step_time:.2f} samples/sec on CPU")
    print(f"      Estimated time for 300 samples (1 epoch): {est_total_minutes:.2f} minutes ({est_total_seconds:.0f} seconds)")
    
    benchmark_data = {
        "device": "CPU (Intel Core Ultra 5 125H)",
        "num_benchmark_steps": num_benchmark_steps,
        "avg_step_time_s": round(avg_step_time, 4),
        "std_step_time_s": round(std_step_time, 4),
        "throughput_samples_per_sec": round(1.0 / avg_step_time, 3),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "frozen_parameters": int(frozen_params),
        "trainable_percentage": round(trainable_pct, 4),
        "estimated_300_sample_minutes": round(est_total_minutes, 2),
        "losses_sample": [round(l, 4) for l in losses],
        "cpu_usage_pct": psutil.cpu_percent(),
        "ram_usage_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    bench_file = RESULTS_DIR / "cpu_benchmark.json"
    with open(bench_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"      Saved benchmark results: {bench_file}")
    print("=" * 65)
    return benchmark_data


if __name__ == "__main__":
    import numpy as np
    run_benchmark(num_benchmark_steps=10)
