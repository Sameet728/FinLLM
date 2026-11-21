"""
FinLLM Inference Utility
=========================
Shared inference functions used by baseline.py, evaluate.py, and app.py.
Handles model loading, prompt formatting, and label parsing.

Student: Sameet Pisal | PRN: 202401120018 | Roll: 41
"""

import re
import time
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict

from config import (
    MODEL_NAME, ADAPTER_DIR, SYSTEM_PROMPT, LABEL_NAMES,
    MAX_NEW_TOKENS, DO_SAMPLE, TEMPERATURE, REPETITION_PENALTY
)


# ─── Prompt Formatting ────────────────────────────────────────────────────────

def build_prompt(sentence: str, tokenizer) -> str:
    """
    Build a chat-formatted prompt using the model's chat template.
    The assistant's response is left open (no assistant turn) for generation.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": sentence},
    ]
    # apply_chat_template handles Qwen's specific format
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # adds <|im_start|>assistant\n
    )
    return prompt


def build_training_text(sentence: str, label: str, tokenizer) -> str:
    """
    Build the full instruction-response text for training (with assistant answer).
    Used by train.py for supervised fine-tuning.
    """
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": sentence},
        {"role": "assistant", "content": label.lower()},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return text


# ─── Output Parsing ───────────────────────────────────────────────────────────

def parse_label(output_text: str) -> Tuple[Optional[str], bool]:
    """
    Extract the sentiment label from model output.
    Returns (label_or_None, is_valid).
    """
    text = output_text.strip().lower()
    
    # Direct match
    for label in LABEL_NAMES:
        if label in text:
            return label, True
    
    # Fuzzy match for common variations
    alias_map = {
        "pos": "positive", "neg": "negative", "neut": "neutral",
        "1": "positive", "-1": "negative", "0": "neutral",
    }
    for alias, label in alias_map.items():
        if alias in text:
            return label, True
    
    return None, False


# ─── Confidence Scoring ───────────────────────────────────────────────────────

def compute_confidence(model, tokenizer, input_ids: torch.Tensor, device: str) -> Dict[str, float]:
    """
    Compute confidence scores by comparing log-probs of the three label tokens.
    
    Method: softmax over first-next-token log-probabilities for
    'positive', 'negative', 'neutral' — a documented, deterministic approach.
    Returns a dict {label: probability}.
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]  # shape: [1, vocab_size]
    
    # Get token IDs for each label (first token of each label word)
    label_token_ids = []
    label_names_ordered = []
    for label in LABEL_NAMES:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if ids:
            label_token_ids.append(ids[0])
            label_names_ordered.append(label)
    
    if not label_token_ids:
        return {label: 1/3 for label in LABEL_NAMES}
    
    # Extract logits for the three label tokens and apply softmax
    label_logits = logits[0, label_token_ids]  # shape: [3]
    probs = torch.softmax(label_logits, dim=-1).tolist()
    
    return {name: round(prob, 4) for name, prob in zip(label_names_ordered, probs)}


# ─── Single Inference ─────────────────────────────────────────────────────────

def predict(
    sentence: str,
    model,
    tokenizer,
    device: str = "cpu",
    return_confidence: bool = True
) -> Dict:
    """
    Run a single inference on a financial sentence.
    
    Returns:
        dict with keys: label, is_valid, raw_output, inference_time_s, confidence
    """
    prompt = build_prompt(sentence, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    
    t0 = time.time()
    with torch.no_grad():
        # Compute confidence BEFORE generating (uses the same forward pass)
        confidence = None
        if return_confidence:
            try:
                confidence = compute_confidence(model, tokenizer, inputs["input_ids"], device)
            except Exception:
                confidence = None
        
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
    t1 = time.time()
    
    # Decode only the new tokens (not the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    label, is_valid = parse_label(raw_output)
    
    return {
        "label": label if is_valid else None,
        "is_valid": is_valid,
        "raw_output": raw_output,
        "inference_time_s": round(t1 - t0, 3),
        "confidence": confidence,
    }


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_base_model(model_name: str = MODEL_NAME):
    """Load pretrained Qwen model and tokenizer (no adapter)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(adapter_path: Path = ADAPTER_DIR):
    """Load pretrained Qwen model with LoRA adapter."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    print("Fine-tuned model loaded successfully.")
    return model, tokenizer
