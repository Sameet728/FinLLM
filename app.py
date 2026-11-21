"""
FinLLM — Streamlit Web Application
=====================================
Professional financial analytics dashboard for FinLLM sentiment analysis.

Usage:
    streamlit run app.py

Student: Sameet Pisal | PRN: 202401120018 | Roll: 41
Institute: MIT Academy of Engineering
"""

import json
import time
import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from config import (
    FINETUNED_METRICS_JSON, BASELINE_METRICS_JSON,
    TRAINING_CONFIG_JSON, SYSTEM_INFO_JSON,
    CHART_CONFUSION_MATRIX, CHART_TRAINING_LOSS,
    CHART_BASELINE_COMPARE, CHART_CLASSWISE, CHART_TRAINABLE_PARAMS,
    DIAG_SYSTEM_ARCH, DIAG_LORA, DIAG_TRANSFORMER,
    DIAG_TRAINING, DIAG_INFERENCE,
    TRAINING_LOGS_CSV, LABEL_NAMES, MODEL_NAME, ADAPTER_DIR,
    RESULTS_DIR,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinLLM — Financial Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F8F9FA; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A3C6E 0%, #2980B9 100%);
        color: white;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: white !important; }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #1A3C6E;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1A3C6E;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sentiment badges */
    .badge-positive {
        background: #D5F5E3; color: #1E8449;
        padding: 6px 16px; border-radius: 20px;
        font-weight: bold; font-size: 1.1rem;
        display: inline-block;
    }
    .badge-negative {
        background: #FADBD8; color: #C0392B;
        padding: 6px 16px; border-radius: 20px;
        font-weight: bold; font-size: 1.1rem;
        display: inline-block;
    }
    .badge-neutral {
        background: #FEF9E7; color: #B7950B;
        padding: 6px 16px; border-radius: 20px;
        font-weight: bold; font-size: 1.1rem;
        display: inline-block;
    }
    
    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1A3C6E 0%, #2980B9 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .header-banner h1 { color: white; margin: 0; font-size: 2.2rem; }
    .header-banner p { color: #AED6F1; margin: 4px 0 0 0; }
    
    /* Section headers */
    .section-header {
        color: #1A3C6E;
        border-bottom: 2px solid #1A3C6E;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    
    /* Cards */
    .info-card {
        background: white;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    
    /* Result box */
    .result-box {
        background: white;
        border-radius: 10px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 16px;
    }
    
    button[kind="primary"] {
        background-color: #1A3C6E !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_results():
    """Load all saved experimental results."""
    results = {}
    
    for name, path in [
        ("finetuned", FINETUNED_METRICS_JSON),
        ("baseline", BASELINE_METRICS_JSON),
        ("training_config", TRAINING_CONFIG_JSON),
        ("system_info", SYSTEM_INFO_JSON),
    ]:
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
        else:
            results[name] = {}
    
    if TRAINING_LOGS_CSV.exists():
        results["training_logs"] = pd.read_csv(TRAINING_LOGS_CSV)
    else:
        results["training_logs"] = None
    
    return results


@st.cache_resource
def load_model():
    """Load the fine-tuned model (cached across sessions)."""
    try:
        from inference import load_finetuned_model
        with st.spinner("Loading FinLLM model..."):
            model, tokenizer = load_finetuned_model()
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)


def run_inference(text, model, tokenizer):
    """Run inference and return result dict."""
    from inference import predict
    return predict(text, model, tokenizer, device="cpu", return_confidence=True)


def run_baseline_inference(text, tokenizer):
    """Run baseline (pretrained) inference."""
    from inference import load_base_model, predict
    base_model, base_tok = load_base_model()
    return predict(text, base_model, base_tok, device="cpu", return_confidence=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("## 📈 FinLLM")
    st.sidebar.markdown("*Financial Sentiment Analysis*")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Overview Dashboard",
            "🔍 Financial News Analyzer",
            "📋 Batch Analysis",
            "⚖️ Model Comparison",
            "📊 Performance Dashboard",
            "🏗️ Architecture & Info",
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Project Details**")
    st.sidebar.markdown("Student: Sameet Pisal")
    st.sidebar.markdown("PRN: 202401120018")
    st.sidebar.markdown("Roll No: 41")
    st.sidebar.markdown("Dept: Department of Data Science")
    st.sidebar.markdown("MIT Academy of Engineering")
    
    return page


# ─── A. Overview Dashboard ────────────────────────────────────────────────────

def page_overview(results):
    st.markdown("""
    <div class="header-banner">
        <h1>📈 FinLLM</h1>
        <p>Financial Intelligence & Sentiment Analysis — Powered by Qwen2.5-0.5B-Instruct + LoRA Fine-Tuning</p>
        <p style="margin-top: 10px; font-size: 0.95rem; background: rgba(255,255,255,0.2); display: inline-block; padding: 4px 12px; border-radius: 6px;">
            🌐 <b>Live Public Web URL:</b> <a href="https://83a916b4c9f2ad.app.lhr.life" target="_blank" style="color: #ffffff; text-decoration: underline;">https://83a916b4c9f2ad.app.lhr.life</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    ft = results.get("finetuned", {})
    tc = results.get("training_config", {})
    si = results.get("system_info", {})
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        acc = ft.get("accuracy", "N/A")
        st.metric("🎯 Model Accuracy", f"{acc*100:.1f}%" if isinstance(acc, float) else acc,
                  help="Fine-tuned FinLLM accuracy on held-out test set")
    with col2:
        f1 = ft.get("macro_f1", "N/A")
        st.metric("📐 Macro F1-Score", f"{f1:.4f}" if isinstance(f1, float) else f1,
                  help="Macro-averaged F1-score across all 3 classes")
    with col3:
        tp = tc.get("trainable_parameters", "N/A")
        st.metric("⚡ Trainable Params", f"{tp:,}" if isinstance(tp, int) else tp,
                  help="LoRA trainable parameters (vs 494M total)")
    with col4:
        ds = tc.get("train_subset", "N/A")
        st.metric("📚 Training Examples", f"{ds:,}" if isinstance(ds, int) else ds,
                  help="Stratified training subset used")
    
    st.markdown("---")
    
    # Model info cards
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 🤖 Model Information")
        st.markdown(f"""
        | Property | Value |
        |---|---|
        | Base Model | Qwen2.5-0.5B-Instruct |
        | Fine-Tuning | LoRA (PEFT) |
        | LoRA Rank | {tc.get('lora_rank', 8)} |
        | LoRA Alpha | {tc.get('lora_alpha', 16)} |
        | Target Modules | {tc.get('target_modules', ['q_proj', 'v_proj'])} |
        | Total Parameters | {tc.get('total_parameters', 'N/A'):,} |
        | Trainable % | {tc.get('trainable_percentage', 'N/A')}% |
        | Adapter Status | ✅ Fine-Tuned |
        """)
    
    with col_b:
        st.markdown("### 📊 Dataset Information")
        st.markdown(f"""
        | Property | Value |
        |---|---|
        | Dataset | Financial PhraseBank |
        | Config | sentences_50agree |
        | Sentiment Classes | Positive, Negative, Neutral |
        | Train Set | {tc.get('train_subset', 'N/A')} examples |
        | Total Dataset | 4,846 sentences |
        | Split | 80% / 10% / 10% |
        | Random Seed | 42 |
        | License | Research Use |
        """)
    
    # Supported sentiments
    st.markdown("### 🏷️ Supported Sentiment Categories")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div style="background:#D5F5E3;border-radius:10px;padding:20px;text-align:center;">
            <h3 style="color:#1E8449;">✅ POSITIVE</h3>
            <p style="color:#1E8449;">Revenue growth, profit increase, earnings beat, cost savings, debt reduction</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:#FADBD8;border-radius:10px;padding:20px;text-align:center;">
            <h3 style="color:#C0392B;">❌ NEGATIVE</h3>
            <p style="color:#C0392B;">Losses, declining demand, forecast cut, layoffs, rising costs</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style="background:#FEF9E7;border-radius:10px;padding:20px;text-align:center;">
            <h3 style="color:#B7950B;">➖ NEUTRAL</h3>
            <p style="color:#B7950B;">Routine announcements, meetings, appointments, regulatory filings</p>
        </div>
        """, unsafe_allow_html=True)


# ─── B. Financial News Analyzer ───────────────────────────────────────────────

def page_analyzer(model, tokenizer, model_error):
    st.markdown("## 🔍 Financial News Analyzer")
    st.markdown("Enter financial news text to classify its sentiment using FinLLM.")
    
    # Example inputs
    examples = {
        "Positive — Revenue growth": "The company reported a 30% increase in quarterly revenue, exceeding analyst expectations.",
        "Negative — Losses": "The company announced substantial losses following a sharp decline in consumer demand.",
        "Neutral — Routine announcement": "The company announced that its annual shareholder meeting will take place next month.",
        "Positive — Cost savings": "Operating profit improved significantly due to successful cost reduction measures.",
        "Negative — Forecast cut": "The company reduced its annual earnings forecast because of increasing raw material expenses.",
        "Neutral — Board appointment": "The board of directors appointed a new independent director on Monday.",
        "Ambiguous — Mixed signals": "Although quarterly revenue increased, the company's net profit declined sharply.",
    }
    
    selected_example = st.selectbox("📌 Load example input:", ["(Type your own)"] + list(examples.keys()))
    
    if selected_example != "(Type your own)":
        default_text = examples[selected_example]
    else:
        default_text = ""
    
    user_text = st.text_area(
        "Financial news text:",
        value=default_text,
        height=120,
        placeholder="Enter financial news, earnings announcement, or market commentary here...",
        key="analyzer_input"
    )
    
    char_count = len(user_text)
    st.caption(f"Characters: {char_count} | Words: {len(user_text.split()) if user_text.strip() else 0}")
    
    col_analyze, col_clear = st.columns([1, 5])
    with col_analyze:
        analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=False):
            st.rerun()
    
    if analyze_btn:
        if not user_text.strip():
            st.warning("⚠️ Please enter financial news text before analyzing.")
            return
        
        if model is None:
            st.error(f"Model not loaded: {model_error}")
            return
        
        with st.spinner("🤔 FinLLM is analyzing..."):
            result = run_inference(user_text, model, tokenizer)
        
        label = result.get("label")
        is_valid = result.get("is_valid", False)
        confidence = result.get("confidence", {})
        raw_output = result.get("raw_output", "")
        infer_time = result.get("inference_time_s", 0)
        
        # Result display
        st.markdown("---")
        st.markdown("### 📊 Analysis Result")
        
        if not is_valid or label is None:
            st.error(f"⚠️ Model returned an unexpected response: '{raw_output}'. This may indicate the model needs more training.")
            return
        
        # Color and icon
        color_map = {"positive": "#D5F5E3", "negative": "#FADBD8", "neutral": "#FEF9E7"}
        icon_map = {"positive": "✅", "negative": "❌", "neutral": "➖"}
        text_map = {"positive": "#1E8449", "negative": "#C0392B", "neutral": "#B7950B"}
        desc_map = {
            "positive": "Favorable financial sentiment — indicates positive impact on company value.",
            "negative": "Unfavorable financial sentiment — indicates negative impact on company value.",
            "neutral": "Neutral financial sentiment — routine information without directional bias."
        }
        
        col_result, col_details = st.columns([1, 1])
        
        with col_result:
            st.markdown(f"""
            <div style="background:{color_map[label]};border-radius:12px;padding:30px;text-align:center;">
                <div style="font-size:3rem;">{icon_map[label]}</div>
                <div style="font-size:2rem;font-weight:bold;color:{text_map[label]};">{label.upper()}</div>
                <div style="color:{text_map[label]};margin-top:8px;">{desc_map[label]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_details:
            st.markdown("**Analysis Details**")
            st.markdown(f"- **Model:** FinLLM (Qwen2.5-0.5B + LoRA)")
            st.markdown(f"- **Raw model output:** `{raw_output}`")
            st.markdown(f"- **Inference time:** {infer_time:.3f}s")
            
            if confidence:
                st.markdown("**Confidence Scores** *(softmax over label token logits)*")
                for lbl in ["positive", "negative", "neutral"]:
                    score = confidence.get(lbl, 0)
                    bar_color = text_map.get(lbl, "#333")
                    st.markdown(f"`{lbl.capitalize():<10}` {score:.4f}")
                    st.progress(float(score), text=None)
                st.caption("*Confidence = normalized log-probability of each label token at generation position*")
        
        # Disclaimer for explanations
        st.markdown("""
        > **Note:** FinLLM was trained only on sentiment labels (positive/negative/neutral).
        > The model was not trained to generate explanations. The classification above is 
        > the genuine model output.
        """)


# ─── C. Batch Analysis ────────────────────────────────────────────────────────

def page_batch(model, tokenizer, model_error):
    st.markdown("## 📋 Batch Financial News Analysis")
    st.markdown("Upload a CSV file with a `sentence` or `text` column to analyze multiple news items.")
    
    BATCH_LIMIT = 25
    
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    
    # Create sample CSV for download
    sample_df = pd.DataFrame({
        "sentence": [
            "The company reported a 30% increase in quarterly revenue.",
            "Net losses widened significantly due to declining market demand.",
            "The board announced the appointment of a new CFO.",
            "Operating margins improved due to cost optimization initiatives.",
            "The company warned of potential supply chain disruptions.",
        ]
    })
    csv_bytes = sample_df.to_csv(index=False).encode()
    
    col_dl, col_sample = st.columns([1, 1])
    with col_dl:
        st.download_button("📥 Download Sample CSV", csv_bytes, "sample_financial_news.csv", "text/csv")
    with col_sample:
        load_sample = st.button("⚡ Load Sample Batch (5 Headlines)")
    
    batch_df = None
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Cannot read CSV: {e}")
            return
    elif load_sample or st.session_state.get("batch_sample_loaded", False):
        st.session_state["batch_sample_loaded"] = True
        batch_df = sample_df
        
    if batch_df is not None:
        
        # Find sentence column
        text_col = None
        for col in ["sentence", "text", "news", "headline", "content"]:
            if col in batch_df.columns:
                text_col = col
                break
        if text_col is None:
            text_col = batch_df.columns[0]
        
        st.info(f"Loaded {len(batch_df)} rows. Using column: `{text_col}`")
        
        if len(batch_df) > BATCH_LIMIT:
            st.warning(f"⚠️ Batch limit is {BATCH_LIMIT}. Processing first {BATCH_LIMIT} rows.")
            batch_df = batch_df.head(BATCH_LIMIT)
        
        if model is None:
            st.error(f"Model not loaded: {model_error}")
            return
        
        if st.button("▶️ Run Batch Analysis", type="primary"):
            from inference import predict
            
            results_rows = []
            progress = st.progress(0)
            status = st.empty()
            
            for i, row in batch_df.iterrows():
                text = str(row[text_col])
                status.text(f"Analyzing [{i+1}/{len(batch_df)}]: {text[:50]}...")
                
                result = predict(text, model, tokenizer, device="cpu", return_confidence=True)
                
                label = result.get("label") or "invalid"
                conf = result.get("confidence", {})
                top_conf = conf.get(label, 0) if conf else 0
                
                results_rows.append({
                    "sentence": text,
                    "predicted_sentiment": label,
                    "confidence": round(top_conf, 4),
                    "inference_time_s": result.get("inference_time_s", 0),
                    "raw_output": result.get("raw_output", ""),
                })
                progress.progress((i + 1) / len(batch_df))
            
            status.empty()
            results_batch = pd.DataFrame(results_rows)
            
            st.success(f"✅ Analyzed {len(results_batch)} items")
            
            # Summary
            col1, col2, col3 = st.columns(3)
            counts = results_batch["predicted_sentiment"].value_counts()
            with col1:
                st.metric("✅ Positive", counts.get("positive", 0))
            with col2:
                st.metric("❌ Negative", counts.get("negative", 0))
            with col3:
                st.metric("➖ Neutral", counts.get("neutral", 0))
            
            # Pie chart
            pie_data = counts.reset_index()
            pie_data.columns = ["Sentiment", "Count"]
            color_map = {"positive": "#27AE60", "negative": "#E74C3C", "neutral": "#F39C12", "invalid": "#95A5A6"}
            fig = px.pie(pie_data, names="Sentiment", values="Count",
                         color="Sentiment", color_discrete_map=color_map,
                         title="Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.dataframe(results_batch, use_container_width=True)
            
            # Download
            csv_out = results_batch.to_csv(index=False).encode()
            st.download_button("📥 Download Results CSV", csv_out, "finllm_batch_results.csv", "text/csv")


# ─── D. Model Comparison ──────────────────────────────────────────────────────

def page_comparison(results, model, tokenizer, model_error):
    st.markdown("## ⚖️ Pretrained vs Fine-Tuned Model Comparison")
    
    text = st.text_area(
        "Financial news text to compare:",
        value="The company reported a 30% increase in quarterly revenue, exceeding analyst expectations.",
        height=100
    )
    
    if st.button("▶️ Compare Both Models", type="primary"):
        if not text.strip():
            st.warning("Please enter text.")
            return
        
        col_base, col_ft = st.columns(2)
        
        with col_base:
            st.markdown("### 🤖 Pretrained Model (Baseline)")
            if model is None:
                st.error(f"Model not loaded: {model_error}")
            else:
                with st.spinner("Running pretrained model..."):
                    from inference import load_base_model, predict
                    base_model, base_tok = load_base_model()
                    base_result = predict(text, base_model, base_tok, device="cpu", return_confidence=True)
                
                label = base_result.get("label") or "invalid"
                lmap = {"positive": "✅", "negative": "❌", "neutral": "➖", "invalid": "⚠️"}
                cmap = {"positive": "#D5F5E3", "negative": "#FADBD8", "neutral": "#FEF9E7", "invalid": "#F2F3F4"}
                
                st.markdown(f"""
                <div style="background:{cmap.get(label,'#F2F3F4')};border-radius:10px;padding:20px;text-align:center;">
                    <div style="font-size:2rem;">{lmap.get(label,'?')}</div>
                    <b style="font-size:1.3rem;">{label.upper()}</b>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"- Raw output: `{base_result.get('raw_output','')}`")
                st.markdown(f"- Inference time: {base_result.get('inference_time_s',0):.3f}s")
        
        with col_ft:
            st.markdown("### ✨ FinLLM (Fine-Tuned)")
            if model is None:
                st.error("Fine-tuned model not loaded.")
            else:
                with st.spinner("Running FinLLM..."):
                    ft_result = run_inference(text, model, tokenizer)
                
                label = ft_result.get("label") or "invalid"
                st.markdown(f"""
                <div style="background:{cmap.get(label,'#F2F3F4')};border-radius:10px;padding:20px;text-align:center;">
                    <div style="font-size:2rem;">{lmap.get(label,'?')}</div>
                    <b style="font-size:1.3rem;">{label.upper()}</b>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"- Raw output: `{ft_result.get('raw_output','')}`")
                st.markdown(f"- Inference time: {ft_result.get('inference_time_s',0):.3f}s")
    
    # Metrics comparison table
    st.markdown("---")
    st.markdown("### 📊 Evaluation Metrics Comparison")
    
    bm = results.get("baseline", {})
    fm = results.get("finetuned", {})
    
    if bm and fm:
        rows = []
        for metric, label in [
            ("accuracy", "Accuracy"),
            ("macro_precision", "Macro Precision"),
            ("macro_recall", "Macro Recall"),
            ("macro_f1", "Macro F1-Score"),
            ("weighted_f1", "Weighted F1-Score"),
            ("invalid_predictions", "Invalid Predictions"),
            ("avg_inference_time_s", "Avg Inference Time (s)"),
        ]:
            b_val = bm.get(metric, "N/A")
            f_val = fm.get(metric, "N/A")
            if isinstance(b_val, float) and isinstance(f_val, float):
                diff = f_val - b_val
                diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            else:
                diff_str = "N/A"
            rows.append({"Metric": label, "Pretrained": str(b_val), "FinLLM": str(f_val), "Difference": diff_str})
        
        cmp_df = pd.DataFrame(rows)
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)


# ─── E. Performance Dashboard ─────────────────────────────────────────────────

def page_performance(results):
    st.markdown("## 📊 Model Performance Dashboard")
    
    # Charts
    for chart_path, title in [
        (CHART_BASELINE_COMPARE, "Pretrained vs Fine-Tuned Comparison"),
        (CHART_CONFUSION_MATRIX, "Confusion Matrix"),
        (CHART_CLASSWISE, "Per-Class Performance"),
        (CHART_TRAINING_LOSS, "Training Loss Curve"),
        (CHART_TRAINABLE_PARAMS, "Parameter Distribution"),
    ]:
        if chart_path.exists():
            st.markdown(f"### {title}")
            st.image(str(chart_path), use_container_width=True)
            st.markdown("---")
        else:
            st.info(f"Chart not yet generated: {chart_path.name}. Run evaluate.py first.")
    
    # Training logs
    logs = results.get("training_logs")
    if logs is not None and len(logs) > 0:
        st.markdown("### 📈 Training Logs (Interactive)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=logs["step"], y=logs["loss"],
                                  mode="lines", name="Step Loss",
                                  line=dict(color="#3498DB", width=1), opacity=0.5))
        fig.add_trace(go.Scatter(x=logs["step"], y=logs["avg_loss"],
                                  mode="lines", name="Avg Loss",
                                  line=dict(color="#1A3C6E", width=2)))
        fig.update_layout(title="Training Loss", xaxis_title="Step", yaxis_title="Loss",
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)


# ─── F. Architecture & Info ───────────────────────────────────────────────────

def page_architecture(results):
    st.markdown("## 🏗️ Model Architecture & Information")
    
    tc = results.get("training_config", {})
    si = results.get("system_info", {})
    
    # Architecture details
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚙️ LoRA Configuration")
        st.json({
            "lora_rank": tc.get("lora_rank", 8),
            "lora_alpha": tc.get("lora_alpha", 16),
            "lora_dropout": tc.get("lora_dropout", 0.05),
            "target_modules": tc.get("target_modules", ["q_proj", "v_proj"]),
            "total_parameters": tc.get("total_parameters"),
            "trainable_parameters": tc.get("trainable_parameters"),
            "trainable_pct": f"{tc.get('trainable_percentage', 0)}%",
        })
    with col2:
        st.markdown("### 💻 System Hardware")
        st.json({
            "cpu": si.get("cpu", "Intel Core Ultra 5 125H"),
            "cpu_cores": si.get("cpu_cores_logical"),
            "ram_gb": si.get("ram_gb"),
            "gpu": si.get("gpu_name", "Intel Arc Graphics"),
            "cuda": si.get("cuda_available", False),
            "python": si.get("python_version", "3.13"),
            "torch": si.get("torch_version", "2.14.1+cpu"),
        })
    
    # Diagrams
    st.markdown("---")
    for diag_path, title in [
        (DIAG_SYSTEM_ARCH, "System Architecture"),
        (DIAG_LORA, "LoRA Architecture"),
        (DIAG_TRANSFORMER, "Transformer Architecture"),
        (DIAG_TRAINING, "Training Pipeline"),
        (DIAG_INFERENCE, "Inference Pipeline"),
    ]:
        if diag_path.exists():
            st.markdown(f"### {title}")
            st.image(str(diag_path), use_container_width=True)
        else:
            st.info(f"Diagram not found: {diag_path.name}. Run generate_diagrams.py.")


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    results = load_results()
    page = render_sidebar()
    
    # Load model once
    model, tokenizer, model_error = load_model()
    
    if "Overview" in page:
        page_overview(results)
    elif "Analyzer" in page:
        page_analyzer(model, tokenizer, model_error)
    elif "Batch" in page:
        page_batch(model, tokenizer, model_error)
    elif "Comparison" in page:
        page_comparison(results, model, tokenizer, model_error)
    elif "Performance" in page:
        page_performance(results)
    elif "Architecture" in page:
        page_architecture(results)


if __name__ == "__main__":
    main()
