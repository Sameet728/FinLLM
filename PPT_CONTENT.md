# FinLLM — PowerPoint Presentation Structure (15 Slides)

**Project:** FinLLM — Financial News Sentiment Analyzer  
**Student:** Sameet Pisal (PRN: 202401120018, Roll: 41)  
**Department:** Department of Data Science, MIT Academy of Engineering  
**Live URL:** https://sameetfinllm.streamlit.app/  
**GitHub:** https://github.com/Sameet728/FinLLM  

---

### Slide 1: Title & Project Overview
- **Title:** FinLLM: Development and Fine-Tuning of an Open-Source LLM for Financial News Sentiment Analysis
- **Subtitle:** Assignment 2: Individual LLM Development & Technical Presentation
- **Author:** Sameet Pisal | PRN: 202401120018 | Roll: 41
- **Affiliation:** Department of Data Science, MIT Academy of Engineering, Pune
- **Recommended Image:** `Diagrams/system_architecture.png`

### Slide 2: Problem Statement & Motivation
- **Core Challenge:** General-purpose LLMs and dictionary classifiers fail on nuanced corporate financial syntax.
- **Key Issues:** Extreme positive bias, misclassification of neutral governance disclosures, high GPU fine-tuning costs.
- **Objective:** Build an efficient, open-source financial LLM adapted via parameter-efficient fine-tuning on consumer CPU hardware.
- **Recommended Image:** `Diagrams/transformer_architecture.png`

### Slide 3: Project Objectives & Technical Goals
- Domain adaptation of Qwen2.5-0.5B-Instruct using Low-Rank Adaptation (LoRA).
- 3-Class Sentiment Classification: Positive, Negative, Neutral.
- Overcome pretrained baseline bias (achieve >= 75% accuracy).
- Develop a LangChain AI Agent with 4 autonomous analytical tools.
- Deploy a responsive, light-themed financial analytics dashboard on Streamlit Cloud.
- **Recommended Image:** `Screenshots/ui_01_overview_top.png`

### Slide 4: Proposed System Architecture
- **Ingestion:** Financial PhraseBank benchmark (4,837 unique sentences).
- **Core LLM:** Qwen2.5-0.5B-Instruct decoder with 24 Transformer blocks.
- **Fine-Tuning:** LoRA rank decomposition on `q_proj` and `v_proj` attention matrices.
- **Deployment:** Streamlit Cloud web dashboard + LangChain agent interface.
- **Recommended Image:** `Diagrams/system_architecture.png`

### Slide 5: Financial Dataset & Stratified Preprocessing
- **Source:** Takala Financial PhraseBank (expert-annotated financial sentences).
- **Deduplication:** 4,837 unique clean records.
- **Stratified Split:** 300 Train (80%), 80 Validation (10%), 180 Held-Out Test (10%).
- **Distribution:** Preserved natural financial market ratios (59.3% Neutral, 28.4% Positive, 12.3% Negative).
- **Recommended Image:** `Diagrams/dataset_distribution.png`

### Slide 6: Pretrained Model & Tokenizer Details
- **Foundation LLM:** Qwen2.5-0.5B-Instruct (494.57M parameters).
- **Structure:** 24 layers, 14 attention heads, 896 hidden dimension.
- **Tokenizer:** Byte-Pair Encoding (BPE), vocabulary of 151,936 tokens.
- **Prompt Engineering:** Strict ChatML formatting with instruction-answer loss masking.
- **Recommended Image:** `Diagrams/transformer_architecture.png`

### Slide 7: LoRA Fine-Tuning Methodology
- **Formulation:** $W = W_0 + rac{lpha}{r} B \cdot A$ ($W_0$ frozen, $A \sim \mathcal{N}$, $B = 0$).
- **Configuration:** Rank $r=8$, Alpha $lpha=16$, Dropout 0.05.
- **Target Layers:** `q_proj` and `v_proj` across all 24 layers (96 matrices).
- **Parameter Efficiency:** 540,672 trainable parameters (0.1093% of model).
- **Recommended Image:** `Diagrams/lora_architecture.png`

### Slide 8: Training Execution & Empirical Convergence
- **Optimization:** AdamW optimizer, learning rate $2 	imes 10^{-4}$, cosine decay schedule.
- **Hardware:** Intel Core Ultra 5 125H CPU (14 cores, 16 GB DDR5 RAM).
- **Duration:** 14.86 minutes across 75 optimization steps.
- **Convergence:** Training loss dropped from 2.6845 down to 0.0114; validation loss: 0.1528.
- **Recommended Image:** `Diagrams/training_loss.png`

### Slide 9: Quantitative Model Comparison (Baseline vs FinLLM)
- **Pretrained Baseline:** 43.33% Accuracy, 0.4793 Macro F1, 0.3250 Weighted F1.
- **Fine-Tuned FinLLM:** 80.00% Accuracy, 0.7753 Macro F1, 0.7952 Weighted F1.
- **Key Breakthrough:** Neutral recall increased from 7.00% to 90.00% (+83.00%), completely resolving zero-shot positive bias.
- **Recommended Image:** `Diagrams/baseline_comparison.png`

### Slide 10: Performance Evaluation & Confusion Matrix
- **Held-Out Test Set:** 180 unseen expert-annotated sentences.
- **Confusion Matrix:** Strong diagonal dominance (31 Positive, 16 Negative, 96 Neutral correct).
- **Per-Class Precision:** Negative: 84.2%, Positive: 60.8%, Neutral: 90.0%.
- **Zero Hallucination:** 100% valid sentiment syntax generated.
- **Recommended Image:** `Diagrams/confusion_matrix.png`

### Slide 11: LangChain AI Financial Agent (CO4)
- **Architecture:** Autonomous tool router connecting LLM to financial tools.
- **Tool Suite:**
  1. `analyze_sentiment`: Single headline classification & confidence scoring.
  2. `analyze_batch`: Tabular processing of multi-headline market feeds.
  3. `summarize_sentiment`: Net sentiment scoring and market bias synthesis.
  4. `generate_analysis_report`: Structured executive financial intelligence briefs.
- **Recommended Image:** `Diagrams/inference_pipeline.png`

### Slide 12: Streamlit Analytics Dashboard Demonstration
- **Module 1: Overview:** Real-time metrics, LoRA badges, links.
- **Module 2: News Analyzer:** Live positive, negative, and neutral sentiment classification.
- **Module 3: Batch Analysis:** CSV upload, dynamic distribution donut chart.
- **Module 4: Model Comparison:** Side-by-side inference cards.
- **Recommended Image:** `Screenshots/ui_05_analyzer_positive.png`

### Slide 13: Functional Quality Assurance & Test Matrix
- **Automated Test Suite:** 14/14 automated unit and integration tests passed (100% pass rate).
- **Verified Aspects:** Split stratification, parameter counts, latency (< 3.0s CPU), softmax calibration, export schemas.
- **Edge Case Tests:** Executed against ambiguous, mixed, and negation syntax.
- **Recommended Image:** `Screenshots/ui_09_batch_results.png`

### Slide 14: Responsible AI, Ethics & Deployment Limitations
- **Financial Compliance:** Explicit disclaimer that outputs are educational NLP classifications, not financial trading advice.
- **Bias Mitigation:** Addressed dataset class imbalance through stratified sampling.
- **Deployment Optimization:** Lightweight CPU inference and memory management for Streamlit Cloud constraints.
- **Recommended Image:** `Screenshots/ui_10_comparison_cards.png`

### Slide 15: Conclusion & Live Project Links
- **Conclusion:** LoRA fine-tuning provides viable, high-precision domain adaptation for open-source LLMs on consumer CPUs (+36.67% accuracy gain).
- **Live Cloud Web App:** https://sameetfinllm.streamlit.app/
- **GitHub Repository:** https://github.com/Sameet728/FinLLM
- **Academic Report:** Available in PDF and DOCX formats.
- **Recommended Image:** `Screenshots/ui_01_overview_top.png`
