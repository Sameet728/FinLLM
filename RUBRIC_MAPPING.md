# FinLLM — Academic Rubric Mapping Matrix (25 Marks)

**Course Outcome Mapping for Assignment 2:**  
Student: Sameet Pisal | PRN: 202401120018 | Roll: 41  
Department of Data Science | MIT Academy of Engineering, Pune  

---

| Course Outcome (CO) | Max Marks | Implemented Features | Source Code Evidence | Verification Artifacts & Screenshots |
| :--- | :--- | :--- | :--- | :--- |
| **CO1: Deep Learning Implementation** | **3 Marks** | - Pretrained Qwen2.5-0.5B-Instruct architecture implementation.<br>- 24 Transformer decoder layers with self-attention.<br>- Byte-Pair Encoding (BPE) tokenization workflow with ChatML format.<br>- System architecture and pipeline design. | `inference.py`<br>`config.py`<br>`prepare_dataset.py` | `Diagrams/system_architecture.png`<br>`Diagrams/transformer_architecture.png`<br>`Screenshots/ui_01_overview_top.png` |
| **CO2: Model Comparison & Optimization** | **4 Marks** | - Quantitative comparison between Pretrained Zero-Shot Baseline and Fine-Tuned FinLLM.<br>- Benchmark on identical 180 held-out test sentences.<br>- Evaluation of accuracy (+36.67%), F1-scores, classwise recall.<br>- Analysis of LoRA vs full fine-tuning. | `baseline.py`<br>`evaluate.py`<br>`results/model_comparison.json` | `Diagrams/baseline_comparison.png`<br>`results/classification_report.csv`<br>`Screenshots/ui_10_comparison_cards.png` |
| **CO3: Actual LLM Fine-Tuning** | **8 Marks** | - LoRA implementation ($r=8, lpha=16$) on `q_proj` and `v_proj`.<br>- 75 optimization steps on Intel CPU (14.86 min).<br>- Training loss convergence (2.6845 down to 0.0114).<br>- Export of 2.17 MB `adapter_model.safetensors`.<br>- Exactly 540,672 trainable parameters (0.1093%). | `train.py`<br>`models/finllm_adapter/adapter_model.safetensors`<br>`results/training_logs.csv` | `Diagrams/lora_architecture.png`<br>`Diagrams/training_loss.png`<br>`results/training_configuration.json` |
| **CO4: Functional LangChain AI Agent** | **4 Marks** | - Functional LangChain Agent (`agent.py`) with 4 registered tools:<br>  1. `analyze_sentiment`<br>  2. `analyze_batch`<br>  3. `summarize_sentiment`<br>  4. `generate_analysis_report`<br>- Natural language routing and tool invocation.<br>- Automated 4-test case execution suite. | `agent.py` | `agent.py` execution logs<br>`Diagrams/inference_pipeline.png`<br>`Documentation/FINAL_PROJECT_INFORMATION.md` |
| **CO5: Performance, Ethics & Deployment** | **6 Marks** | - Comprehensive metrics (80.0% accuracy, 0.7753 macro F1, 90.0% neutral recall).<br>- Confusion matrix analysis on 180 sentences.<br>- Automated 14-point functional test suite (100% pass rate).<br>- Ethical risk analysis, bias mitigation, and trading disclaimers.<br>- Live deployment on Streamlit Cloud and GitHub. | `tests/test_functional.py`<br>`app.py`<br>`streamlit_app.py`<br>`results/functional_test_results.json` | `Diagrams/confusion_matrix.png`<br>`Screenshots/ui_05_analyzer_positive.png`<br>`Screenshots/ui_06_analyzer_negative.png`<br>`Screenshots/ui_07_analyzer_neutral.png`<br>Live URL: https://sameetfinllm.streamlit.app/ |

**Total Score Target:** **25 / 25 Marks**
