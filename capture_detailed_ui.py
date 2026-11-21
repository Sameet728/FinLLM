"""
capture_detailed_ui.py
=======================
Captures clean, full-coverage, genuine UI screenshots for each section of all 6
dashboard pages of the live FinLLM application. Zero terminal/cmd screenshots.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = Path("C:/Users/samee/Desktop/LLM/FinLLM/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900}, device_scale_factor=1.25)
        page = context.new_page()
        
        base_url = "http://localhost:8501"
        print(f"Connecting to {base_url}...")
        page.goto(base_url, timeout=30000)
        page.wait_for_timeout(3000)
        
        # ═════════════════════════════════════════════════════════════════════
        # 1. OVERVIEW DASHBOARD
        # ═════════════════════════════════════════════════════════════════════
        print("\n[Page 1] Overview Dashboard...")
        page.get_by_text("Overview Dashboard").click()
        page.wait_for_timeout(2000)
        
        # 1A. Top View (Banner, Live URL, 4 Key Metric Cards)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_01_overview_top.png"))
        print("  ✓ Captured ui_01_overview_top.png")
        
        # 1B. Middle View (Model Specs & Dataset Parameters Table)
        page.evaluate("window.scrollTo(0, 320)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_02_overview_specs.png"))
        print("  ✓ Captured ui_02_overview_specs.png")
        
        # 1C. Bottom View (Supported Sentiment Categories Cards)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_03_overview_categories.png"))
        print("  ✓ Captured ui_03_overview_categories.png")
        
        # ═════════════════════════════════════════════════════════════════════
        # 2. FINANCIAL NEWS ANALYZER
        # ═════════════════════════════════════════════════════════════════════
        print("\n[Page 2] Financial News Analyzer...")
        page.get_by_text("Financial News Analyzer").click()
        page.wait_for_timeout(2000)
        
        # 2A. Input Area & Controls
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_04_analyzer_input.png"))
        print("  ✓ Captured ui_04_analyzer_input.png")
        
        # 2B. Positive Demo Execution
        print("  Running Positive Demo...")
        page.locator("textarea").first.fill("The company reported a 30% increase in quarterly revenue, exceeding analyst expectations.")
        page.wait_for_timeout(400)
        page.get_by_role("button", name="Analyze Sentiment").click()
        page.wait_for_timeout(3500)
        page.evaluate("window.scrollTo(0, 280)")
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_05_analyzer_positive.png"))
        print("  ✓ Captured ui_05_analyzer_positive.png")
        
        # 2C. Negative Demo Execution
        print("  Running Negative Demo...")
        page.locator("textarea").first.fill("The company announced substantial losses following a sharp decline in consumer demand.")
        page.wait_for_timeout(400)
        page.get_by_role("button", name="Analyze Sentiment").click()
        page.wait_for_timeout(3500)
        page.evaluate("window.scrollTo(0, 280)")
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_06_analyzer_negative.png"))
        print("  ✓ Captured ui_06_analyzer_negative.png")
        
        # 2D. Neutral Demo Execution (Fixing base model bias)
        print("  Running Neutral Demo...")
        page.locator("textarea").first.fill("The company announced that its annual shareholder meeting will take place next month.")
        page.wait_for_timeout(400)
        page.get_by_role("button", name="Analyze Sentiment").click()
        page.wait_for_timeout(3500)
        page.evaluate("window.scrollTo(0, 280)")
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_07_analyzer_neutral.png"))
        print("  ✓ Captured ui_07_analyzer_neutral.png")
        
        # ═════════════════════════════════════════════════════════════════════
        # 3. BATCH ANALYSIS
        # ═════════════════════════════════════════════════════════════════════
        print("\n[Page 3] Batch Analysis...")
        page.get_by_text("Batch Analysis").click()
        page.wait_for_timeout(2000)
        
        # 3A. Controls View
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_08_batch_controls.png"))
        print("  ✓ Captured ui_08_batch_controls.png")
        
        # 3B. Load Sample Batch & Analyze
        try:
            sample_btn = page.get_by_role("button", name="Load Sample Batch")
            if sample_btn.is_visible():
                sample_btn.click()
                page.wait_for_timeout(1500)
                run_btn = page.get_by_role("button", name="Run Batch Analysis")
                if run_btn.is_visible():
                    run_btn.click()
                    page.wait_for_timeout(8000)
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(1000)
                    page.screenshot(path=str(SCREENSHOTS_DIR / "ui_09_batch_results.png"))
                    print("  ✓ Captured ui_09_batch_results.png")
        except Exception as e:
            print(f"  Note on batch analysis: {e}")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.screenshot(path=str(SCREENSHOTS_DIR / "ui_09_batch_results.png"))
            
        # ═════════════════════════════════════════════════════════════════════
        # 4. MODEL COMPARISON
        # ═════════════════════════════════════════════════════════════════════
        print("\n[Page 4] Model Comparison...")
        page.get_by_text("Model Comparison").click()
        page.wait_for_timeout(2000)
        
        try:
            cmp_btn = page.get_by_role("button", name="Compare Both Models")
            if cmp_btn.is_visible():
                cmp_btn.click()
                page.wait_for_timeout(6000)
        except Exception as e:
            print(f"  Note on comparison button: {e}")
            
        # 4A. Side-by-Side Cards View
        page.evaluate("window.scrollTo(0, 200)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_10_comparison_cards.png"))
        print("  ✓ Captured ui_10_comparison_cards.png")
        
        # 4B. Detailed Metrics Table Scrolled View
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_11_comparison_table.png"))
        print("  ✓ Captured ui_11_comparison_table.png")
        
        # ═════════════════════════════════════════════════════════════════════
        # 5. PERFORMANCE DASHBOARD
        # ═════════════════════════════════════════════════════════════════════
        print("\n[Page 5] Performance Dashboard...")
        page.get_by_text("Performance Dashboard").click()
        page.wait_for_timeout(2500)
        
        # 5A. Top: Baseline vs Fine-Tuned Bar Chart
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_12_perf_comparison_chart.png"))
        print("  ✓ Captured ui_12_perf_comparison_chart.png")
        
        # 5B. Middle: Confusion Matrix & Per-Class Performance
        page.evaluate("window.scrollTo(0, 680)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_13_perf_confusion_matrix.png"))
        print("  ✓ Captured ui_13_perf_confusion_matrix.png")
        
        # 5C. Bottom: Loss Curve & Interactive Plotly Chart
        page.evaluate("window.scrollTo(0, 1600)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_14_perf_training_loss.png"))
        print("  ✓ Captured ui_14_perf_training_loss.png")
        
        # ═════════════════════════════════════════════════════════════════════
        # 6. ARCHITECTURE & INFO
        # ═════════════════════════════════════════════════════════════════════
        print("\n[Page 6] Architecture & Info...")
        page.get_by_text("Architecture & Info").click()
        page.wait_for_timeout(2500)
        
        # 6A. Top: LoRA Configuration JSON & System Hardware JSON
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_15_arch_specs_json.png"))
        print("  ✓ Captured ui_15_arch_specs_json.png")
        
        # 6B. Scrolled: Embedded Architecture Diagrams
        page.evaluate("window.scrollTo(0, 950)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "ui_16_arch_diagrams.png"))
        print("  ✓ Captured ui_16_arch_diagrams.png")
        
        browser.close()
        print("\n✓ ALL 16 detailed UI screenshots successfully captured!")

if __name__ == "__main__":
    run()
