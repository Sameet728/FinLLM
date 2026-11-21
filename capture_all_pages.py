"""
capture_all_pages.py
Automates deep, scrolled screenshot capture for each page of the FinLLM Streamlit application.
Captures both top and scrolled-to-bottom views for all 6 pages with live executed demos.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = Path("C:/Users/samee/Desktop/LLM/FinLLM/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        
        base_url = "http://localhost:8501"
        print(f"Connecting to {base_url}...")
        page.goto(base_url, timeout=30000)
        page.wait_for_timeout(3000)
        
        # ── 1. Overview Dashboard ─────────────────────────────────────────────
        print("\n[Page 1] Overview Dashboard...")
        page.get_by_text("Overview Dashboard").click()
        page.wait_for_timeout(2000)
        # Top view
        page.screenshot(path=str(SCREENSHOTS_DIR / "21_application_home.png"))
        print("  ✓ Captured 21_application_home.png (Top)")
        # Scroll to bottom
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "21b_overview_scrolled.png"))
        print("  ✓ Captured 21b_overview_scrolled.png (Scrolled)")
        
        # ── 2. Financial News Analyzer ─────────────────────────────────────────
        print("\n[Page 2] Financial News Analyzer...")
        page.get_by_text("Financial News Analyzer").click()
        page.wait_for_timeout(2000)
        
        # 2A. Positive Demo
        print("  Running Positive demo...")
        page.locator("textarea").first.fill("The company reported a 30% increase in quarterly revenue, exceeding analyst expectations.")
        page.wait_for_timeout(500)
        page.get_by_role("button", name="Analyze Sentiment").click()
        page.wait_for_timeout(3500)
        page.evaluate("window.scrollTo(0, 300)")
        page.screenshot(path=str(SCREENSHOTS_DIR / "22_positive_prediction.png"))
        print("  ✓ Captured 22_positive_prediction.png")
        
        # 2B. Negative Demo
        print("  Running Negative demo...")
        page.locator("textarea").first.fill("The company announced substantial losses following a sharp decline in consumer demand.")
        page.wait_for_timeout(500)
        page.get_by_role("button", name="Analyze Sentiment").click()
        page.wait_for_timeout(3500)
        page.evaluate("window.scrollTo(0, 300)")
        page.screenshot(path=str(SCREENSHOTS_DIR / "23_negative_prediction.png"))
        print("  ✓ Captured 23_negative_prediction.png")
        
        # 2C. Neutral Demo
        print("  Running Neutral demo...")
        page.locator("textarea").first.fill("The company announced that its annual shareholder meeting will take place next month.")
        page.wait_for_timeout(500)
        page.get_by_role("button", name="Analyze Sentiment").click()
        page.wait_for_timeout(3500)
        page.evaluate("window.scrollTo(0, 300)")
        page.screenshot(path=str(SCREENSHOTS_DIR / "24_neutral_prediction.png"))
        print("  ✓ Captured 24_neutral_prediction.png")
        
        # ── 3. Batch Analysis ──────────────────────────────────────────────────
        print("\n[Page 3] Batch Analysis...")
        page.get_by_text("Batch Analysis").click()
        page.wait_for_timeout(2000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "25_batch_analysis.png"))
        print("  ✓ Captured 25_batch_analysis.png (Top)")
        
        # Click Load Sample Batch
        try:
            sample_btn = page.get_by_role("button", name="Load Sample Batch")
            if sample_btn.is_visible():
                sample_btn.click()
                page.wait_for_timeout(1500)
                # Click Run Batch Analysis
                run_batch_btn = page.get_by_role("button", name="Run Batch Analysis")
                if run_batch_btn.is_visible():
                    run_batch_btn.click()
                    page.wait_for_timeout(8000)
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(1000)
                    page.screenshot(path=str(SCREENSHOTS_DIR / "25b_batch_results_scrolled.png"))
                    print("  ✓ Captured 25b_batch_results_scrolled.png (Scrolled)")
        except Exception as e:
            print(f"  Note on batch analysis demo: {e}")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.screenshot(path=str(SCREENSHOTS_DIR / "25b_batch_results_scrolled.png"))
            
        # ── 4. Model Comparison ────────────────────────────────────────────────
        print("\n[Page 4] Model Comparison...")
        page.get_by_text("Model Comparison").click()
        page.wait_for_timeout(2000)
        
        # Click Compare Both Models
        try:
            compare_btn = page.get_by_role("button", name="Compare Both Models")
            if compare_btn.is_visible():
                compare_btn.click()
                page.wait_for_timeout(6000)
        except Exception as e:
            print(f"  Note on compare click: {e}")
            
        page.evaluate("window.scrollTo(0, 200)")
        page.screenshot(path=str(SCREENSHOTS_DIR / "26_model_comparison_app.png"))
        print("  ✓ Captured 26_model_comparison_app.png (Top/Cards)")
        
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "26b_comparison_metrics_scrolled.png"))
        print("  ✓ Captured 26b_comparison_metrics_scrolled.png (Scrolled/Table)")
        
        # ── 5. Performance Dashboard ───────────────────────────────────────────
        print("\n[Page 5] Performance Dashboard...")
        page.get_by_text("Performance Dashboard").click()
        page.wait_for_timeout(2500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "27_performance_dashboard.png"))
        print("  ✓ Captured 27_performance_dashboard.png (Top)")
        
        page.evaluate("window.scrollTo(0, 900)")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "27b_performance_scrolled.png"))
        print("  ✓ Captured 27b_performance_scrolled.png (Scrolled)")
        
        # ── 6. Architecture & Info ─────────────────────────────────────────────
        print("\n[Page 6] Architecture & Info...")
        page.get_by_text("Architecture & Info").click()
        page.wait_for_timeout(2500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "28_model_information.png"))
        print("  ✓ Captured 28_model_information.png (Top)")
        
        page.evaluate("window.scrollTo(0, 1000)")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "28b_architecture_scrolled.png"))
        print("  ✓ Captured 28b_architecture_scrolled.png (Scrolled)")
        
        browser.close()
        print("\n✓ ALL page and scrolled screenshots successfully captured!")

if __name__ == "__main__":
    run()
