"""
FinLLM — LangChain Agent Workflow Verification Suite
====================================================
Tests the structural integrity, tool registrations, docstrings,
and intent routing of the LangChain Financial Intelligence Agent.

Usage:
    python tests/test_agent_workflow.py

Student: Sameet Pisal | PRN: 202401120018 | Roll: 41
Department of Data Science | MIT Academy of Engineering, Pune
"""

import sys
import json
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import (
    FinancialNewsAgent,
    analyze_sentiment,
    analyze_batch,
    summarize_sentiment,
    generate_analysis_report
)


class TestAgentWorkflow(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.agent = FinancialNewsAgent()

    def test_01_tool_registration(self):
        """Verify that all four required tools are registered in the agent."""
        expected_tools = {
            "analyze_sentiment",
            "analyze_batch",
            "summarize_sentiment",
            "generate_analysis_report"
        }
        registered = set(self.agent.tools.keys())
        self.assertEqual(expected_tools, registered, "Agent tools do not match expected 4 tools.")
        print("  [PASS] TC-W01: All 4 tools successfully registered.")

    def test_02_tool_metadata_and_docstrings(self):
        """Verify each tool is a valid LangChain tool with docstring and name."""
        for name, tool_obj in self.agent.tools.items():
            self.assertTrue(hasattr(tool_obj, "name"), f"Tool {name} lacks name attribute.")
            self.assertTrue(hasattr(tool_obj, "description"), f"Tool {name} lacks description attribute.")
            self.assertTrue(len(tool_obj.description) > 10, f"Tool {name} description is too short.")
        print("  [PASS] TC-W02: All tools adhere to LangChain tool schema standards.")

    def test_03_intent_routing_single(self):
        """Verify single headline prompt routes to analyze_sentiment."""
        query = "The company reported a 30% increase in quarterly revenue."
        tool = self.agent.route_intent(query)
        self.assertEqual(tool, "analyze_sentiment")
        print("  [PASS] TC-W03: Single headline prompt correctly routed to analyze_sentiment.")

    def test_04_intent_routing_batch(self):
        """Verify multi-headline query routes to analyze_batch."""
        query = "Analyze these headlines:\n1. Revenue rose by 20%\n2. Factory closed down"
        tool = self.agent.route_intent(query)
        self.assertEqual(tool, "analyze_batch")
        print("  [PASS] TC-W04: Multi-line batch query correctly routed to analyze_batch.")

    def test_05_intent_routing_summary(self):
        """Verify distribution summary query routes to summarize_sentiment."""
        query = "Summarize the sentiment distribution and breakdown for Q3 portfolio."
        tool = self.agent.route_intent(query)
        self.assertEqual(tool, "summarize_sentiment")
        print("  [PASS] TC-W05: Distribution request correctly routed to summarize_sentiment.")

    def test_06_intent_routing_report(self):
        """Verify report generation query routes to generate_analysis_report."""
        query = "Generate an executive analysis report for the technology sector."
        tool = self.agent.route_intent(query)
        self.assertEqual(tool, "generate_analysis_report")
        print("  [PASS] TC-W06: Executive brief query correctly routed to generate_analysis_report.")

    def test_07_summarize_sentiment_logic(self):
        """Verify mathematical calculation in summarize_sentiment tool."""
        mock_data = json.dumps({
            "results": [
                {"sentiment": "Positive"},
                {"sentiment": "Positive"},
                {"sentiment": "Positive"},
                {"sentiment": "Negative"},
                {"sentiment": "Neutral"},
            ]
        })
        res_json = summarize_sentiment.invoke(mock_data)
        res = json.loads(res_json)
        self.assertEqual(res["counts"]["Positive"], 3)
        self.assertEqual(res["counts"]["Negative"], 1)
        self.assertEqual(res["counts"]["Neutral"], 1)
        self.assertEqual(res["total_analyzed"], 5)
        self.assertEqual(res["percentages"]["Positive"], "60.0%")
        self.assertIn("Bullish", res["market_sentiment_signal"])
        print("  [PASS] TC-W07: Sentiment distribution and net market score computed correctly.")

    def test_08_generate_report_format(self):
        """Verify formal report generation formatting and disclaimers."""
        context = "Automated test for FinLLM report generation"
        report_text = generate_analysis_report.invoke(context)
        self.assertIn("FINLLM FINANCIAL INTELLIGENCE REPORT", report_text)
        self.assertIn("REGULATORY COMPLIANCE & DISCLAIMER", report_text)
        self.assertIn("Department of Data Science", report_text)
        print("  [PASS] TC-W08: Executive analysis report format and compliance verified.")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FINLLM: LANGCHAIN AGENT WORKFLOW VERIFICATION SUITE")
    print("="*70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAgentWorkflow)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n>>> ALL 8 AGENT WORKFLOW TESTS PASSED (100% PASS RATE) <<<\n")
        sys.exit(0)
    else:
        print("\n>>> WORKFLOW TESTS FAILED <<<\n")
        sys.exit(1)
