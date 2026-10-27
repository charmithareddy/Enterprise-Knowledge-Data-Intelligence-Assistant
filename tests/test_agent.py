"""
Unit tests for the Enterprise Agent Intent Classification & Tool Execution.
"""

import unittest
from src.agent.orchestrator import enterprise_agent

class TestEnterpriseAgent(unittest.TestCase):

    def test_intent_classification(self):
        # SQL intent
        self.assertEqual(enterprise_agent.classify_intent("How many employees in Engineering?"), "SQL_ANALYTICS")
        self.assertEqual(enterprise_agent.classify_intent("Total revenue by region in sales"), "SQL_ANALYTICS")

        # ML intent
        self.assertEqual(enterprise_agent.classify_intent("Predict quarterly revenue for next year"), "ML_PREDICTION")
        self.assertEqual(enterprise_agent.classify_intent("What is the retention risk for an engineer?"), "ML_PREDICTION")

        # Graph intent
        self.assertEqual(enterprise_agent.classify_intent("Who governs the incident response protocol?"), "GRAPH_RAG")
        self.assertEqual(enterprise_agent.classify_intent("What is the relationship between Security and MFA?"), "GRAPH_RAG")

        # Document RAG intent
        self.assertEqual(enterprise_agent.classify_intent("What is the refund policy?"), "DOCUMENT_RAG")

    def test_text_to_sql_execution(self):
        res = enterprise_agent.execute_text_to_sql("Show all departments", api_key="")
        self.assertIsNotNone(res["data"])
        self.assertGreater(res["row_count"], 0)

    def test_ml_prediction_execution(self):
        res = enterprise_agent.execute_ml_prediction("Predict revenue for next quarter")
        self.assertEqual(res["type"], "revenue")
        self.assertIn("forecasted_revenue_m", res["result"])

    def test_graph_query_execution(self):
        res = enterprise_agent.execute_graph_query("remote work security")
        self.assertGreater(len(res["triples"]), 0)

if __name__ == "__main__":
    unittest.main()
