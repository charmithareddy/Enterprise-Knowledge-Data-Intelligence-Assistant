"""
Unit tests for the Enterprise Knowledge Graph & GraphRAG module.
"""

import unittest
from src.graph.knowledge_graph import enterprise_kg

class TestEnterpriseKnowledgeGraph(unittest.TestCase):

    def test_graph_initialization(self):
        metrics = enterprise_kg.get_summary_metrics()
        self.assertGreater(metrics["total_nodes"], 10)
        self.assertGreater(metrics["total_edges"], 10)

    def test_search_subgraph(self):
        res = enterprise_kg.search_subgraph("security incident")
        self.assertIn("matched_entities", res)
        self.assertIn("triples", res)
        self.assertGreater(len(res["triples"]), 0)

    def test_format_graph_context(self):
        context = enterprise_kg.format_graph_context_for_prompt("remote work")
        self.assertIn("KNOWLEDGE GRAPH RELATIONSHIPS", context)
        self.assertIn("Remote Work Policy", context)

    def test_shortest_path(self):
        path = enterprise_kg.find_shortest_path("Engineering", "Data Security & SOC2 Policy")
        self.assertIsNotNone(path)
        self.assertGreaterEqual(len(path), 2)

if __name__ == "__main__":
    unittest.main()
