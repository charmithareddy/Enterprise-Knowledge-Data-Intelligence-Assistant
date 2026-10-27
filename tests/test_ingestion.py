"""
Unit tests for Document Ingestion and Chunking.
"""

import unittest
from pathlib import Path
from src.ingestion.document_manager import DocumentManager
from src.config.settings import DOCS_DIR

class TestDocumentIngestion(unittest.TestCase):

    def test_load_preloaded_documents(self):
        docs = DocumentManager.load_preloaded_documents(DOCS_DIR)
        self.assertGreaterEqual(len(docs), 4)
        names = [d["name"] for d in docs]
        self.assertIn("company_policy.txt", names)
        self.assertIn("q4_financial_earnings.txt", names)
        self.assertIn("it_security_compliance.txt", names)
        self.assertIn("product_sla_guidelines.txt", names)

    def test_policy_aware_chunking(self):
        sample_policy = (
            "1. REFUND POLICY\nCustomers get 30 days.\n\n"
            "2. CUSTOMER SUPPORT\nSupport is available Monday to Friday."
        )
        chunks = DocumentManager.chunk_document(sample_policy, "policy.txt")
        self.assertEqual(len(chunks), 2)
        self.assertIn("1. REFUND POLICY", chunks[0]["section"])
        self.assertIn("2. CUSTOMER SUPPORT", chunks[1]["section"])

if __name__ == "__main__":
    unittest.main()
