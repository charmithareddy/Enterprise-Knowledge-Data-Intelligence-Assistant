"""
Unit test for FastEmbed Vector Store retrieval on enterprise documents.
"""

import unittest
from src.ingestion.document_manager import DocumentManager
from src.retrieval.vector_store import vector_store
from src.config.settings import DOCS_DIR

class TestVectorStore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        docs = DocumentManager.load_preloaded_documents(DOCS_DIR)
        all_chunks = []
        for doc in docs:
            all_chunks.extend(DocumentManager.chunk_document(doc["text"], doc["name"]))
        vector_store.build_index(all_chunks)

    def test_retrieve_refund_policy(self):
        ctx, chunks = vector_store.retrieve("What is the refund policy?", top_k=2)
        self.assertGreater(len(chunks), 0)
        self.assertIn("REFUND", ctx.upper())

    def test_retrieve_financials(self):
        ctx, chunks = vector_store.retrieve("What was the annual recurring revenue for FY2025?", top_k=2)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(any("142.8" in c["content"] or "ARR" in c["content"] for c in chunks))

    def test_retrieve_security_mfa(self):
        ctx, chunks = vector_store.retrieve("What are the MFA rules for production infrastructure?", top_k=2)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(any("MFA" in c["content"] or "Zero-Trust" in c["content"] for c in chunks))

if __name__ == "__main__":
    unittest.main()
