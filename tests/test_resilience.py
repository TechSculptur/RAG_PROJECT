"""
Unit and integration test script for PDF ingestion batching, rate-limit resilience, and error handling.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.documents import Document
from google.api_core.exceptions import ResourceExhausted, TooManyRequests, InternalServerError

from app.services.ingestion import (
    generate_chunk_id,
    is_daily_quota_error,
    is_rate_limit_error,
    GeminiDailyQuotaExhaustedError,
    GeminiRateLimitExceededError,
    ResilientEmbeddingsWrapper,
    ProcessPdfResult,
    process_pdf
)


class TestIngestionResilience(unittest.TestCase):

    def test_chunk_id_determinism(self):
        """Test that deterministic IDs are identical for the same chunk content and position."""
        doc1 = Document(page_content="Machine learning enables systems to learn from data.", metadata={"page": 2, "source": "app/data/test.pdf"})
        doc2 = Document(page_content="Machine learning enables systems to learn from data.", metadata={"page": 2, "source": "app/data/test.pdf"})
        id1 = generate_chunk_id(doc1, 5, "app/data/test.pdf")
        id2 = generate_chunk_id(doc2, 5, "app/data/test.pdf")
        self.assertEqual(id1, id2, "Identical chunks should produce identical deterministic IDs")

        # Different page or index produces different ID
        id3 = generate_chunk_id(doc1, 6, "app/data/test.pdf")
        self.assertNotEqual(id1, id3)

    def test_daily_quota_detection(self):
        """Test distinguishing daily quota exhaustion from temporary rate limits."""
        daily_err1 = ResourceExhausted("Quota exceeded for quota metric 'Queries' and limit 'Queries per day' of service 'generativelanguage.googleapis.com'")
        self.assertTrue(is_daily_quota_error(daily_err1))
        self.assertTrue(is_rate_limit_error(daily_err1))

        daily_err2 = Exception("GoogleGenerativeAIError: Resource exhausted: Per-day quota exhausted for free tier.")
        self.assertTrue(is_daily_quota_error(daily_err2))

        # Temporary rate limit should NOT be classified as daily quota
        temp_err = ResourceExhausted("Resource has been exhausted (e.g. check quota). Online prediction requests per base model per minute")
        self.assertFalse(is_daily_quota_error(temp_err))
        self.assertTrue(is_rate_limit_error(temp_err))

        temp_429 = TooManyRequests("429 Rate limit exceeded. Try again in a few seconds.")
        self.assertFalse(is_daily_quota_error(temp_429))
        self.assertTrue(is_rate_limit_error(temp_429))

        # General error
        gen_err = ValueError("Invalid input syntax")
        self.assertFalse(is_daily_quota_error(gen_err))
        self.assertFalse(is_rate_limit_error(gen_err))

    def test_resilient_embeddings_wrapper_backoff(self):
        """Test that ResilientEmbeddingsWrapper retries on 429 and succeeds."""
        mock_raw = MagicMock()
        # Fail twice with 429, then succeed
        mock_raw.embed_documents.side_effect = [
            ResourceExhausted("Temporary rate limit per minute"),
            ResourceExhausted("Temporary rate limit per minute"),
            [[0.1, 0.2, 0.3]]
        ]

        wrapper = ResilientEmbeddingsWrapper(
            underlying=mock_raw,
            max_retries=3,
            initial_backoff=0.01,
            backoff_factor=1.5,
            max_backoff=0.1
        )

        result = wrapper.embed_documents(["Test text"])
        self.assertEqual(result, [[0.1, 0.2, 0.3]])
        self.assertEqual(mock_raw.embed_documents.call_count, 3)

    def test_resilient_embeddings_wrapper_daily_quota_fails_fast(self):
        """Test that ResilientEmbeddingsWrapper fails immediately without retrying on daily quota exhaustion."""
        mock_raw = MagicMock()
        mock_raw.embed_documents.side_effect = ResourceExhausted(
            "Quota exceeded for quota metric 'Queries' and limit 'Queries per day'"
        )

        wrapper = ResilientEmbeddingsWrapper(
            underlying=mock_raw,
            max_retries=5,
            initial_backoff=0.01
        )

        with self.assertRaises(GeminiDailyQuotaExhaustedError):
            wrapper.embed_documents(["Test text"])
        
        # Should only be called once (fail-fast, no retries)
        self.assertEqual(mock_raw.embed_documents.call_count, 1)

    def test_process_pdf_result_backward_compatibility(self):
        """Test that ProcessPdfResult works both as a dict and can be converted to int."""
        res = ProcessPdfResult({
            "total_chunks": 40,
            "chunks_added": 40,
            "already_indexed": False,
            "file_path": "test.pdf"
        })
        self.assertEqual(res["total_chunks"], 40)
        self.assertEqual(res["chunks_added"], 40)
        self.assertEqual(int(res), 40)
        self.assertFalse(res["already_indexed"])

    @patch("app.services.ingestion.load_pdf")
    @patch("app.services.ingestion.Chroma")
    def test_deduplication_skips_existing_chunks(self, mock_chroma_class, mock_load_pdf):
        """Test that if all chunks are already present in ChromaDB, re-embedding is skipped."""
        mock_load_pdf.return_value = [
            Document(page_content="Page 1 content here.", metadata={"page": 0, "source": "dummy.pdf"}),
            Document(page_content="Page 2 content here.", metadata={"page": 1, "source": "dummy.pdf"}),
        ]

        mock_vector_store = MagicMock()
        mock_chroma_class.return_value = mock_vector_store

        # Mock Chroma get returning all IDs as already existing
        def fake_get(ids=None):
            return {"ids": ids or []}
        mock_vector_store.get.side_effect = fake_get

        result = process_pdf("dummy.pdf", batch_size=5)
        self.assertTrue(result["already_indexed"])
        self.assertEqual(result["chunks_added"], 0)
        self.assertGreater(result["total_chunks"], 0)
        # add_documents should NOT have been called at all!
        mock_vector_store.add_documents.assert_not_called()


if __name__ == "__main__":
    unittest.main()
