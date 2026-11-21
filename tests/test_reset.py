"""
Unit tests for application reset functionality without admin authorization.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import reset_application_data, BASE_DIR
from app.services.retrieval import reset_chroma_store


class TestResetFunctionality(unittest.TestCase):

    def test_reset_direct_execution(self):
        """Reset should execute directly without requiring an admin key."""
        test_pdf = os.path.join(BASE_DIR, "data", "test_dummy_reset.pdf")
        keep_txt = os.path.join(BASE_DIR, "data", "test_keep.txt")
        try:
            # Create a dummy PDF and a non-PDF file
            with open(test_pdf, "w") as f:
                f.write("dummy pdf content")
            with open(keep_txt, "w") as f:
                f.write("dummy keep text")

            with patch("app.main.reset_chroma_store") as mock_chroma, \
                 patch("app.main.eval_store.clear_evaluations") as mock_eval:
                res = reset_application_data()

            self.assertEqual(res["status"], "success")
            self.assertIn("test_dummy_reset.pdf", res["deleted_files"])
            self.assertFalse(os.path.exists(test_pdf), "PDF file should be deleted")
            self.assertTrue(os.path.exists(keep_txt), "Non-PDF file should not be deleted")
            mock_chroma.assert_called_once()
            mock_eval.assert_called_once()
        finally:
            if os.path.exists(test_pdf):
                os.remove(test_pdf)
            if os.path.exists(keep_txt):
                os.remove(keep_txt)

    def test_reset_chroma_store_execution(self):
        """Test reset_chroma_store runs without error on collection."""
        try:
            reset_chroma_store()
        except Exception as e:
            self.fail(f"reset_chroma_store raised exception: {e}")


if __name__ == "__main__":
    unittest.main()
