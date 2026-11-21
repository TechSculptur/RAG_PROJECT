"""
End-to-End Test for Ragas Evaluation Integration
"""
import os
import sys
import uuid
import time
from dotenv import load_dotenv

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from app.services.evaluation import evaluate_interaction
from app.services import eval_store

print("Starting Ragas E2E Test...")

# 1. Test evaluation directly on a realistic RAG interaction
question = "What is the capital of France and what is its role?"
answer = "Paris is the capital of France and serves as the nation's political, economic, and cultural center."
contexts = [
    "Paris is the capital and most populous city of France.",
    "As the capital, Paris is the seat of the government of France and is also a major European cultural and economic hub."
]

interaction_id = str(uuid.uuid4())

# 2. Save initial interaction in SQLite
print(f"1. Saving interaction {interaction_id} to SQLite...")
record = eval_store.save_interaction(
    interaction_id=interaction_id,
    session_id="test_session_ragas",
    question=question,
    answer=answer,
    document_name="france_facts.pdf",
    context_text="\n\n".join(contexts),
    retrieved_chunks=[{"text": c, "source": "france_facts.pdf", "page": 1} for c in contexts],
    retrieval_latency=0.125,
    generation_latency=0.742,
    total_latency=0.867,
    eval_status="evaluating"
)
print("Saved record initial status:", record["eval_status"])

# 3. Evaluate with Ragas
print("2. Running Ragas evaluation (Faithfulness, Context Relevance, Answer Relevance)...")
eval_res = evaluate_interaction(question, answer, contexts)
print("Evaluation results:", eval_res)

# 4. Update SQLite
print("3. Updating SQLite record with Ragas scores...")
updated = eval_store.update_evaluation(
    interaction_id=interaction_id,
    faithfulness=eval_res.get("faithfulness"),
    context_relevance=eval_res.get("context_relevance"),
    answer_relevance=eval_res.get("answer_relevance"),
    eval_status=eval_res.get("status", "completed"),
    eval_error=eval_res.get("error"),
    eval_details=eval_res.get("details")
)

print("Updated record:")
print(f"  Faithfulness: {updated['faithfulness']}")
print(f"  Context Relevance: {updated['context_relevance']}")
print(f"  Answer Relevance: {updated['answer_relevance']}")
print(f"  Status: {updated['eval_status']}")

# 5. Check Analytics Summary
print("4. Fetching Analytics Summary for test session...")
summary = eval_store.get_analytics_summary(session_id="test_session_ragas")
print("Analytics Summary:")
print(f"  Total Questions: {summary['total_questions']}")
print(f"  Evaluated Questions: {summary['evaluated_questions']}")
print(f"  Avg Faithfulness: {summary['avg_faithfulness']}")
print(f"  Avg Context Relevance: {summary['avg_context_relevance']}")
print(f"  Avg Answer Relevance: {summary['avg_answer_relevance']}")
print(f"  Avg Overall Score (Mean of 3 metrics): {summary['avg_overall_score']}")

# Verification checks
assert updated["eval_status"] == "completed", f"Expected 'completed', got {updated['eval_status']}"
assert updated["faithfulness"] is not None and 0.0 <= updated["faithfulness"] <= 1.0
assert updated["context_relevance"] is not None and 0.0 <= updated["context_relevance"] <= 1.0
assert updated["answer_relevance"] is not None and 0.0 <= updated["answer_relevance"] <= 1.0
assert summary["avg_overall_score"] is not None
print("\n>>> ALL RAGAS VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
