"""
RAG Evaluation Engine using Ragas
Implements three reference-free metrics matching standard Ragas definitions:
1. Faithfulness (claims in answer supported by context)
2. Context Relevance (retrieved context chunks pertinent to the question)
3. Answer Relevance (generated answer directly addresses the question)

Operates reference-free (no ground truth / golden dataset required).
Uses existing Groq evaluator LLM and Gemini embeddings via Ragas wrappers.
"""

import os
import json
import re
import asyncio    
import traceback
import logging
import typing as t
from dataclasses import dataclass, field
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field

from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics.base import MetricWithLLM, EvaluationMode
from ragas.llms.prompt import Prompt
from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions

from app.services.ingestion import (
    ResilientEmbeddingsWrapper,
    is_daily_quota_error,
    is_rate_limit_error,
)

logger = logging.getLogger(__name__)

# ================= RAGAS EVALUATOR LLM & EMBEDDINGS =================

def get_eval_llm():
    """Initializes existing ChatGroq model for deterministic evaluation."""
    eval_model = os.getenv("GROQ_EVAL_MODEL", "openai/gpt-oss-120b")
    api_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(
        model=eval_model,
        temperature=0.0,
        groq_api_key=api_key
    )

def get_eval_embeddings():
    """Initializes existing Gemini embeddings model for Answer Relevance with rate limit resilience."""
    raw_emb = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return ResilientEmbeddingsWrapper(raw_emb)


# ================= CUSTOM RAGAS METRIC: CONTEXT RELEVANCE =================
# Built strictly on Ragas MetricWithLLM architecture for Ragas compatibility without outdated imports

class ContextRelevanceChunkVerification(BaseModel):
    chunk_index: t.Any = Field(..., description="0-based integer index of the context chunk")
    is_relevant: t.Any = Field(..., description="1 if the chunk contains information directly relevant, necessary, or helpful to answering the question, 0 otherwise")
    reason: t.Optional[str] = Field(default="", description="Brief reason for the verdict")

class ContextRelevanceEvaluation(BaseModel):
    chunk_evaluations: t.List[ContextRelevanceChunkVerification] = Field(
        default_factory=list,
        description="Relevance evaluations for each retrieved context chunk"
    )
    overall_score: t.Optional[float] = Field(
        None,
        description="Ignored. Score is computed programmatically from chunk_evaluations"
    )

_context_rel_instructions = get_json_format_instructions(ContextRelevanceEvaluation)
_context_rel_parser = RagasoutputParser(pydantic_object=ContextRelevanceEvaluation)

CONTEXT_RELEVANCE_PROMPT = Prompt(
    name="context_relevance",
    instruction="""Given a user question and retrieved document context chunks, evaluate whether each chunk contains pertinent, necessary, and helpful information to address the question.
For each chunk from index 0 to N-1:
- Set chunk_index to the 0-based integer index of the chunk.
- Set is_relevant to 1 if the chunk contains information directly relevant, necessary, or helpful to answering the question.
- Set is_relevant to 0 if the chunk is irrelevant, redundant, or unrelated noise.
- Provide a brief reason for the verdict.
Every retrieved chunk must be evaluated individually.""",
    output_format_instruction=_context_rel_instructions,
    examples=[
        {
            "question": "What is the capital of France?",
            "context": "Chunk 0: Paris is the capital and largest city of France.\nChunk 1: France has a temperate climate.",
            "output": ContextRelevanceEvaluation(
                chunk_evaluations=[
                    ContextRelevanceChunkVerification(chunk_index=0, is_relevant=1, reason="Directly answers the question by identifying Paris."),
                    ContextRelevanceChunkVerification(chunk_index=1, is_relevant=0, reason="Climate information is irrelevant to the capital question.")
                ]
            ).dict()
        }
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="json",
)

@dataclass
class ContextRelevance(MetricWithLLM):
    """
    Ragas-native reference-free metric assessing Context Relevance.
    Measures the proportion of retrieved chunks pertinent to the user's question.
    """
    name: str = "context_relevance"
    evaluation_mode: EvaluationMode = EvaluationMode.qc
    context_prompt: Prompt = field(default_factory=lambda: CONTEXT_RELEVANCE_PROMPT)
    max_retries: int = 1
    last_diagnostics: t.Optional[t.Dict[str, t.Any]] = field(default=None, init=False, repr=False)

    def _safe_extract_chunk_evaluations(self, text: str) -> t.List[t.Dict[str, t.Any]]:
        """
        Safely attempts to extract candidate chunk evaluations from text if standard parsing fails.
        Never trusts raw regex captures directly; parses candidate JSON structures using json.loads.
        """
        if not text or not text.strip():
            return []

        candidates = []

        # 1. Check markdown fenced code blocks: ```json ... ``` or ``` ... ```
        block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if block_match:
            candidates.append(block_match.group(1).strip())

        # 2. Check outermost JSON object { ... } or array [ ... ]
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            candidates.append(text[first_brace : last_brace + 1].strip())

        first_bracket = text.find("[")
        last_bracket = text.rfind("]")
        if first_bracket != -1 and last_bracket > first_bracket:
            candidates.append(text[first_bracket : last_bracket + 1].strip())

        for cand in candidates:
            try:
                data = json.loads(cand)
                if isinstance(data, dict):
                    evals = data.get("chunk_evaluations")
                    if isinstance(evals, list):
                        return [item for item in evals if isinstance(item, dict)]
                elif isinstance(data, list):
                    return [item for item in data if isinstance(item, dict)]
            except Exception:
                continue

        # 3. Limited fallback: Find individual JSON objects containing "chunk_index"
        extracted = []
        for match in re.finditer(r'\{[^{}]*"chunk_index"[^{}]*\}', text):
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict) and "chunk_index" in obj and "is_relevant" in obj:
                    extracted.append(obj)
            except Exception:
                continue

        return extracted

    async def _ascore(self, row: t.Dict, callbacks=None) -> float:
        assert self.llm is not None, "LLM must be configured before evaluation"
        question = row.get("question", "")
        contexts = row.get("contexts", [])
        total_chunks = len(contexts)

        if total_chunks == 0:
            diag = {
                "total_retrieved_chunks": 0,
                "successfully_evaluated_chunks": 0,
                "missing_chunks": 0,
                "invalid_decisions": 0,
                "duplicate_decisions": 0,
                "out_of_range_indices": 0,
            }
            row["diagnostics"] = diag
            self.last_diagnostics = diag
            return 0.0

        formatted_context = "\n\n".join(f"Chunk {i}: {c}" for i, c in enumerate(contexts))
        p_value = self.context_prompt.format(question=question, context=formatted_context)

        result = await self.llm.generate(p_value, callbacks=callbacks)
        text = result.generations[0][0].text

        raw_items: t.List[t.Any] = []
        try:
            parsed = await _context_rel_parser.aparse(text, p_value, self.llm, self.max_retries)
            if parsed is not None and getattr(parsed, "chunk_evaluations", None):
                raw_items = parsed.chunk_evaluations
        except Exception as pe:
            logger.warning(f"Ragas output parser failed for context relevance: {pe}")

        # Safe JSON extraction fallback if parser produced nothing
        if not raw_items:
            raw_items = self._safe_extract_chunk_evaluations(text)

        # If completely unrecoverable, return np.nan for _clean_score to handle
        if not raw_items:
            logger.error("Context relevance evaluator LLM response could not be parsed or recovered.")
            diag = {
                "total_retrieved_chunks": total_chunks,
                "successfully_evaluated_chunks": 0,
                "missing_chunks": total_chunks,
                "invalid_decisions": 0,
                "duplicate_decisions": 0,
                "out_of_range_indices": 0,
                "error": "Malformed or unrecoverable LLM response",
            }
            row["diagnostics"] = diag
            self.last_diagnostics = diag
            return np.nan

        # Validation pipeline
        # Validation pipeline
        validated_decisions: t.Dict[int, int] = {}
        evaluated_chunk_indices: t.Set[int] = set()
        invalid_decisions = 0
        duplicate_decisions = 0
        out_of_range_indices = 0

        for item in raw_items:
            raw_idx = item.get("chunk_index") if isinstance(item, dict) else getattr(item, "chunk_index", None)
            raw_rel = item.get("is_relevant") if isinstance(item, dict) else getattr(item, "is_relevant", None)

            # 1. Validate chunk index
            # Explicitly reject Boolean chunk indices before integer validation
            if isinstance(raw_idx, bool):
                invalid_decisions += 1
                continue
            elif isinstance(raw_idx, int):
                idx = raw_idx
            elif isinstance(raw_idx, str):
                raw_idx_str = raw_idx.strip()
                if raw_idx_str.isdigit() or (raw_idx_str.startswith("-") and raw_idx_str[1:].isdigit()):
                    idx = int(raw_idx_str)
                else:
                    invalid_decisions += 1
                    continue
            else:
                invalid_decisions += 1
                continue

            # Bounds check
            if idx < 0 or idx >= total_chunks:
                out_of_range_indices += 1
                continue

            # Duplicate check: once a valid decision is stored, ignore subsequent duplicates
            if idx in evaluated_chunk_indices:
                duplicate_decisions += 1
                continue

            # 2. Validate relevance value
            # Explicitly accept valid 0/1 and Boolean relevance values; reject non-binary relevance values
            rel_val: t.Optional[int] = None
            if isinstance(raw_rel, bool):
                rel_val = 1 if raw_rel else 0
            elif isinstance(raw_rel, int):
                if raw_rel in (0, 1):
                    rel_val = raw_rel
                else:
                    # Non-binary integer (e.g. 2, -1) -> reject
                    invalid_decisions += 1
            elif isinstance(raw_rel, str):
                cleaned_rel = raw_rel.strip().lower()
                if cleaned_rel in ("1", "true"):
                    rel_val = 1
                elif cleaned_rel in ("0", "false"):
                    rel_val = 0
                else:
                    # Non-binary string -> reject
                    invalid_decisions += 1
            elif isinstance(raw_rel, float):
                if raw_rel == 1.0:
                    rel_val = 1
                elif raw_rel == 0.0:
                    rel_val = 0
                else:
                    # Non-binary float -> reject
                    invalid_decisions += 1
            else:
                # Any other type (None, list, dict) -> reject
                invalid_decisions += 1

            # Mark the chunk index as evaluated only after a valid binary decision is obtained
            if rel_val is not None:
                validated_decisions[idx] = rel_val
                evaluated_chunk_indices.add(idx)

        # Missing chunks tracking: missing evaluations and invalid decisions are treated as irrelevant (0)
        missing_indices = [i for i in range(total_chunks) if i not in validated_decisions]
        missing_chunks = len(missing_indices)

        final_decisions = {i: 0 for i in missing_indices}
        final_decisions.update(validated_decisions)

        # Calculate final relevance score exclusively in Python
        # If all decisions are invalid, relevant_count is 0 and returns 0.0
        relevant_count = sum(final_decisions[i] for i in range(total_chunks))
        score = relevant_count / total_chunks
        score = float(max(0.0, min(1.0, score)))

        diag = {
            "total_retrieved_chunks": total_chunks,
            "successfully_evaluated_chunks": len(validated_decisions),
            "missing_chunks": missing_chunks,
            "invalid_decisions": invalid_decisions,
            "duplicate_decisions": duplicate_decisions,
            "out_of_range_indices": out_of_range_indices,
            "evaluated_chunk_indices": sorted(list(validated_decisions.keys())),
            "missing_chunk_indices": missing_indices,
        }
        if len(validated_decisions) == 0 and invalid_decisions > 0:
            diag["note"] = "All chunk relevance decisions were invalid; treated as 0"

        row["diagnostics"] = diag
        self.last_diagnostics = diag

        return score

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        pass

    def save(self, cache_dir: str | None = None) -> None:
        pass


# Global Context Relevance Metric Instance
context_relevance_metric = ContextRelevance()

_run_config = RunConfig(timeout=60, max_retries=2)
_initialized = False

def _ensure_ragas_initialized():
    """Configures LLM & Embeddings wrappers on Ragas metrics."""
    global _initialized
    if _initialized:
        return

    groq_llm = get_eval_llm()
    ragas_llm = LangchainLLMWrapper(groq_llm, run_config=_run_config)

    gemini_emb = get_eval_embeddings()
    ragas_embeddings = LangchainEmbeddingsWrapper(gemini_emb)

    # 1. Faithfulness
    faithfulness.llm = ragas_llm
    faithfulness.init(_run_config)

    # 2. Context Relevance
    context_relevance_metric.llm = ragas_llm
    context_relevance_metric.init(_run_config)

    # 3. Answer Relevancy
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings
    answer_relevancy.strictness = 1
    answer_relevancy.init(_run_config)

    _initialized = True


# ================= HELPER FUNCTIONS =================

def _run_coroutine(coro):
    """Executes a coroutine safely across worker threads and running event loops."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return loop.run_until_complete(coro)


def _clean_score(val: t.Any) -> t.Optional[float]:
    """Ensures evaluation score is a valid float strictly between 0.0 and 1.0, or None."""
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return None
    try:
        num = float(val)
        num = max(0.0, min(1.0, num))
        return round(num, 4)
    except (ValueError, TypeError):
        return None


# ================= INDIVIDUAL METRIC EVALUATORS =================

def evaluate_faithfulness(question: str, answer: str, contexts: t.List[str]) -> t.Tuple[t.Optional[float], t.Dict[str, t.Any]]:
    """
    Evaluates whether the generated response is strictly faithful to the retrieved context using Ragas.
    Returns: (score [0.0 - 1.0] or None, details_dict)
    """
    if not contexts:
        if "cannot find the answer" in answer.lower() or "not contained in the" in answer.lower():
            return 1.0, {"reason": "Refusal based on empty context", "score": 1.0, "source": "ragas_rule"}
        return 0.0, {"reason": "Empty context with ungrounded claims", "score": 0.0, "source": "ragas_rule"}

    try:
        _ensure_ragas_initialized()
        row = {
            "question": question,
            "answer": answer,
            "contexts": contexts
        }
        raw_score = _run_coroutine(faithfulness.ascore(row))
        score = _clean_score(raw_score)
        return score, {"score": score, "source": "ragas_faithfulness"}
    except Exception as e:
        logger.error(f"Ragas faithfulness error: {e}", exc_info=True)
        return None, {"error": str(e), "traceback": traceback.format_exc(), "source": "ragas_faithfulness"}


def evaluate_context_relevance(question: str, contexts: t.List[str]) -> t.Tuple[t.Optional[float], t.Dict[str, t.Any]]:
    """
    Evaluates how relevant and focused the retrieved chunks are with respect to the question using Ragas.
    Returns: (score [0.0 - 1.0] or None, details_dict)
    """
    if not contexts:
        diag = {
            "total_retrieved_chunks": 0,
            "successfully_evaluated_chunks": 0,
            "missing_chunks": 0,
            "invalid_decisions": 0,
            "duplicate_decisions": 0,
            "out_of_range_indices": 0,
        }
        return 0.0, {
            "reason": "No context chunks retrieved",
            "score": 0.0,
            "source": "ragas_context_relevance",
            "diagnostics": diag,
        }

    try:
        _ensure_ragas_initialized()
        row = {
            "question": question,
            "contexts": contexts
        }
        raw_score = _run_coroutine(context_relevance_metric.ascore(row))
        score = _clean_score(raw_score)
        diagnostics = row.get("diagnostics") or getattr(context_relevance_metric, "last_diagnostics", None)

        details: t.Dict[str, t.Any] = {
            "score": score,
            "source": "ragas_context_relevance",
        }
        if diagnostics is not None:
            details["diagnostics"] = diagnostics
        if score is None:
            details["error"] = (diagnostics or {}).get("error", "Context relevance evaluation failed or was invalid")

        return score, details
    except Exception as e:
        logger.error(f"Ragas context relevance error: {e}", exc_info=True)
        return None, {"error": str(e), "traceback": traceback.format_exc(), "source": "ragas_context_relevance"}


def evaluate_answer_relevance(question: str, answer: str, contexts: t.Optional[t.List[str]] = None) -> t.Tuple[t.Optional[float], t.Dict[str, t.Any]]:
    """
    Evaluates whether the generated answer directly addresses the user's question using Ragas.
    Returns: (score [0.0 - 1.0] or None, details_dict)
    """
    if not answer or not answer.strip():
        return 0.0, {"reason": "Empty answer", "score": 0.0, "source": "ragas_answer_relevance"}

    try:
        _ensure_ragas_initialized()
        row = {
            "question": question,
            "answer": answer,
            "contexts": contexts or [""]
        }
        raw_score = _run_coroutine(answer_relevancy.ascore(row))
        score = _clean_score(raw_score)
        return score, {"score": score, "source": "ragas_answer_relevance"}
    except Exception as e:
        logger.error(f"Ragas answer relevance error: {e}", exc_info=True)
        if is_daily_quota_error(e):
            err_msg = "Google Gemini API daily quota exhausted"
        elif is_rate_limit_error(e):
            err_msg = "Google Gemini API rate limit reached after retries"
        else:
            err_msg = str(e)
        return None, {"error": err_msg, "traceback": traceback.format_exc(), "source": "ragas_answer_relevance"}


# ================= COMBINED EVALUATION ORCHESTRATOR =================

def evaluate_interaction(
    question: str,
    answer: str,
    contexts: t.List[str]
) -> t.Dict[str, t.Any]:
    """
    Runs all 3 Ragas reference-free evaluations independently.
    Handles individual failures gracefully so a failure in one metric does not affect others.
    """
    details = {}
    errors = []

    # 1. Faithfulness
    f_score, f_details = evaluate_faithfulness(question, answer, contexts)
    details["faithfulness"] = f_details
    if f_score is None and "error" in f_details:
        errors.append(f"Faithfulness: {f_details['error']}")

    # 2. Context Relevance
    c_score, c_details = evaluate_context_relevance(question, contexts)
    details["context_relevance"] = c_details
    if c_score is None and "error" in c_details:
        errors.append(f"Context Relevance: {c_details['error']}")

    # 3. Answer Relevance
    a_score, a_details = evaluate_answer_relevance(question, answer, contexts)
    details["answer_relevance"] = a_details
    if a_score is None and "error" in a_details:
        errors.append(f"Answer Relevance: {a_details['error']}")

    # Determine overall status
    valid_scores = [s for s in [f_score, c_score, a_score] if s is not None]
    if len(valid_scores) == 3:
        status = "completed"
    elif len(valid_scores) > 0:
        status = "partial"
    else:
        status = "failed"

    return {
        "faithfulness": f_score,
        "context_relevance": c_score,
        "answer_relevance": a_score,
        "status": status,
        "error": "; ".join(errors) if errors else None,
        "details": details
    }
