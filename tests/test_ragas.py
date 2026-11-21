import os
import sys
import asyncio
from dotenv import load_dotenv

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics.base import MetricWithLLM, EvaluationMode
from ragas.llms.prompt import Prompt
from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from langchain_core.pydantic_v1 import BaseModel, Field
from dataclasses import dataclass, field
import typing as t
import numpy as np

# 1. Custom Ragas Context Relevance Metric
class ContextRelevanceChunkVerification(BaseModel):
    chunk_index: int = Field(..., description="Index of the context chunk")
    is_relevant: int = Field(..., description="1 if the chunk contains information directly relevant to the question, 0 otherwise")
    reason: str = Field(..., description="Reason for the verdict")

class ContextRelevanceEvaluation(BaseModel):
    chunk_evaluations: t.List[ContextRelevanceChunkVerification] = Field(..., description="Relevance evaluations for each chunk")
    overall_score: float = Field(..., description="Proportion of retrieved context pertinent to the question between 0.0 and 1.0")

_context_rel_instructions = get_json_format_instructions(ContextRelevanceEvaluation)
_context_rel_parser = RagasoutputParser(pydantic_object=ContextRelevanceEvaluation)

CONTEXT_RELEVANCE_PROMPT = Prompt(
    name="context_relevance",
    instruction="""Given a user question and retrieved document context chunks, evaluate whether the retrieved context chunks contain pertinent, necessary, and helpful information to address the question.
For each chunk, determine if it is relevant (1) or irrelevant (0).
Calculate the overall_score between 0.0 and 1.0 as the proportion of relevant chunks. If no chunks are relevant, score is 0.0.""",
    output_format_instruction=_context_rel_instructions,
    examples=[
        {
            "question": "What is the capital of France?",
            "context": "Chunk 0: Paris is the capital and largest city of France.",
            "output": ContextRelevanceEvaluation(
                chunk_evaluations=[
                    ContextRelevanceChunkVerification(chunk_index=0, is_relevant=1, reason="Directly answers the capital of France.")
                ],
                overall_score=1.0
            ).dict()
        }
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="json",
)

@dataclass
class ContextRelevance(MetricWithLLM):
    name: str = "context_relevance"
    evaluation_mode: EvaluationMode = EvaluationMode.qc
    context_prompt: Prompt = field(default_factory=lambda: CONTEXT_RELEVANCE_PROMPT)
    max_retries: int = 1

    async def _ascore(self, row: t.Dict, callbacks=None) -> float:
        assert self.llm is not None, "LLM must be set"
        question = row["question"]
        contexts = row["contexts"]
        if not contexts:
            return 0.0
        
        formatted_context = "\n".join(f"Chunk {i}: {c}" for i, c in enumerate(contexts))
        p_value = self.context_prompt.format(question=question, context=formatted_context)
        
        result = await self.llm.generate(p_value, callbacks=callbacks)
        text = result.generations[0][0].text
        
        parsed = await _context_rel_parser.aparse(text, p_value, self.llm, self.max_retries)
        if parsed is None:
            return np.nan
        
        score = parsed.overall_score
        if score is None and parsed.chunk_evaluations:
            score = sum(1 for c in parsed.chunk_evaluations if c.is_relevant == 1) / len(parsed.chunk_evaluations)
        return float(max(0.0, min(1.0, score or 0.0)))

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        pass

    def save(self, cache_dir: str | None = None) -> None:
        pass

context_relevance = ContextRelevance()

# Setup LLM & Embeddings
eval_model = os.getenv("GROQ_EVAL_MODEL", "openai/gpt-oss-120b")
groq_llm = ChatGroq(model=eval_model, temperature=0.0, groq_api_key=os.getenv("GROQ_API_KEY"))
ragas_llm = LangchainLLMWrapper(groq_llm)

gemini_emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
ragas_embeddings = LangchainEmbeddingsWrapper(gemini_emb)

# Assign to metrics
faithfulness.llm = ragas_llm
context_relevance.llm = ragas_llm
answer_relevancy.llm = ragas_llm
answer_relevancy.embeddings = ragas_embeddings
answer_relevancy.strictness = 1

sample_row = {
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris.",
    "contexts": ["Paris is the capital and most populous city of France."]
}

async def run_test():
    from ragas.run_config import RunConfig
    rc = RunConfig()
    faithfulness.init(rc)
    context_relevance.init(rc)
    answer_relevancy.init(rc)
    
    print("Evaluating Faithfulness...")
    f_score = await faithfulness.ascore(sample_row)
    print("Faithfulness:", f_score)
    
    print("Evaluating Context Relevance...")
    c_score = await context_relevance.ascore(sample_row)
    print("Context Relevance:", c_score)
    
    print("Evaluating Answer Relevancy...")
    a_score = await answer_relevancy.ascore(sample_row)
    print("Answer Relevancy:", a_score)

if __name__ == "__main__":
    asyncio.run(run_test())
