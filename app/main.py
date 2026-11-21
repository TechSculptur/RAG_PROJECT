from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import shutil
import uuid
import traceback
from fastapi.responses import FileResponse
from app.services.retrieval import execute_rag_pipeline, rag_chain
from app.services.ingestion import (
    process_pdf,
    GeminiDailyQuotaExhaustedError,
    GeminiRateLimitExceededError,
    is_daily_quota_error,
    is_rate_limit_error,
)
from app.services.evaluation import evaluate_interaction
from app.services import eval_store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Modular RAG API with Live Analytics")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/style.css")
def serve_css():
    return FileResponse(os.path.join(BASE_DIR, "style.css"))

@app.get("/script.js")
def serve_js():
    return FileResponse(os.path.join(BASE_DIR, "script.js"))


class QueryRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]]
    session_id: Optional[str] = "default"
    document_name: Optional[str] = None


def _background_eval_task(interaction_id: str, question: str, answer: str, contexts: List[str]):
    """Asynchronous background worker to compute reference-free evaluation metrics."""
    try:
        eval_result = evaluate_interaction(question, answer, contexts)
        eval_store.update_evaluation(
            interaction_id=interaction_id,
            faithfulness=eval_result.get("faithfulness"),
            context_relevance=eval_result.get("context_relevance"),
            answer_relevance=eval_result.get("answer_relevance"),
            eval_status=eval_result.get("status", "completed"),
            eval_error=eval_result.get("error"),
            eval_details=eval_result.get("details")
        )
    except Exception as e:
        print(f"Background evaluation error for interaction {interaction_id}: {e}")
        traceback.print_exc()
        eval_store.update_evaluation(
            interaction_id=interaction_id,
            faithfulness=None,
            context_relevance=None,
            answer_relevance=None,
            eval_status="failed",
            eval_error=str(e),
            eval_details={"error": str(e), "traceback": traceback.format_exc()}
        )


@app.post("/query")
async def query_documents(request: QueryRequest, background_tasks: BackgroundTasks):
    try:
        format_his = []
        for mes in request.chat_history:
            sender = "human" if mes.get("role") in ["user", "human"] else "ai"
            format_his.append((sender, mes.get("content", "")))

        # 1. Execute RAG retrieval and answer generation with latency tracking
        rag_result = execute_rag_pipeline(request.question, format_his)

        interaction_id = str(uuid.uuid4())

        # Determine document name from request or retrieved chunk metadata
        doc_name = request.document_name
        if not doc_name and rag_result.get("retrieved_chunks"):
            first_chunk_source = rag_result["retrieved_chunks"][0].get("source")
            if first_chunk_source and first_chunk_source != "Unknown":
                doc_name = first_chunk_source

        # 2. Persist interaction in SQLite with initial 'evaluating' status
        eval_store.save_interaction(
            interaction_id=interaction_id,
            session_id=request.session_id or "default",
            question=request.question,
            answer=rag_result["answer"],
            document_name=doc_name or "General",
            reformulated_question=rag_result.get("reformulated_question"),
            context_text=rag_result.get("context_text"),
            retrieved_chunks=rag_result.get("retrieved_chunks"),
            retrieval_latency=rag_result.get("retrieval_latency"),
            generation_latency=rag_result.get("generation_latency"),
            total_latency=rag_result.get("total_latency"),
            eval_status="evaluating"
        )

        # 3. Schedule asynchronous live evaluation without delaying the response
        background_tasks.add_task(
            _background_eval_task,
            interaction_id=interaction_id,
            question=request.question,
            answer=rag_result["answer"],
            contexts=rag_result.get("contexts", [])
        )

        # 4. Return response immediately to caller
        return {
            "question": request.question,
            "answer": rag_result["answer"],
            "interaction_id": interaction_id,
            "retrieval_latency": rag_result.get("retrieval_latency"),
            "generation_latency": rag_result.get("generation_latency"),
            "total_latency": rag_result.get("total_latency"),
            "retrieved_chunks": rag_result.get("retrieved_chunks", []),
            "document_name": doc_name or "General",
            "eval_status": "evaluating"
        }

    except Exception as e:
        traceback.print_exc()
        if is_daily_quota_error(e):
            raise HTTPException(
                status_code=429,
                detail="Google Gemini API daily quota exhausted. Please try again tomorrow or upgrade your API tier."
            )
        if is_rate_limit_error(e):
            raise HTTPException(
                status_code=429,
                detail="Google Gemini API rate limit reached. Please wait a moment and try again."
            )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed."
            )

        data_folder = os.path.join("app", "data")
        os.makedirs(data_folder, exist_ok=True)
        file_path = os.path.join(data_folder, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = process_pdf(file_path)

        if isinstance(result, dict):
            chunks_added = result.get("chunks_added", 0)
            total_chunks = result.get("total_chunks", 0)
            already_indexed = result.get("already_indexed", False)
        else:
            chunks_added = int(result)
            total_chunks = chunks_added
            already_indexed = (chunks_added == 0 and total_chunks > 0)

        if total_chunks == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract readable text from '{file.filename}'. The PDF might be empty or scanned images without searchable text."
            )

        if already_indexed:
            msg = (
                f"'{file.filename}' is already indexed in the vector database "
                f"({total_chunks} chunks). Skipping re-embedding to prevent duplicate chunks."
            )
            displayed_chunks = total_chunks
        else:
            msg = f"Successfully uploaded and processed '{file.filename}' ({chunks_added} chunks added)."
            displayed_chunks = chunks_added

        return {
            "message": msg,
            "filename": file.filename,
            "chunks_added": displayed_chunks,
            "new_chunks_added": chunks_added,
            "total_chunks": total_chunks,
            "already_indexed": already_indexed
        }

    except HTTPException:
        raise
    except GeminiDailyQuotaExhaustedError as e:
        print("UPLOAD ERROR - Daily Quota Exhausted:")
        traceback.print_exc()
        raise HTTPException(
            status_code=429,
            detail=(
                "Google Gemini API daily quota exhausted. "
                "The daily embedding limit has been reached for your API key. "
                "Please wait until tomorrow for the quota to reset, or upgrade your Gemini API tier."
            )
        )
    except GeminiRateLimitExceededError as e:
        print("UPLOAD ERROR - Rate Limit Exceeded:")
        traceback.print_exc()
        raise HTTPException(
            status_code=429,
            detail=(
                "Google Gemini API rate limit reached after multiple retry attempts. "
                "Please wait 1-2 minutes for the rate limit window to clear, then try uploading again."
            )
        )
    except Exception as e:
        print("UPLOAD ERROR:")
        traceback.print_exc()
        if is_daily_quota_error(e):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Google Gemini API daily quota exhausted. "
                    "The daily embedding limit has been reached for your API key. "
                    "Please wait until tomorrow for the quota to reset, or upgrade your Gemini API tier."
                )
            )
        if is_rate_limit_error(e):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Google Gemini API rate limit reached. "
                    "Please wait a moment and try uploading again."
                )
            )
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


# ================= EVALUATION & ANALYTICS REST API =================

@app.get("/api/analytics")
def get_analytics(
    session_id: Optional[str] = None,
    document_name: Optional[str] = None
):
    """Returns statistical aggregate evaluation metrics for the Live Dashboard."""
    try:
        return eval_store.get_analytics_summary(session_id=session_id, document_name=document_name)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evaluations")
def get_evaluations(
    session_id: Optional[str] = None,
    document_name: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Returns chronological list of interaction records with individual evaluation traces."""
    try:
        return eval_store.list_interactions(
            session_id=session_id,
            document_name=document_name,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evaluations/{interaction_id}")
def get_single_evaluation(interaction_id: str):
    """Returns evaluation record for a specific interaction."""
    record = eval_store.get_interaction(interaction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Interaction not found.")
    return record


@app.delete("/api/evaluations")
def clear_evaluation_history(
    session_id: Optional[str] = None,
    document_name: Optional[str] = None
):
    """Clears evaluation history records."""
    try:
        eval_store.clear_evaluations(session_id=session_id, document_name=document_name)
        return {"status": "success", "message": "Evaluation history cleared."}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
