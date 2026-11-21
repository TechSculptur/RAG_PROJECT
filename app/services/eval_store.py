"""
Evaluation Storage & Aggregation Module for Modular RAG
Persists all RAG interactions and evaluation scores in SQLite.
"""

import sqlite3
import json
import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "evaluations.db")

def init_db():
    """Initializes the SQLite database and ensures the schema is ready."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_evaluations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                document_name TEXT,
                question TEXT NOT NULL,
                reformulated_question TEXT,
                answer TEXT NOT NULL,
                context_text TEXT,
                retrieved_chunks TEXT,
                retrieval_latency REAL,
                generation_latency REAL,
                total_latency REAL,
                faithfulness REAL,
                context_relevance REAL,
                answer_relevance REAL,
                eval_status TEXT NOT NULL,
                eval_error TEXT,
                eval_details TEXT,
                created_at TEXT NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_session ON rag_evaluations(session_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_document ON rag_evaluations(document_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_status ON rag_evaluations(eval_status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_created_at ON rag_evaluations(created_at);")
        conn.commit()
    finally:
        conn.close()

# Initialize DB on module import
init_db()

def get_connection():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def save_interaction(
    interaction_id: str,
    session_id: str,
    question: str,
    answer: str,
    document_name: Optional[str] = None,
    reformulated_question: Optional[str] = None,
    context_text: Optional[str] = None,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    retrieval_latency: Optional[float] = None,
    generation_latency: Optional[float] = None,
    total_latency: Optional[float] = None,
    eval_status: str = "evaluating"
) -> Dict[str, Any]:
    """Stores a newly generated interaction with initial evaluation status."""
    chunks_json = json.dumps(retrieved_chunks if retrieved_chunks else [])
    created_at = datetime.utcnow().isoformat() + "Z"

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO rag_evaluations (
                id, session_id, document_name, question, reformulated_question,
                answer, context_text, retrieved_chunks, retrieval_latency,
                generation_latency, total_latency, eval_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            interaction_id,
            session_id or "default",
            document_name or "Unknown Document",
            question,
            reformulated_question,
            answer,
            context_text or "",
            chunks_json,
            retrieval_latency,
            generation_latency,
            total_latency,
            eval_status,
            created_at
        ))
        conn.commit()
        return get_interaction(interaction_id)
    finally:
        conn.close()

def update_evaluation(
    interaction_id: str,
    faithfulness: Optional[float],
    context_relevance: Optional[float],
    answer_relevance: Optional[float],
    eval_status: str,
    eval_error: Optional[str] = None,
    eval_details: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Updates evaluation scores and status for an interaction."""
    details_json = json.dumps(eval_details) if eval_details else None

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE rag_evaluations
            SET faithfulness = ?,
                context_relevance = ?,
                answer_relevance = ?,
                eval_status = ?,
                eval_error = ?,
                eval_details = ?
            WHERE id = ?
        """, (
            faithfulness,
            context_relevance,
            answer_relevance,
            eval_status,
            eval_error,
            details_json,
            interaction_id
        ))
        conn.commit()
        return get_interaction(interaction_id)
    finally:
        conn.close()

def get_interaction(interaction_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single interaction by ID."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rag_evaluations WHERE id = ?", (interaction_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return _format_row(row)
    finally:
        conn.close()

def list_interactions(
    session_id: Optional[str] = None,
    document_name: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Lists historical interactions with optional filtering."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM rag_evaluations WHERE 1=1"
        params = []

        if session_id and isinstance(session_id, str):
            query += " AND session_id = ?"
            params.append(session_id)
        if document_name and isinstance(document_name, str) and document_name != "all":
            query += " AND document_name = ?"
            params.append(document_name)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [_format_row(r) for r in rows]
    finally:
        conn.close()

def get_analytics_summary(
    session_id: Optional[str] = None,
    document_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes statistical aggregate metrics across evaluated interactions.
    Recalculates averages using ONLY successfully evaluated scores for each respective metric.
    Excludes pending and failed evaluations from averages.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        base_query = " FROM rag_evaluations WHERE 1=1"
        params = []

        if session_id:
            base_query += " AND session_id = ?"
            params.append(session_id)
        if document_name and document_name != "all":
            base_query += " AND document_name = ?"
            params.append(document_name)

        # 1. Total counts
        cursor.execute(f"SELECT COUNT(*) as total {base_query}", params)
        total_questions = cursor.fetchone()["total"]

        cursor.execute(f"SELECT COUNT(*) as pending {base_query} AND eval_status IN ('pending', 'evaluating')", params)
        pending_questions = cursor.fetchone()["pending"]

        cursor.execute(f"SELECT COUNT(*) as failed {base_query} AND eval_status = 'failed'", params)
        failed_questions = cursor.fetchone()["failed"]

        cursor.execute(f"SELECT COUNT(*) as completed {base_query} AND eval_status IN ('completed', 'partial')", params)
        evaluated_questions = cursor.fetchone()["completed"]

        # 2. Metric-specific averages and counts (strictly ignoring NULLs)
        cursor.execute(f"""
            SELECT 
                AVG(faithfulness) as avg_faithfulness,
                COUNT(faithfulness) as count_faithfulness,
                AVG(context_relevance) as avg_context_relevance,
                COUNT(context_relevance) as count_context_relevance,
                AVG(answer_relevance) as avg_answer_relevance,
                COUNT(answer_relevance) as count_answer_relevance,
                AVG(retrieval_latency) as avg_retrieval_latency,
                AVG(generation_latency) as avg_generation_latency,
                AVG(total_latency) as avg_total_latency
            {base_query}
        """, params)
        metrics_row = cursor.fetchone()

        # 3. List of unique document names for filtering
        cursor.execute("SELECT DISTINCT document_name FROM rag_evaluations WHERE document_name IS NOT NULL ORDER BY document_name")
        documents = [row["document_name"] for row in cursor.fetchall() if row["document_name"]]

        def round_or_none(val, digits=4):
            return round(val, digits) if val is not None else None

        avg_f = round_or_none(metrics_row["avg_faithfulness"])
        avg_c = round_or_none(metrics_row["avg_context_relevance"])
        avg_a = round_or_none(metrics_row["avg_answer_relevance"])

        valid_metrics = [m for m in [avg_f, avg_c, avg_a] if m is not None]
        avg_overall_score = round(sum(valid_metrics) / len(valid_metrics), 4) if valid_metrics else None

        return {
            "total_questions": total_questions,
            "evaluated_questions": evaluated_questions,
            "pending_questions": pending_questions,
            "failed_questions": failed_questions,
            "avg_overall_score": avg_overall_score,
            "avg_faithfulness": avg_f,
            "count_faithfulness": metrics_row["count_faithfulness"],
            "avg_context_relevance": avg_c,
            "count_context_relevance": metrics_row["count_context_relevance"],
            "avg_answer_relevance": avg_a,
            "count_answer_relevance": metrics_row["count_answer_relevance"],
            "avg_retrieval_latency": round_or_none(metrics_row["avg_retrieval_latency"], 3),
            "avg_generation_latency": round_or_none(metrics_row["avg_generation_latency"], 3),
            "avg_total_latency": round_or_none(metrics_row["avg_total_latency"], 3),
            "documents": documents
        }
    finally:
        conn.close()

def clear_evaluations(session_id: Optional[str] = None, document_name: Optional[str] = None):
    """Deletes evaluation records optionally filtered by session or document."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        query = "DELETE FROM rag_evaluations WHERE 1=1"
        params = []
        if session_id and isinstance(session_id, str):
            query += " AND session_id = ?"
            params.append(session_id)
        if document_name and isinstance(document_name, str) and document_name != "all":
            query += " AND document_name = ?"
            params.append(document_name)
        cursor.execute(query, params)
        conn.commit()
    finally:
        conn.close()

def _format_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Helper to convert sqlite3.Row to standard dict with JSON parsed."""
    d = dict(row)
    try:
        d["retrieved_chunks"] = json.loads(d["retrieved_chunks"]) if d["retrieved_chunks"] else []
    except Exception:
        d["retrieved_chunks"] = []
    try:
        d["eval_details"] = json.loads(d["eval_details"]) if d["eval_details"] else None
    except Exception:
        d["eval_details"] = None
    return d
