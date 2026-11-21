import os
import re
import time
import math
import hashlib
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from google.api_core.exceptions import ResourceExhausted, TooManyRequests, GoogleAPICallError
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import pypdf as pdf

# ================= LOGGING SETUP =================
logger = logging.getLogger("ingestion")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [ingestion]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ================= CONFIGURATION =================
# Configurable batch size: defaults to 10 chunks per batch to avoid exceeding API limits
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
# Configurable delay in seconds between consecutive embedding batches
BATCH_DELAY = float(os.getenv("EMBEDDING_BATCH_DELAY", "2.0"))
# Maximum retry attempts for temporary rate limit (429 / RESOURCE_EXHAUSTED) errors
MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "5"))
# Initial backoff delay in seconds
INITIAL_BACKOFF = float(os.getenv("EMBEDDING_INITIAL_BACKOFF", "2.0"))
# Multiplier for exponential backoff
BACKOFF_FACTOR = float(os.getenv("EMBEDDING_BACKOFF_FACTOR", "2.0"))
# Maximum backoff delay cap in seconds
MAX_BACKOFF = float(os.getenv("EMBEDDING_MAX_BACKOFF", "60.0"))


# ================= CUSTOM EXCEPTIONS =================
class GeminiEmbeddingError(Exception):
    """Base exception for Gemini Embedding failures."""
    pass


class GeminiDailyQuotaExhaustedError(GeminiEmbeddingError):
    """Raised when the daily quota for Google Gemini API is exhausted."""
    pass


class GeminiRateLimitExceededError(GeminiEmbeddingError):
    """Raised when temporary 429 rate limit persists after all retry attempts."""
    pass


# ================= ERROR CLASSIFICATION HELPERS =================
def _extract_error_details(exc: Exception) -> str:
    """Combines exception message, cause, and args into a lowercase string for inspection."""
    parts = [str(exc)]
    if hasattr(exc, "message") and exc.message:
        parts.append(str(exc.message))
    cause = getattr(exc, "__cause__", None)
    if cause:
        parts.append(str(cause))
        if hasattr(cause, "message") and cause.message:
            parts.append(str(cause.message))
    if hasattr(exc, "args"):
        parts.extend(str(a) for a in exc.args)
    return " ".join(parts).lower()


def is_daily_quota_error(exc: Exception) -> bool:
    """
    Checks if an exception indicates that the Google Gemini API daily quota is exhausted.
    Daily quota exhaustion should NOT be retried indefinitely.
    """
    err_text = _extract_error_details(exc)
    normalized_text = err_text.replace("-", " ").replace("_", " ")
    daily_indicators = [
        "per day",
        "per_day",
        "per-day",
        "daily",
        "day limit",
        "queries per day",
        "requests per day",
        "generaterequestsperday",
        "free tier limit reached for the day",
        "quota exceeded for quota metric 'queries' and limit 'queries per day'",
        "daily quota exceeded",
        "exhausted daily quota",
        "quota exceeded for metric 'queries' and limit 'queries per day'",
    ]
    return any(indicator in err_text or indicator in normalized_text for indicator in daily_indicators)


def is_rate_limit_error(exc: Exception) -> bool:
    """
    Checks if an exception is a rate limit error (HTTP 429 or RESOURCE_EXHAUSTED).
    Distinguishes temporary rate limits from other API failures.
    """
    if isinstance(exc, (ResourceExhausted, TooManyRequests)):
        return True
    cause = getattr(exc, "__cause__", None)
    if cause and isinstance(cause, (ResourceExhausted, TooManyRequests)):
        return True

    # Check status code if present
    code = getattr(exc, "code", None) or (getattr(cause, "code", None) if cause else None)
    if code == 429:
        return True

    err_text = _extract_error_details(exc)
    rate_indicators = [
        "429",
        "resource_exhausted",
        "resource has been exhausted",
        "too many requests",
        "rate limit",
        "quota exceeded",
        "per minute",
        "per_minute",
        "queries per minute",
        "requests per minute",
    ]
    return any(indicator in err_text for indicator in rate_indicators)


# ================= RESILIENT EMBEDDINGS WRAPPER =================
class ResilientEmbeddingsWrapper(Embeddings):
    """
    Wraps a LangChain Embeddings instance to provide exponential backoff retry
    on temporary HTTP 429 / RESOURCE_EXHAUSTED errors, while immediately failing fast
    when daily quotas are exhausted.
    """
    def __init__(
        self,
        underlying: Embeddings,
        max_retries: int = MAX_RETRIES,
        initial_backoff: float = INITIAL_BACKOFF,
        backoff_factor: float = BACKOFF_FACTOR,
        max_backoff: float = MAX_BACKOFF
    ):
        self.underlying = underlying
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        attempt = 0
        while True:
            try:
                return self.underlying.embed_documents(texts)
            except Exception as exc:
                if is_daily_quota_error(exc):
                    logger.error(f"Gemini API daily quota exhausted in embed_documents: {exc}")
                    raise GeminiDailyQuotaExhaustedError(
                        "Google Gemini API daily quota exhausted. Please wait until tomorrow or upgrade your API tier."
                    ) from exc

                if is_rate_limit_error(exc):
                    attempt += 1
                    if attempt > self.max_retries:
                        logger.error(f"Rate limit in embed_documents exceeded {self.max_retries} retries: {exc}")
                        raise GeminiRateLimitExceededError(
                            f"Google Gemini API rate limit reached after {self.max_retries} retries."
                        ) from exc
                    delay = min(self.initial_backoff * (self.backoff_factor ** (attempt - 1)), self.max_backoff)
                    logger.warning(
                        f"Rate limit in embed_documents. Retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    raise

    def embed_query(self, text: str) -> List[float]:
        attempt = 0
        while True:
            try:
                return self.underlying.embed_query(text)
            except Exception as exc:
                if is_daily_quota_error(exc):
                    logger.error(f"Gemini API daily quota exhausted in embed_query: {exc}")
                    raise GeminiDailyQuotaExhaustedError(
                        "Google Gemini API daily quota exhausted. Please wait until tomorrow or upgrade your API tier."
                    ) from exc

                if is_rate_limit_error(exc):
                    attempt += 1
                    if attempt > self.max_retries:
                        logger.error(f"Rate limit in embed_query exceeded {self.max_retries} retries: {exc}")
                        raise GeminiRateLimitExceededError(
                            f"Google Gemini API rate limit reached after {self.max_retries} retries."
                        ) from exc
                    delay = min(self.initial_backoff * (self.backoff_factor ** (attempt - 1)), self.max_backoff)
                    logger.warning(
                        f"Rate limit in embed_query. Retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    raise


# ================= PDF LOADING & TEXT SPLITTING =================
def load_pdf(path: str) -> List[Document]:
    docs = []
    with open(path, 'rb') as f1:
        data = pdf.PdfReader(f1)
        for ind, cont in enumerate(data.pages):
            text = cont.extract_text()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path,
                            "page": ind
                        }
                    )
                )
    return docs


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

# Underlying Gemini Embeddings
raw_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
embeddings = ResilientEmbeddingsWrapper(raw_embeddings)


# ================= DETERMINISTIC CHUNK ID GENERATOR =================
def generate_chunk_id(chunk: Document, chunk_index: int, source_path: str) -> str:
    """
    Generates a deterministic ID for a chunk based on source filename, page, index, and content hash.
    Ensures that identical chunks across duplicate uploads produce the same ID, preventing duplicates in ChromaDB.
    """
    page = chunk.metadata.get("page", 0)
    base_name = os.path.basename(source_path) if source_path else "doc"
    safe_base_name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', base_name)
    content_hash = hashlib.sha256(
        f"{safe_base_name}_p{page}_idx{chunk_index}_{chunk.page_content.strip()}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{safe_base_name}_p{page}_idx{chunk_index}_{content_hash}"


# ================= MAIN INGESTION FUNCTION =================
class ProcessPdfResult(dict):
    """Dictionary result representing ingestion outcome, castable to int for backward compatibility."""
    def __int__(self):
        return self.get("chunks_added", 0)


def process_pdf(
    file_path: str,
    batch_size: int = BATCH_SIZE,
    batch_delay: float = BATCH_DELAY,
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF,
    backoff_factor: float = BACKOFF_FACTOR,
    max_backoff: float = MAX_BACKOFF
) -> ProcessPdfResult:
    """
    Processes a PDF file, splits it into chunks, checks ChromaDB for existing chunks
    to avoid re-embedding duplicates, and embeds new chunks in configurable batches
    with exponential backoff and inter-batch delays.

    Returns:
        ProcessPdfResult dict with keys:
            - "total_chunks": Total number of chunks in the document
            - "chunks_added": Number of new chunks embedded and added
            - "already_indexed": True if all chunks were already present in ChromaDB
            - "file_path": Path to the processed PDF file
    """
    logger.info(f"Starting ingestion for file: {file_path}")
    docs = load_pdf(file_path)
    if not docs:
        logger.warning(f"No text extracted from PDF: {file_path}")
        return ProcessPdfResult({
            "total_chunks": 0,
            "chunks_added": 0,
            "already_indexed": False,
            "file_path": file_path
        })

    chunks = splitter.split_documents(docs)
    total_chunks = len(chunks)
    if not chunks:
        logger.warning(f"No chunks produced after splitting: {file_path}")
        return ProcessPdfResult({
            "total_chunks": 0,
            "chunks_added": 0,
            "already_indexed": False,
            "file_path": file_path
        })

    logger.info(f"Loaded {len(docs)} page(s) and split into {total_chunks} chunk(s) for '{file_path}'.")

    # Connect to existing ChromaDB vector store
    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    # 1. Generate deterministic IDs for all chunks
    all_chunk_ids = [generate_chunk_id(chunk, idx, file_path) for idx, chunk in enumerate(chunks)]

    # 2. Check for existing chunks in ChromaDB to prevent duplicate embedding
    existing_ids = set()
    try:
        existing_records = vector_store.get(ids=all_chunk_ids)
        if existing_records and "ids" in existing_records:
            existing_ids = set(existing_records["ids"])
            logger.info(f"Found {len(existing_ids)} existing chunk(s) already in ChromaDB out of {total_chunks}.")
    except Exception as e:
        logger.warning(f"Could not check existing IDs in ChromaDB ({e}). Proceeding to embed all chunks.")
        existing_ids = set()

    # Filter out chunks that are already embedded and present in ChromaDB
    chunks_to_embed: List[Document] = []
    ids_to_embed: List[str] = []
    for chunk, cid in zip(chunks, all_chunk_ids):
        if cid not in existing_ids:
            chunks_to_embed.append(chunk)
            ids_to_embed.append(cid)

    # If all chunks are already indexed, skip embedding entirely
    if not chunks_to_embed:
        logger.info(
            f"All {total_chunks} chunks from '{file_path}' are already embedded and stored in ChromaDB. "
            "Skipping re-embedding to avoid duplicates and save API quota."
        )
        return ProcessPdfResult({
            "total_chunks": total_chunks,
            "chunks_added": 0,
            "already_indexed": True,
            "file_path": file_path
        })

    total_new = len(chunks_to_embed)
    total_batches = math.ceil(total_new / batch_size)
    logger.info(
        f"Embedding {total_new} new chunk(s) across {total_batches} batch(es) "
        f"(batch size = {batch_size}, inter-batch delay = {batch_delay}s, max retries = {max_retries})."
    )

    chunks_added_count = 0
    # Process each batch
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_new)
        batch_chunks = chunks_to_embed[start_idx:end_idx]
        batch_ids = ids_to_embed[start_idx:end_idx]
        batch_num = batch_idx + 1

        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks, items {start_idx + 1} to {end_idx})...")

        # Retry loop with exponential backoff for this batch
        attempt = 0
        batch_success = False

        while attempt <= max_retries and not batch_success:
            try:
                # Add documents with deterministic IDs
                vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
                batch_success = True
                chunks_added_count += len(batch_chunks)
                logger.info(f"Batch {batch_num}/{total_batches} successfully embedded and added to ChromaDB.")
            except Exception as exc:
                # 1. Distinguish daily quota exhaustion: do NOT retry indefinitely
                if is_daily_quota_error(exc):
                    logger.error(
                        f"FAILED batch {batch_num}/{total_batches}: Gemini API daily quota exhausted: {exc}"
                    )
                    raise GeminiDailyQuotaExhaustedError(
                        "Google Gemini API daily quota has been exhausted. "
                        "Please wait until tomorrow for the quota to reset, or upgrade your Gemini API tier."
                    ) from exc

                # 2. Check if temporary rate limit error (429 / RESOURCE_EXHAUSTED)
                if is_rate_limit_error(exc):
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(
                            f"FAILED batch {batch_num}/{total_batches}: Rate limit exceeded after {max_retries} retries."
                        )
                        raise GeminiRateLimitExceededError(
                            f"Google Gemini API rate limit reached. Exceeded {max_retries} retries on batch {batch_num}/{total_batches}. "
                            "Please wait a few moments before trying again."
                        ) from exc

                    backoff_delay = min(initial_backoff * (backoff_factor ** (attempt - 1)), max_backoff)
                    logger.warning(
                        f"Gemini API rate limit on batch {batch_num}/{total_batches}. "
                        f"Backoff attempt {attempt}/{max_retries} in {backoff_delay:.1f}s. Error: {exc}"
                    )
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"FAILED batch {batch_num}/{total_batches}: Non-rate-limit error: {exc}")
                    raise

        # Configurable delay between batches to respect RPM limits (skipped after last batch)
        if batch_num < total_batches and batch_delay > 0:
            logger.info(f"Waiting {batch_delay:.1f}s before batch {batch_num + 1}/{total_batches} to respect Gemini API rate limits...")
            time.sleep(batch_delay)

    logger.info(
        f"Successfully processed '{file_path}': added {chunks_added_count} new chunks "
        f"({total_chunks} total chunks indexed in ChromaDB)."
    )
    return ProcessPdfResult({
        "total_chunks": total_chunks,
        "chunks_added": chunks_added_count,
        "already_indexed": False,
        "file_path": file_path
    })


if __name__ == "__main__":
    pass
