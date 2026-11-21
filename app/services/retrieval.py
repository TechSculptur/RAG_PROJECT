from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import os
import time
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from app.services.ingestion import (
    ResilientEmbeddingsWrapper,
    GeminiDailyQuotaExhaustedError,
    GeminiRateLimitExceededError,
    is_daily_quota_error,
    is_rate_limit_error,
)

logger = logging.getLogger("retrieval")

load_dotenv()

def format_docs(results):
    docList = [doc.page_content for doc in results]
    return '\n\n'.join(docList)

# Note: Using OpenAI's open-weight GPT-OSS 120B model via Groq!
llm = ChatGroq(model='openai/gpt-oss-120b')

# Resilient Gemini Embeddings with retry and backoff for rate limits
raw_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
embeddings = ResilientEmbeddingsWrapper(raw_embeddings)

contextualize_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be "
    "understood without the chat history. "
    "Do NOT answer the question, "
    "just reformulate it if needed and otherwise "
    "return it as is."
)

cont_pr_temp = ChatPromptTemplate.from_messages([
    ('system', contextualize_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{question}')
])

cont_chain = cont_pr_temp | llm | StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a highly accurate enterprise assistant. "
            "Answer the user's question using ONLY the provided context below. "
            "If the answer is not contained in the context, say 'I cannot find the answer in the provided documents.'\n\n"
            "Context:\n{context}"
        ),
        MessagesPlaceholder('chat_history'),
        (
            "human",
            "{question}"
        )
    ]
)

vector_store = Chroma(
    persist_directory='chroma_db',
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

data_prep = (
    RunnablePassthrough.assign(
        question=cont_chain 
    ).assign(
        context=RunnableLambda(lambda x: x['question']) | retriever | RunnableLambda(format_docs) 
    )
)

rag_chain = data_prep | prompt | llm | StrOutputParser()

def execute_rag_pipeline(question: str, chat_history: list = None) -> dict:
    """
    Executes the full RAG pipeline while preserving detailed evaluation data:
    - User question & standalone reformulated question
    - Retrieved document chunks with individual metadata (source, page)
    - Full formatted context text
    - Generated response
    - Exact retrieval latency and generation latency measurements
    """
    if chat_history is None:
        chat_history = []

    # 1. Contextualize/reformulate question if chat history exists
    reformulated_question = question
    if chat_history:
        try:
            reformulated_question = cont_chain.invoke({
                "question": question,
                "chat_history": chat_history
            })
        except Exception:
            reformulated_question = question

    # 2. Retrieve documents and measure retrieval latency
    t_ret_start = time.perf_counter()
    try:
        retrieved_docs = retriever.invoke(reformulated_question)
    except Exception as exc:
        if is_daily_quota_error(exc):
            logger.error(f"Gemini API daily quota exhausted during retrieval: {exc}")
            raise GeminiDailyQuotaExhaustedError(
                "Google Gemini API daily quota exhausted during retrieval. "
                "Please wait until tomorrow or upgrade your API tier."
            ) from exc
        if is_rate_limit_error(exc):
            logger.error(f"Gemini API rate limit exceeded during retrieval: {exc}")
            raise GeminiRateLimitExceededError(
                "Google Gemini API rate limit reached during retrieval. "
                "Please wait a few moments and try again."
            ) from exc
        raise
    retrieval_latency = round(time.perf_counter() - t_ret_start, 3)

    # 3. Extract individual chunk details and metadata
    chunk_list = []
    for idx, doc in enumerate(retrieved_docs):
        meta = doc.metadata or {}
        source_path = meta.get("source", "Unknown")
        chunk_list.append({
            "chunk_index": idx,
            "text": doc.page_content,
            "source": os.path.basename(source_path) if source_path else "Unknown",
            "full_source": source_path,
            "page": meta.get("page", None)
        })

    formatted_context = format_docs(retrieved_docs)

    # 4. Generate answer and measure generation latency
    t_gen_start = time.perf_counter()
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": formatted_context,
        "chat_history": chat_history,
        "question": reformulated_question
    })
    generation_latency = round(time.perf_counter() - t_gen_start, 3)

    return {
        "question": question,
        "reformulated_question": reformulated_question,
        "answer": answer,
        "retrieved_chunks": chunk_list,
        "contexts": [c["text"] for c in chunk_list],
        "context_text": formatted_context,
        "retrieval_latency": retrieval_latency,
        "generation_latency": generation_latency,
        "total_latency": round(retrieval_latency + generation_latency, 3)
    }

chat_history = []
if __name__ == "__main__":
    while True:
        query = input('YOU: ')
        break_lst = ['exit', 'quit', '0']
        if query.lower() in break_lst:
            break
        res = rag_chain.invoke({
            "question": query,
            "chat_history": chat_history
        })
        print(f'\nAI:{res}\n')
