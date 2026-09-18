from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict
import os
import shutil

from app.services.retrieval import rag_chain
from app.services.ingestion import process_pdf


app = FastAPI(title="Modular RAG API")


class QueryRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]]


@app.post("/query")
def query_documents(request: QueryRequest):

    try:

        # Convert frontend chat history
        # into LangChain message format
        format_his = []

        for mes in request.chat_history:

            sender = "human" if mes["role"] == "user" else "ai"

            format_his.append(
                (sender, mes["content"])
            )


        # Send question + chat history to RAG chain
        response = rag_chain.invoke({
            "question": request.question,
            "chat_history": format_his
        })


        return {
            "question": request.question,
            "answer": response
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):

    try:

        # Check file extension
        if not file.filename.lower().endswith(".pdf"):

            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed."
            )


        # Create data folder if it doesn't exist
        data_folder = os.path.join("app", "data")

        os.makedirs(
            data_folder,
            exist_ok=True
        )


        # Create path for uploaded PDF
        file_path = os.path.join(
            data_folder,
            file.filename
        )


        # Save uploaded PDF
        with open(file_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # Process the newly uploaded PDF
        chunks_added = process_pdf(file_path)


        return {
            "message": f"Successfully uploaded and processed {file.filename}",
            "chunks_added": chunks_added
        }


    except HTTPException:
        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )