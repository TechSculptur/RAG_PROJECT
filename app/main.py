from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict
import os
import shutil
from fastapi.responses import FileResponse
from app.services.retrieval import rag_chain
from app.services.ingestion import process_pdf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI(title="Modular RAG API")
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


@app.post("/query")
def query_documents(request: QueryRequest):

    try:
        format_his = []
        for mes in request.chat_history:
            sender = "human" if mes["role"] == "user" else "ai"
            format_his.append((sender, mes["content"])            )

        response = rag_chain.invoke({
            "question": request.question,
            "chat_history": format_his
        })


        return {
            "question": request.question,
            "answer": response
        }


    except Exception as e:

        raise HTTPException(status_code=500,detail=str(e))


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):

    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed."
            )

        data_folder = os.path.join("app", "data")
        os.makedirs(data_folder,exist_ok=True)
        file_path = os.path.join(
            data_folder,
            file.filename
        )
        with open(file_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

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