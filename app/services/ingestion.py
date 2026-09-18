from dotenv import load_dotenv
load_dotenv()

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pypdf as pdf


def load_pdf(path: str) -> list[Document]:
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

embeddings = HuggingFaceEmbeddings()

def process_pdf(file_path: str):
    docs = load_pdf(file_path)
    chunks = splitter.split_documents(docs)
    if not chunks:
        print("No content found in PDF.")
        return 0
    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    vector_store.add_documents(chunks)
    print(f"Successfully processed: {file_path}")
    print(f"Added {len(chunks)} chunks to ChromaDB.")
    return len(chunks)
if __name__ == "__main__":
    pass