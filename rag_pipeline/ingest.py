import os
import pickle

from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from rag_pipeline.document_loader import load_text_file, chunk_text
from rag_pipeline.rag_chain import vector_store, embeddings

PERSIST_PATH = "faiss_index"

def ingest_document(
        file_path: str,
        persist_path: str = PERSIST_PATH,
        chunk_size: int = 500,
        chunk_overlap: int = 100
) -> None:
    """
    Loads a document, chunks it,embeds it and saves the FAISS vector index and embeddings.
    :param file_path: Path to the document file
    :param persist_path: where FAISS index and embedding model are saved
    :param chunk_size: Number of character per chunk
    :param chunk_overlap: overlap between chunks to preserve context
    :return:
    """

    text = load_text_file(file_path)
    chunks: List[str] = chunk_text(text, chunk_size = chunk_size, overlap=chunk_overlap)
    embeddings = OpenAIEmbeddings
    #build FAISS INDEX
    vector_store = FAISS.from_texts(chunks, embeddings=embeddings)
    #
    os.makedirs(persist_path,exist_ok=True)
    vector_store.save_local(persist_path)
    with open(os.path.join(persist_path,"embeddings.pkl"),"wb") as f:
        pickle.dump(embeddings,f)
    print("✅ Ingestion complete.")

def load_vector_store(persist_path: str = PERSIST_PATH) -> FAISS:
    """
    Load the FAISS index and its embedded model from disk.
    Returns a FAISS vector store ready for similarity search
    :param persist_path:
    :return:
    """
    with open(os.path.join(persist_path,"embeddings.pkl"),"rb") as f:
        embeddings = pickle.load(f)
    vector_store = FAISS.load_local(persist_path,embeddings)
    return vector_store



