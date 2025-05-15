# helper

from langchain.embeddings import OpenAIEmbeddings #for turning text into vectors
from langchain.vectorstores import FAISS #the  vector database which will be used
import os
import pickle #for saving and loading the FAISS index


def create_vector_store(chunks, persist_path="faiss_index"):
    """
    Creates a FAISS vector store from documents' chunks

    """

    embeddings = OpenAIEmbeddings()
    # a vector store from the documents chunks and their embeddings
    vector_store = FAISS.from_documents(chunks,embeddings)
    os.makedirs(persist_path, exist_ok=True)
    # saving the faiss index
    vector_store.save_local(persist_path)
    #Saving the embedding object
    with open(os.path.join(persist_path,"embeddings.pkl"),"wb") as f:
        pickle.dump(embeddings,f)

    print(f"✅ Vector store saved to:{persist_path}")

    return vector_store

def load_vector_store(persist_path = "faiss_index"):
    """
    load the FAISS vector store for later use

    """
    #loading the saved OpenAI embedding model
    with open(os.path.join(persist_path,"embedding.pkl"),"rb") as f:
        embeddings = pickle.load(f)

    #loading the FAISS index from disk using the same embedding model
    vector_store = FAISS.load_local(persist_path,embeddings)
    return vector_store



