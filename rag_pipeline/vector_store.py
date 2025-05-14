from langchain.embeddings import OpenAIEmbeddings #for turning text into vectors
from langchain.vectorstores import FAISS #the  vector database which will be used
import os
import pickle #for saving and loading the FAISS index


def create_vector_store(chunks, presist_path="faiss_index"):
    """
    Creates a FAISS vector store from documents' chunks

    """

    embeddings = OpenAIEmbeddings()
    # a vector store from the documents chunks and their embeddings
    vector_store = FAISS.from_documents(chunks,embeddings)

    vector_store.save_local(presist_path)
    #Saving the embedding object
    with open(os.path.join(presist_path,"embeddings.pkl"),"wb") as f:
        pickle.dump(embeddings,f)

    print(f"✅ Vector store saved to:{presist_path}")

    return vector_store

def load_vector_store(presist_path = "faiss_index"):
    """
    load the FAISS vectore store for later use

    """
    #loading the saved OpenAI embedding model
    with open(os.path.join(presist_path,"embedding.pkl"),"rb") as f:
        embeddings = pickle.load(f)

    #loading the FAISS index from disk using the same embedding model
    vector_store = FAISS.load_local(presist_path,embeddings)
    return vector_store



