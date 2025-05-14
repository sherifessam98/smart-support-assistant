from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from typing import List, Tuple
import torch
from transformers import pipeline
from rag_pipeline.ingest import load_vector_store
import pickle


def ask_question(query:str,
                 vectorestore_dir:str = "faiss_index",
                 k:int = 3,
                 model_name: str = "google/flan-t5-base",
                 temperature: float = 0.0
                 ) -> Tuple[str, List]:
    # Load FAISS index
    vector_store = load_vector_store(persist_path=vectorestore_dir)

    # Set up retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    # Initialize GPT
    # set up a HuggingFace text-generation pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
        max_length=512
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain(query)
    return result["result"], result["source_documents"]