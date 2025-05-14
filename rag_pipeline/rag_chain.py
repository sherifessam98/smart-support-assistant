from langchain.chains import RetreivalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from typing import List, Tuple
from rag_pipeline.ingest import load_vector_store
import pickle


def ask_question(query:str,
                 vectorestore_dir:str = "faiss_index",
                 k:int = 3,
                 model_name: str = "gpt-3.5-turbo",
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
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    qa_chain = RetreivalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain(query)
    return result["result"], result["source_documents"]