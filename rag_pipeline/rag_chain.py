from langchain.chains import RetreivalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
import pickle


#Loading the same embedding model used for saving the FAISS index
with open("embedding.pkl","rb") as f:
    embeddings = pickle.load(f)

vector_store = FAISS.loadl_local("faiss_index",embeddings)
#Creating a retriever for which is used for searching for relevant chunks
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})
#Setting up the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature = 0)

qa_chain = RetreivalQA.from_chain_type(
    llm=llm,
    chain_type = "stuff",
    retriever = retriever,
    return_source_documents = True
)

def ask_question(query:str):
    response = qa_chain(query)
    return response["result"],response["source_documents"]

