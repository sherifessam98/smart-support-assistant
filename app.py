import streamlit as st
import os

from openai import vector_stores

from rag_pipeline.ingest import ingest_document, load_vector_store
from rag_pipeline.rag_chain import ask_question


DATA_DIR = "data"
FAISS_DIR = "faiss_index"


st.set_page_config(page_title="Smart Support Assistant") # Browser Tab title
st.title("🤖 Smart Support Assistant")
st.write("Ask questions about your document.")

#uploader widget
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

# Uploading file
if uploaded_file:
    os.makedirs(DATA_DIR,exist_ok=True)
    file_path = os.path.join(DATA_DIR,uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved'{uploaded_file.name}'")

    # ingest into FAISS
    if not os.path.exists(FAISS_DIR):
        with st.spinner("Indexing document..."):
            ingest_document(file_path , persist_path=FAISS_DIR)
        st.success("Document Indexed, you can now ask your questions.")
    else:
        st.info("Using existing index. Re-upload to re-index.")

    # Question Retrieval
    query = st.text_input("Ask a question about your document:")
    if query:
        with st.spinner("Generating Answer"):
            # Loading vector store
            answer, sources = ask_question(query,
                                           vectorestore_dir=FAISS_DIR)
        st.markdown("### ✅ Answer")
        st.write(answer)




