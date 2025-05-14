import streamlit as st
from rag_pipeline.rag_chain import ask_question
from ingest import save_uploaded_file, process_file


st.set_page_config(page_title="Smart Support Assistant") # Browser Tab title

st.title("🤖 Smart Support Assistant")
st.write("Ask questions about your document.")

#uploader widget
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])


if uploaded_file:
    with st.spinner("Processing file and building index..."):
        file_path = save_uploaded_file(uploaded_file)
        process_file(file_path)






