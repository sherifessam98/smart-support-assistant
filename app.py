import streamlit as st
from rag_pipeline.document_loader import load_text_file, chunk_text
import os


st.set_page_config(page_title="Smart Support Assistant") # Browser Tab title

st.title("📄Document Chunk Viewer")
#uploader widget
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file is not None:
    file_content = uploaded_file.read().decode("utf-8")
    st.success("File Loaded")
    chunks = chunk_text(file_content)
    st.subheader(f"Generated {len(chunks)} Chunks")
    #Looping through each chunk and displaying it in the UI
    for i,chunk in enumerate(chunks):
        st.markdown(f"**Chunk {i+1}:**") #chunk title
        st.write(chunk)
        st.markdown("------")




