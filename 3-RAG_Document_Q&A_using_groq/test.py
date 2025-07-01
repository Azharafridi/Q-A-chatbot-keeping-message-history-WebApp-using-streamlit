import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Go one directory up and into 'research_papers'
pdf_dir = os.path.abspath(os.path.join(os.getcwd(), "../research_papers"))

st.write("Trying to load PDFs from:", pdf_dir)
st.write("Directory exists:", os.path.exists(pdf_dir))
st.write("Files in directory:", os.listdir(pdf_dir) if os.path.exists(pdf_dir) else "Directory not found")

st.session_state.loader = PyPDFDirectoryLoader(pdf_dir)

try:
    st.session_state.docs = st.session_state.loader.load()
    st.write(f"Loaded {len(st.session_state.docs)} documents.")
except Exception as e:
    st.error(f"Error loading PDFs: {e}")
