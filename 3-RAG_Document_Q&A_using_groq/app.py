import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain


from dotenv import load_dotenv
load_dotenv()

## load the groq api key
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

## creating llm model
llm = ChatGroq(groq_api_key=groq_api_key, model_name= "Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(

    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>

    Question: {input}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("../research_papers") ## data indestions
        try:
            st.session_state.docs = st.session_state.loader.load()
            st.write("Number of docs loaded:", len(st.session_state.docs))
        except Exception as e:
            st.error(f"Error loading PDFs: {e}")

        #st.session_state.docs = st.session_state.loader.load() ## document loading

        # if not st.session_state.docs:
        #     st.error("No documents found. Please make sure the research_papers folder contains valid PDFs.")
        #     return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        if not st.session_state.final_documents:
            st.error("Text splitter produced no chunks. Check the PDF contents.")
            return
    
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
st.title("RAG Document Q&A Using Groq Lama3")
user_prompt = st.text_input("enter your query from research paper")

if st.button("Document Embeddings"):
    create_vector_embeddings()
    st.write("your vector database is ready")

import time

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("please generate document embeddings first by clicking the button")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrival = st.session_state.vectors.as_retreiver()
        retrival_chain = create_retrieval_chain(retrival, document_chain)

        start = time.process_time()
        response = retrival_chain.invoke({"input": user_prompt})
        print(f"Response time : {time.process_time() - start}")
        st.write(response['answer'])

        ## With streamlit expander

        with st.expander("Document similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------------------')