import streamlit as st
import os 
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

## laod the groq api key
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')    
llm = ChatGroq(groq_api_key=groq_api_key, model_name= "Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions on the provided context only.
    Please provide the most accurate responce based on the question

    <context>
    {context}
    <context>
    Question: {input}



    """
)
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("/research_papers") ## data ingestion step
        st.session_state.docs = st.session_state.loader.load() ## document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
st.title("RAG document Q&A With Groq And Lama3")


user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embeddings"):
    create_vector_embeddings()
    st.write("Vector Database is Ready")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retreiver =st.session_state.vectors.as_retriever()
    retreiver_chain = create_retrieval_chain(retreiver, document_chain)

    start = time.process_time()
    response = retreiver_chain.invoke({'input': user_prompt})
    print(f"Response time : {time.process_time()-start}")

    st.write(response['answer'])

    # with a streamlit expander
    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_context)
            st.write('-------------------------')




