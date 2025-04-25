## RAG Q&A Conversation With PDF Including Chat History

# importing libraries
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableWithMessageHistory
import os 

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## set up Streamlit app

st.title("Converstaion RAG With PDF uploads and chat history")
st.write("Upload PDF's and chat with their content")

## input the Groq api key
api_key = st.text_input("Enter your Groq API Key:", type="password")

## check if grop api key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    ## chat interfaces

    session_id = st.text_input("Session ID", value="default_session")
    ## statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store = {}
    uploaded_files = st.file_uploader("Choose A PDF", accept_multiple_files=True)

    # Process uplaoded PDFs
    if uploaded_files:
        documents = []
        for uplaoded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uplaoded_file.getvalue())
                file_name = uplaoded_file.name
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        ## split and create emmbeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding = embeddings, persist_directory="db")
        retriever = vectorstore.as_retriever()

        ## create our prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question which can be understand "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."

        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        ## Answer question prompt

        system_prompt = (
                "You are an assitant for question-answering tasks. "
                "Use the following pieces of context to answer "
                "the question. If you don't know the answer, say that you"
                "don't know. Use three sentences maximum and keep the "
                "answer consise. "
                "\n\n"
                "{context}" ### this context will be replace by stuff documnet chain
            
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        ## creating chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        ## function for session history and retrive as basechatMessageHistory

        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]


        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("You: ")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config= {
                    "configurable": {
                        "session_id": session_id
                    }
                }, # constructs a key "abc123" in store.
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("chat history:", session_history.messages)
else:
    st.write("please enter the Groq api key")
