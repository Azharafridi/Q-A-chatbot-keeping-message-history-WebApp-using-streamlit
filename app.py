import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os 
from dotenv import load_dotenv
load_dotenv()

## langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = "Q-A-chatbot-With OpenAI"

## Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a helpful assistant. Please respond to the user's queries."),
    ("human", "Question: {question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model = llm)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question": question})
    return answer

## title of the app
st.title("Enhanced Q&A Chatbot With OpenAI")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

## drop down t select various Open AI models

llm = st.sidebar.selectbox("Select an OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

## adjusting response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## main interface for user input
st.text_input("Go ahead and ask any question:")
user_input = st.text_input("You: ")

if user_input: 
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)

else:
    st.write("Please enter a question.")