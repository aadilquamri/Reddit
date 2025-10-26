# app.py
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangChain / Groq
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# -----------------------
# Load environment variables
# -----------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "your_groq_api_key")
os.environ["LANGCHAIN_API_KEY "] = os.getenv("Langchain_api_key", "your_langchain_api_key")

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("rag_dataset.csv")

# Combine columns into single text field
df["combined_text"] = (
    df["selftext"].fillna("") + " "
    + df["title"].fillna("") + " "
    + df["comments"].fillna("")
)
docs = [Document(page_content=text) for text in df["combined_text"].tolist()]

# -----------------------
# Split text into chunks
# -----------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(docs)

# -----------------------
# Embeddings + Vector DB
# -----------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# -----------------------
# Chat model (Groq LLM)
# -----------------------
chat = ChatGroq(model="deepseek-r1-distill-llama-70b")


# -----------------------
# RetrievalQA pipeline
# -----------------------
qa = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    return_source_documents=True
)

def get_response(user_input: str) -> str:
    """Retrieve context and generate answer from Groq"""
    result = qa.invoke(user_input)
    return result["result"]

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("Ask me anything about Startups")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your input here......"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot response
    response = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    
