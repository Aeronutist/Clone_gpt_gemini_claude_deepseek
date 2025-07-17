import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
import torch
import os

# Set page layout
st.set_page_config(page_title="Claude/Gemini Clone", layout="wide")
st.title("💬 Claude/Gemini Clone with Mistral + LangChain")

# Load model + tokenizer (you can cache if needed)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

# Load FAISS vector store
@st.cache_resource
def load_faiss():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize components
llm = load_model()
retriever = load_faiss().as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=False
)

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.chat_input("Ask your question about the document...")

if user_question:
    with st.spinner("Thinking..."):
        response = qa_chain({"question": user_question})
        answer = response["answer"]

    st.session_state.chat_history.append((user_question, answer))

# Display chat history
for i, (q, a) in enumerate(st.session_state.chat_history):
    st.chat_message("user").markdown(f"**You:** {q}")
    st.chat_message("assistant").markdown(f"**AI:** {a}")
