import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# FIX 1: Update FAISS import to langchain_community
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings # Also good practice for embeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline # Good practice for HuggingFacePipeline
import torch
import os

# Set page layout
st.set_page_config(page_title="Claude/Gemini Clone", layout="wide")
st.title("💬 Claude/Gemini Clone with Mistral + LangChain")

# Define a cache directory for Hugging Face models
# FIX 2: Define and create a cache directory in /tmp
CACHE_DIR = "/tmp/huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True) # Ensure the directory exists

# Load model + tokenizer
@st.cache_resource
def load_model():
    # FIX 3: Add cache_dir to AutoTokenizer.from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        use_fast=True,
        cache_dir=CACHE_DIR
    )
    # FIX 4: Add cache_dir to AutoModelForCausalLM.from_pretrained
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

# Load FAISS vector store
@st.cache_resource
def load_faiss():
    # It's good practice to also import HuggingFaceEmbeddings from langchain_community
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Ensure faiss_index directory exists and is accessible
    # You might need to ensure this directory and its contents are uploaded with your app
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
