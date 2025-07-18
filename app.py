import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
import torch
import os

# Set page layout
st.set_page_config(page_title="Claude/Gemini Clone", layout="wide")
st.title("💬 Claude/Gemini Clone with Mistral + LangChain")

# Cache directory for HuggingFace
CACHE_DIR = "/tmp/huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

HF_TOKEN = st.secrets.get("HF_TOKEN")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        use_fast=True,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        token=HF_TOKEN
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

# Upload PDF or TXT file
uploaded_file = st.file_uploader("📁 Upload a PDF or TXT file", type=["pdf", "txt"])

@st.cache_resource
#dfrom langchain_community.document_loaders import PyPDFLoader, TextLoader

def build_vectorstore(uploaded_file):
    # Save uploaded file to disk
    file_path = os.path.join("/tmp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Choose loader based on file type
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a .pdf or .txt file.")

    docs = loader.load()
    
    # Split into chunks
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(docs)

    # Embed & store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
    

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# MAIN
if uploaded_file is not None:
    with st.spinner("Building vector store from uploaded document..."):
        vectorstore = build_vectorstore(uploaded_file)
else:
    with st.spinner("Loading existing FAISS vector store..."):
        vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever()
llm = load_model()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=False
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.chat_input("Ask your question about the document...")

if user_question:
    with st.spinner("Thinking..."):
        response = qa_chain({"question": user_question})
        answer = response["answer"]

    st.session_state.chat_history.append((user_question, answer))

for q, a in st.session_state.chat_history:
    st.chat_message("user").markdown(f"**You:** {q}")
    st.chat_message("assistant").markdown(f"**AI:** {a}")
