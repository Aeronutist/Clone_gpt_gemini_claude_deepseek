
# 📘 Claude/Gemini Clone with Mistral, LangChain, and FAISS

A full-stack conversational AI app that mimics the capabilities of tools like **Claude**, **Gemini**, or **ChatGPT** – built entirely using open-source LLMs, LangChain, FAISS vector search, and Streamlit. This project answers questions about uploaded documents using semantic search and a conversational memory interface.

---

## 🚀 Demo

🔗 [Live App on Streamlit]( https://clonegptgeminiclaudedeepseek-3d3t2mr9rvvcfdbp7ljspj.streamlit.app/)

---

## 📂 Features

- ✅ Load any **PDF or TXT** document  
- ✅ Chunk and embed the content with **Sentence Transformers**  
- ✅ Create a searchable **FAISS vector store**  
- ✅ Use **Mistral-7B-Instruct** for high-quality answers  
- ✅ Maintain **chat history** using LangChain Memory  
- ✅ Clean and responsive **chat interface (like Claude or Gemini)**  
- ✅ Fully deployable on **Streamlit Cloud**  

---

## 🛠️ Tech Stack

| Component        | Technology                            |
|------------------|----------------------------------------|
| LLM              | Mistral-7B-Instruct (Hugging Face)     |
| Vector Search    | FAISS                                  |
| Embeddings       | all-MiniLM-L6-v2 (Sentence Transformers)|
| Framework        | LangChain                              |
| UI               | Streamlit                              |
| Deployment       | Streamlit Cloud / Local                |

---

## 📁 Folder Structure

```
📁 ClaudeGeminiClone/
├── app.py                  # Streamlit frontend + backend
├── faiss_index/            # FAISS vector store (generated after chunking)
├── requirements.txt
└── README.md
```

---

## ✅ Installation

### 1. Clone this repository

```bash
git clone https://github.com/your-username/ClaudeGeminiClone.git
cd ClaudeGeminiClone
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run locally

```bash
streamlit run app.py
```

> ⚠️ This project is heavy. It's recommended to use **Google Colab** or a **GPU-based cloud instance** for loading Mistral-7B.

---

## 🔄 How It Works

1. User uploads a document (PDF or text)
2. Text is split into chunks
3. Chunks are embedded and indexed via FAISS
4. Mistral generates responses using LangChain’s `ConversationalRetrievalChain`
5. Chat memory stores previous Q&A interactions

---

## 💡 Use Cases

- Legal document summarization  
- Academic paper Q&A  
- Book or report assistant  
- Research assistant chatbot  
- Product documentation chatbot  

---

## 📦 requirements.txt

```
transformers
torch
sentence-transformers
faiss-cpu
streamlit
langchain
```

---

## 👨‍💻 Author

> Built with passion and ambition to become a **top-level data scientist**.

**Mentored by:** GPT – AI Research Assistant  
**Developer:** [Your Name](https://github.com/your-profile)

---

## 🏆 Project 14 – Done  
Next: **Project 15 – Research-Level QA with LangGraph or RAG-as-Agent**
