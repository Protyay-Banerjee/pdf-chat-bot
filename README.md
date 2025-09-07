# 📄 PDF Chatbot

AI-powered chatbot that lets you **upload PDFs** and **ask questions** about their content.  
Built with **FastAPI** (backend), a small local **Flan-T5** model for generation, **sentence-transformers** for embeddings, and **Chroma** for vector storage. This repository is optimized for **CPU-only** usage and per-user (per-session) indices.

---

## 🔥 Highlights

- Upload a PDF and index it per session (`./chroma_store/{session_id}`)  
- Ask natural-language questions about the PDF content via a chat UI  
- CPU-friendly default configuration (`google/flan-t5-small`)  
- Background indexing and status polling in the frontend  
- Simple, single-folder project structure for local development

---

## 🗂 Project structure

```
pdfchatbot/
├── main.py # FastAPI backend
├── templates/
│ └── index.html # Frontend UI
├── chroma_store/ # Created at runtime (per-session)
├── requirements.txt # Dependencies
├── README.md # This file
└── .gitignore
```
## 🚀 Quick start (local, CPU)

> Tested on Python 3.10+ (Python 3.12 is supported too). Work in a virtual environment.

1. Clone (if you haven't already)

```bash
git clone git@github.com:Protyay-Banerjee/pdf-chat-bot.git
cd pdf-chat-bot
```

```## Create & activate a virtual environment
python -m venv venv
```

```source venv/bin/activate
# macOS / Linux
```

```# Windows (PowerShell)
venv\Scripts\Activate.ps1
```
```# Windows (cmd)
venv\Scripts\activate
```

```# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

# Option A: run via python (main includes uvicorn run block)
python main.py

```# Option B: run via uvicorn
uvicorn main:app --host 127.0.0.1 --port 8000
```

# Open your browser at: 
http://127.0.0.1:8000

## 🧩 How it works (overview)

- Session: Frontend requests a server-generated session_id. The session is stored in-memory on the server.

- Upload & Index: User uploads a PDF tied to that session. Server extracts text (PyPDF2), splits into chunks, computes embeddings (sentence-transformers), and persists a Chroma vector store in ./chroma_store/{session_id}. Indexing runs in background.

- Chat: Frontend polls /status until the session becomes ready. Chat requests call /chat with the same session_id. The server builds a RetrievalQA chain that retrieves top-k chunks and uses the local LLM to answer.

## ⚙️ Key implementation choices & config

Model: google/flan-t5-small (CPU-optimized choice). Change in main.py if you have more RAM/GPU.
Embeddings: sentence-transformers/all-MiniLM-L6-v2
Vector DB: Chroma, persisted per session in chroma_store/
QA chain: RetrievalQA using map_reduce (safer on CPU for long docs)
Chunking: Conservative defaults: chunk_size=500, chunk_overlap=50
You can adjust chunk_size, chunk_overlap, and retriever search_kwargs in main.py to tune memory vs. quality.

## 🧪 Quick troubleshooting
- 500 / allocation errors → reduce chunk size / reduce k / switch to map_reduce. Delete old chroma_store/ session folders when experimenting.
- 400 "Missing session_id" → frontend must use the server-generated session_id (create a session first).
- SSH push issues to GitHub → if Permission denied (publickey) appears, add your SSH public key to GitHub or use HTTPS remote.

## 📄 License
This project is provided under the MIT License — see LICENSE for details.

## ❤️ Acknowledgements
Built with FastAPI, Hugging Face Transformers, LangChain, and Chroma.

Inspired by open-source PDF→QA examples and local LLM workflows.
