# main.py
import os
import shutil
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
from PyPDF2 import PdfReader

from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.docstore.document import Document

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# === Configuration & paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_ROOT = os.path.join(BASE_DIR, "chroma_store")
os.makedirs(CHROMA_ROOT, exist_ok=True)

# Reduce tokenizer parallelism logs/noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model / CPU configuration
MODEL_NAME = "google/flan-t5-small"   # smaller, more CPU-friendly
# MAX_NEW_TOKENS = 256
MAX_NEW_TOKENS = 128

# Thread pool for blocking work (PyPDF2, Chroma creation, embeddings)
executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static if folder exists (optional)
static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# In-memory session storage
sessions: Dict[str, dict] = {}     # sessions[sid] = {"status": "idle"|"processing"|"ready"|"error", "error": Optional[str]}
retrievers: Dict[str, object] = {} # retrievers[sid] = LangChain retriever

# === Load models ===
# Force CPU device (device = -1 for transformers pipeline)
device = -1

print("Loading tokenizer & model (this can take a while on first run)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=MAX_NEW_TOKENS,
)
llm = HuggingFacePipeline(pipeline=pipe)
print("Model loaded (CPU mode).")

# Embedding model (sentence-transformers; CPU-friendly)
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# === Utilities ===
def extract_text_from_pdf(fileobj) -> str:
    """
    Extract text from a file-like object (BytesIO or UploadFile.file).
    """
    reader = PdfReader(fileobj)
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text()
        except Exception:
            text = None
        if text:
            pages.append(text)
    return " ".join(pages)


def build_chroma_for_session(session_id: str, text: str) -> None:
    """
    Blocking function to create chunks, compute embeddings, and persist Chroma store for a session.
    Runs inside ThreadPoolExecutor.
    """
    try:
        print(f"[build] Building Chroma for session {session_id} — chunking text...")
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        docs = [Document(page_content=c) for c in chunks]

        persist_dir = os.path.join(CHROMA_ROOT, session_id)
        os.makedirs(persist_dir, exist_ok=True)

        print(f"[build] Creating Chroma vector store in {persist_dir} (may take time)...")
        vector_store = Chroma.from_documents(
            docs,
            embedding_function,
            persist_directory=persist_dir
        )
        retrievers[session_id] = vector_store.as_retriever(search_kwargs={"k": 3})
        sessions[session_id]["status"] = "ready"
        sessions[session_id]["error"] = None
        print(f"[build] Chroma build complete for session {session_id}.")
    except Exception as e:
        print(f"[build] Error while building Chroma for {session_id}: {e}")
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)
        # cleanup partially-created directory
        try:
            shutil.rmtree(os.path.join(CHROMA_ROOT, session_id))
        except Exception:
            pass
        raise


async def run_build_in_executor(session_id: str, text: str):
    """
    Coroutine to offload the blocking build function to the thread pool.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, build_chroma_for_session, session_id, text)


# === Routes ===
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/create_session")
async def create_session():
    sid = str(uuid.uuid4())
    sessions[sid] = {"status": "idle", "error": None}
    print(f"[session] Created session {sid}")
    return JSONResponse({"session_id": sid})


@app.post("/upload")
async def upload_pdf(
    session_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(None),
):
    """
    Upload a PDF tied to `session_id` (form field or header). The PDF is read and indexing runs in background.
    """
    sid = session_id or x_session_id
    if not sid:
        raise HTTPException(status_code=400, detail="Missing session_id (form field or x-session-id header).")

    if sid not in sessions:
        raise HTTPException(status_code=400, detail="Unknown session_id. Create a session first (/create_session).")

    # basic file validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    contents = await file.read()
    max_bytes = 40 * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(status_code=400, detail="File too large. Max 40 MB.")

    from io import BytesIO
    fileobj = BytesIO(contents)

    try:
        # extract text in thread (PyPDF2 is blocking)
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(executor, extract_text_from_pdf, fileobj)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {e}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in PDF (maybe scanned).")

    # schedule background build (non-blocking)
    sessions[sid]["status"] = "processing"
    sessions[sid]["error"] = None
    print(f"[upload] Scheduling Chroma build for session {sid} (background).")
    asyncio.create_task(run_build_in_executor(sid, text))

    return JSONResponse({"message": "Upload accepted. Processing in background.", "session_id": sid})


@app.get("/status")
async def status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Unknown session_id.")
    return JSONResponse({"session_id": session_id, **sessions[session_id]})


@app.post("/chat")
async def chat(
    session_id: Optional[str] = Form(None),
    x_session_id: Optional[str] = Header(None),
    message: str = Form(...),
):
    sid = session_id or x_session_id
    if not sid:
        raise HTTPException(status_code=400, detail="Missing session_id (form or x-session-id).")
    if sid not in sessions:
        raise HTTPException(status_code=400, detail="Unknown session_id.")
    if sessions[sid]["status"] != "ready":
        raise HTTPException(status_code=400, detail="Session not ready. Upload and wait for processing to finish.")

    retriever = retrievers.get(sid)
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever missing for session — try re-uploading.")

    try:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever)
        if len(message) > 2000:
            raise HTTPException(status_code=400, detail="Question too long.")
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(executor, qa_chain.run, message)
        return JSONResponse({"response": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while answering: {e}")


@app.post("/clear_session")
async def clear_session(session_id: Optional[str] = Form(None), request: Request = None):
    """
    Accepts session_id via form or JSON body {"session_id": "..."}.
    """
    sid = session_id
    if not sid and request is not None:
        try:
            body = await request.json()
            sid = body.get("session_id")
        except Exception:
            sid = None

    if not sid:
        raise HTTPException(status_code=400, detail="Missing session_id.")

    if sid in sessions:
        try:
            persist_dir = os.path.join(CHROMA_ROOT, sid)
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
        except Exception:
            pass
        sessions.pop(sid, None)
        retrievers.pop(sid, None)
        print(f"[clear] Cleared session {sid}")

    return JSONResponse({"message": "Cleared."})


if __name__ == "__main__":
    # Run directly: stable for local testing (no autoreload)
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
    # uvicorn.run("main:app", host="    