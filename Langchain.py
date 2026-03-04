# ============================================================
#  RAG PIPELINE - FastAPI  (Render-Ready)
#  Routes:
#    POST /upload-csv  -> upload CSV, replaces old one, re-indexes
#    POST /ask         -> send question, get LLM answer
#    GET  /answer      -> get answer via query param
#    GET  /health      -> system status
#    GET  /people      -> list all loaded people
# ============================================================
#
#  INSTALL:
#  pip install fastapi uvicorn langchain langchain-core langchain-community
#  pip install langchain-text-splitters
#  pip install langchain-huggingface "huggingface_hub>=0.33.4,<1.0.0"
#  pip install sentence-transformers faiss-cpu python-dotenv pandas groq
#
#  .env file:
#  GROQ_API_KEY=your_groq_api_key_here
#
#  RUN locally:
#  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ============================================================

import os
import shutil
import pandas as pd
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


load_dotenv()

GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
VECTOR_STORE_PATH = "vector_store"
CSV_PATH          = "data/uploaded.csv"   # always this one fixed path
GROQ_MODEL        = "llama-3.3-70b-versatile"

# GLOBAL STATE

class AppState:
    vector_store  : Optional[FAISS]                 = None
    embeddings    : Optional[HuggingFaceEmbeddings] = None
    client        : Optional[Groq]                  = None
    csv_loaded    : bool                            = False
    csv_filename  : str                             = ""
    total_rows    : int                             = 0
    total_vectors : int                             = 0

state = AppState()



def load_csv(file_path: str):
    df = pd.read_csv(file_path)
    df.dropna(how="all", inplace=True)
    df.fillna("N/A", inplace=True)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    required_cols = {"Name", "Description"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must have columns: {required_cols}. Found: {list(df.columns)}"
        )

    documents = []
    for idx, row in df.iterrows():
        content = f"Name: {row['Name']}\nDescription: {row['Description']}"
        doc = Document(
            page_content=content,
            metadata={
                "name"      : row["Name"],
                "row_index" : int(idx),
                "source"    : file_path,
            },
        )
        documents.append(doc)

    return documents, df


def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks, embeddings):
    # Always wipe old FAISS so previous CSV data never leaks into new index
    shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store


def load_vector_store_from_disk(embeddings):
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_groq_client():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    return Groq(api_key=GROQ_API_KEY)


def ask_llm(question: str, context: str, client) -> str:
    prompt = f"""You are a helpful assistant with access to a people directory.
Answer the question using ONLY the context provided below.
If the answer is not found in the context, say "I don't have that information."
Be clear, concise, and accurate.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role"   : "system",
                "content": "You are a helpful assistant that answers questions based only on the provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 55)
    print("   RAG PIPELINE  —  FastAPI / Render Mode")
    print(f"   LLM  : {GROQ_MODEL}")
    print(f"   Embed: all-MiniLM-L6-v2")
    print("=" * 55)

    # Load embedding model once
    print("Loading embedding model...")
    state.embeddings = load_embedding_model()
    print(" Embedding model ready")

    # Load Groq client
    try:
        state.client = load_groq_client()
        print(" Groq client ready")
    except RuntimeError as e:
        print(f"  Groq warning: {e}")

    # If FAISS index already on disk (Render restart), reload it
    faiss_index_file = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    if os.path.exists(faiss_index_file):
        print("Found existing FAISS index — loading from disk...")
        state.vector_store  = load_vector_store_from_disk(state.embeddings)
        state.csv_loaded    = True
        state.total_vectors = state.vector_store.index.ntotal
        print(f" Loaded {state.total_vectors} vectors")
    else:
        print("  No FAISS index — upload a CSV via POST /upload-csv")

    print("\n Server ready!  Docs → /docs\n")
    yield
    print(" Shutting down...")



app = FastAPI(
    title       = "RAG People Directory API",
    description = """
## How to use

**Step 1** → `POST /upload-csv` — upload your CSV (Name + Description columns).  
Each new upload **replaces** the previous file and rebuilds the vector index.

**Step 2** → `POST /ask` — send `{"question": "Who is Sakthi?"}` → get LLM answer.

**Step 3** → `GET /answer?question=Who is Sakthi?` — same via URL (browser friendly).

**Check** → `GET /health` — shows if CSV is loaded and vectors indexed.
    """,
    version  = "1.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# SCHEMAS
# ============================================================
class AskRequest(BaseModel):
    question : str
    k        : int = 3

    class Config:
        json_schema_extra = {
            "example": {"question": "Who is Sakthi and what does he do?", "k": 3}
        }

class AskResponse(BaseModel):
    question    : str
    answer      : str
    sources     : list[str]
    chunks_used : int


# ============================================================
# GUARD — called before any query endpoint
# ============================================================
def require_csv():
    if not state.csv_loaded or state.vector_store is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No CSV uploaded yet. "
                "Upload one first using POST /upload-csv"
            ),
        )
    if not state.client:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not set. Add it to .env or Render env vars.",
        )


# ============================================================
# ROUTES
# ============================================================

# ── GET /health ──────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Check if the system is ready to answer questions."""
    return {
        "status"          : "ready" if state.csv_loaded else "waiting_for_csv",
        "csv_loaded"      : state.csv_loaded,
        "csv_filename"    : state.csv_filename,
        "total_rows"      : state.total_rows,
        "vectors_indexed" : state.total_vectors,
        "groq_model"      : GROQ_MODEL,
        "embed_model"     : "all-MiniLM-L6-v2",
        "message"         : (
            " Ready! Use POST /ask or GET /answer?question=..."
            if state.csv_loaded
            else "  Upload a CSV via POST /upload-csv to get started."
        ),
    }


# ── POST /upload-csv ─────────────────────────────────────────
@app.post("/upload-csv", tags=["Data"])
async def upload_csv(file: UploadFile = File(...)):
    """
    **Upload a CSV file (Name + Description columns required).**

    - Deletes the previous CSV and FAISS index completely
    - Parses and re-indexes the new file
    - After upload, POST /ask uses only the new data
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    # Save — overwrites previous uploaded.csv
    os.makedirs("data", exist_ok=True)
    contents = await file.read()
    with open(CSV_PATH, "wb") as f:
        f.write(contents)

    # Parse and validate
    try:
        documents, df = load_csv(CSV_PATH)
    except ValueError as e:
        os.remove(CSV_PATH)          
        state.csv_loaded = False
        raise HTTPException(status_code=422, detail=str(e))

    chunks             = split_into_chunks(documents)
    state.vector_store = build_vector_store(chunks, state.embeddings)

    # Update global state
    state.csv_loaded    = True
    state.csv_filename  = file.filename
    state.total_rows    = len(df)
    state.total_vectors = state.vector_store.index.ntotal

    return {
        "message"         : f" '{file.filename}' uploaded and indexed.",
        "rows_loaded"     : state.total_rows,
        "chunks_created"  : len(chunks),
        "vectors_indexed" : state.total_vectors,
        "people_loaded"   : df["Name"].tolist(),
        "next_step"       : "Now POST /ask or GET /answer?question=... to query.",
    }


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(body: AskRequest):
    """
    **POST your question — get an answer from Groq LLM.**

    Request body:
    ```json
    { "question": "Who is Sakthi?", "k": 3 }
    ```
    """
    require_csv()

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # FAISS similarity search
    raw_docs = state.vector_store.similarity_search(body.question, k=body.k)
    context  = "\n\n".join(
        f"[Match {i+1} – {d.metadata.get('name','?')}]\n{d.page_content}"
        for i, d in enumerate(raw_docs)
    )

    # Groq LLM answer
    answer  = ask_llm(body.question, context, state.client)
    sources = list(dict.fromkeys(d.metadata.get("name", "Unknown") for d in raw_docs))

    return AskResponse(
        question    = body.question,
        answer      = answer,
        sources     = sources,
        chunks_used = len(raw_docs),
    )


@app.get("/answer", response_model=AskResponse, tags=["RAG"])
def get_answer(question: str, k: int = 3):
    """
    **GET your answer via URL query param.**

    Example:
    ```
    GET /answer?question=Who is Sakthi?
    ```
    Same RAG pipeline as POST /ask. Easier for browser testing.
    """
    require_csv()

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    raw_docs = state.vector_store.similarity_search(question, k=k)
    context  = "\n\n".join(
        f"[Match {i+1} – {d.metadata.get('name','?')}]\n{d.page_content}"
        for i, d in enumerate(raw_docs)
    )
    answer  = ask_llm(question, context, state.client)
    sources = list(dict.fromkeys(d.metadata.get("name", "Unknown") for d in raw_docs))

    return AskResponse(
        question    = question,
        answer      = answer,
        sources     = sources,
        chunks_used = len(raw_docs),
    )


@app.get("/people", tags=["Data"])
def list_people():
    """List all people currently loaded from the CSV."""
    if not os.path.exists(CSV_PATH):
        raise HTTPException(
            status_code=404,
            detail="No CSV uploaded yet. Use POST /upload-csv first.",
        )
    df = pd.read_csv(CSV_PATH).fillna("N/A")
    return {
        "total"  : len(df),
        "people" : df[["Name", "Description"]].to_dict(orient="records"),
    }