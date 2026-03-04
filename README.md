# RAG People Directory API

Ask questions about people from a CSV file using Groq LLM + FAISS.

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Create `.env` file**
```
GROQ_API_KEY=gsk_your_key_here
```
Get free key at → https://console.groq.com

**3. Run**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**4. Open Swagger UI**
```
http://localhost:8000/docs
```

---

## CSV Format

Your CSV must have these two columns:

| Name   | Description                        |
|--------|------------------------------------|
| Sakthi | Sakthi is a software engineer...   |
| Arun   | Arun is a data scientist...        |

---

## API Endpoints

| Method | Endpoint       | Description                        |
|--------|----------------|------------------------------------|
| GET    | `/health`      | Check if CSV is loaded and ready   |
| POST   | `/upload-csv`  | Upload CSV file                    |
| POST   | `/ask`         | Ask a question (request body)      |
| GET    | `/answer`      | Ask a question (URL param)         |
| GET    | `/people`      | List all people from CSV           |

---

## Usage

**Upload CSV**
```bash
curl -X POST http://localhost:8000/upload-csv \
  -F "file=@test.csv"
```

**Ask a question (POST)**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is Sakthi?"}'
```

**Ask a question (GET)**
```
http://localhost:8000/answer?question=Who is Sakthi?
```

---

## Deploy on Render

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variable: `GROQ_API_KEY = gsk_your_key`
7. Deploy!

---

## Tech Stack

- **FastAPI** — web framework
- **Groq** — free LLM API (LLaMA 3.3 70B)
- **FAISS** — vector similarity search
- **LangChain** — document processing
- **HuggingFace** — embedding model (all-MiniLM-L6-v2)