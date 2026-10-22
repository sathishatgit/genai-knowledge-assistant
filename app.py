"""AGUI Web Server for Intelligent GenAI Knowledge Assistant.

Provides interactive REST endpoints and serves the live web interface.
Run with: python app.py
"""

import sys
import warnings
from pathlib import Path

# Suppress deprecation notices during UI server run
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.config import GEMINI_API_KEY, JINA_API_KEY, EXA_API_KEY
from src.rag import KnowledgeAssistant
from src.bm25_search import BM25SearchEngine
from src.vector_store import ChromaVectorStore
from ingest import run_ingestion

app = FastAPI(title="AGUI - GenAI Interactive Agent")

STATIC_DIR = PROJECT_ROOT / "src" / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Initialize singletons
bm25_engine = BM25SearchEngine()
bm25_engine.load_index()
vector_store = ChromaVectorStore()
assistant = KnowledgeAssistant(bm25_engine=bm25_engine, vector_store=vector_store)


class QueryRequest(BaseModel):
    query: str
    route: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the interactive single-page web UI."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(INDEX_HTML)


from src import config
from src import llm

class ProviderSelectRequest(BaseModel):
    provider: str
    model: Optional[str] = None


@app.get("/api/status")
async def get_status():
    """Return live system component status, active provider, and document counts."""
    providers = llm.get_available_providers()
    active_prov = config.LLM_PROVIDER
    active_info = providers.get(active_prov, {})
    return {
        "gemini": bool(config.GEMINI_API_KEY and not config.GEMINI_API_KEY.startswith("your_")),
        "jina": bool(config.JINA_API_KEY and not config.JINA_API_KEY.startswith("your_")),
        "exa": bool(config.EXA_API_KEY and not config.EXA_API_KEY.startswith("your_")),
        "active_provider": active_prov,
        "active_model": active_info.get("model", ""),
        "active_ready": active_info.get("ready", False),
        "providers": providers,
        "chroma_count": vector_store.count(),
        "bm25_count": len(bm25_engine.chunks) if bm25_engine.chunks else 0,
    }


@app.get("/api/providers")
async def get_providers():
    """Return all supported LLM providers and the currently active one."""
    return {
        "active_provider": config.LLM_PROVIDER,
        "providers": llm.get_available_providers()
    }


@app.post("/api/providers/select")
async def select_provider(req: ProviderSelectRequest):
    """Switch active LLM provider dynamically at runtime."""
    prov = req.provider.strip().lower()
    if prov not in llm.SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider: {prov}. Supported: {list(llm.SUPPORTED_PROVIDERS)}"
        )
    config.LLM_PROVIDER = prov
    if req.model:
        if prov == "gemini":
            config.GEMINI_MODEL = req.model.strip()
        elif prov == "openai":
            config.OPENAI_MODEL = req.model.strip()
        elif prov == "nvidia":
            config.NVIDIA_MODEL = req.model.strip()
        elif prov == "ollama":
            config.OLLAMA_MODEL = req.model.strip()

    providers = llm.get_available_providers()
    return {
        "success": True,
        "active_provider": config.LLM_PROVIDER,
        "model": providers.get(config.LLM_PROVIDER, {}).get("model", ""),
        "ready": providers.get(config.LLM_PROVIDER, {}).get("ready", False)
    }


@app.get("/api/documents")
async def list_documents():
    """List all documents currently in the local knowledge base directory."""
    from src.config import DOCUMENTS_DIR
    if not DOCUMENTS_DIR.exists():
        return []

    docs = []
    for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        try:
            stat = file_path.stat()
            content = file_path.read_text(encoding="utf-8")
            docs.append({
                "filename": file_path.name,
                "size_bytes": stat.st_size,
                "char_count": len(content),
                "preview": content[:120].strip() + ("..." if len(content) > 120 else ""),
                "content": content
            })
        except Exception:
            pass
    return docs


class DirectTextDocumentRequest(BaseModel):
    title: str
    content: str


@app.post("/api/documents/text")
async def add_document_from_text(req: DirectTextDocumentRequest):
    """Add a new document by directly providing a title/filename and paragraph content."""
    from src.config import DOCUMENTS_DIR
    if not req.title.strip() or not req.content.strip():
        raise HTTPException(status_code=400, detail="Title and content are required.")

    # Sanitize title for filename
    clean_title = "".join(c for c in req.title.strip() if c.isalnum() or c in ("_", "-", " ")).strip()
    clean_title = clean_title.replace(" ", "_")
    if not clean_title.endswith(".txt"):
        clean_title += ".txt"

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = DOCUMENTS_DIR / clean_title

    try:
        target_path.write_text(req.content.strip(), encoding="utf-8")
        # Automatically re-index knowledge base
        run_ingestion(reset_db=True)
        bm25_engine.load_index()
        return {
            "success": True,
            "message": f"Document '{clean_title}' saved and indexed successfully!",
            "filename": clean_title,
            "chroma_count": vector_store.count(),
            "bm25_count": len(bm25_engine.chunks) if bm25_engine.chunks else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save document: {str(e)}")


from fastapi import UploadFile, File

@app.post("/api/documents/upload")
async def upload_document_file(file: UploadFile = File(...)):
    """Upload a .txt document file and immediately ingest it into the knowledge base."""
    from src.config import DOCUMENTS_DIR
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only plain text (.txt) files are supported.")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = DOCUMENTS_DIR / file.filename

    try:
        content = await file.read()
        target_path.write_bytes(content)
        # Automatically re-index knowledge base
        run_ingestion(reset_db=True)
        bm25_engine.load_index()
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded and indexed successfully!",
            "filename": file.filename,
            "chroma_count": vector_store.count(),
            "bm25_count": len(bm25_engine.chunks) if bm25_engine.chunks else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")


@app.post("/api/ingest")
async def trigger_ingest():
    """Trigger knowledge base re-ingestion."""
    try:
        success = run_ingestion(reset_db=True)
        # Reload memory indices
        bm25_engine.load_index()
        return {
            "success": success,
            "message": "Ingestion completed successfully!",
            "chroma_count": vector_store.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def process_user_query(req: QueryRequest):
    """Process user query through the multi-route agent pipeline."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = assistant.process_query(req.query.strip(), forced_route=req.route)
        return result
    except Exception as e:
        return {
            "query": req.query,
            "route": "direct_llm",
            "reason": f"Execution error: {str(e)}",
            "answer": f"An error occurred while executing the agent pipeline: {str(e)}",
            "internal_sources": [],
            "web_sources": []
        }


def run():
    import uvicorn
    print("\n=======================================================")
    print(" AGUI Interactive Agent Hub Live Server")
    print("=======================================================")
    print(" Access the live interactive web UI at:")
    print(" >>> http://localhost:8000")
    print(" >>> http://127.0.0.1:8000")
    print(" Press Ctrl+C to stop the server.")
    print("=======================================================\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
