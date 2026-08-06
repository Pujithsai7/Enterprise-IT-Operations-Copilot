import os
import uvicorn
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils import (
    parse_uploaded_file,
    detect_document_category,
    FAISSVectorStore,
    DocumentRegistry
)
from agents import build_copilot_graph

app = FastAPI(
    title="Enterprise IT Operations Copilot Backend API",
    description="FastAPI Service orchestrating LangGraph Multi-Agent Workflows, Hybrid Retrieval, Qdrant Vector Store, and LLMs.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Vector Store Instance initialized on backend
vector_store = FAISSVectorStore()
document_registry = DocumentRegistry()

class DiagnoseRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = []
    api_key: Optional[str] = None
    model_choice: Optional[str] = "kimi-k2.7-code:cloud"

class DiagnoseResponse(BaseModel):
    query: str
    final_response: str
    confidence_score: int
    executed_agents: List[str]
    validation_results: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Enterprise IT Operations Copilot FastAPI Backend Running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_chunks": len(vector_store.chunks),
        "primary_model": "kimi-k2.7-code:cloud"
    }

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    uploaded_docs = []
    for f in files:
        pages = parse_uploaded_file(f.file, filename_override=f.filename)
        full_text = "\n".join([p.get("content", "") if isinstance(p, dict) else str(p) for p in pages])
        auto_category = detect_document_category(f.filename, full_text)

        
        uploaded_docs.append({
            "id": len(vector_store.chunks) + 1,
            "source_type": auto_category,
            "title": f.filename,
            "pages": pages,
            "content": full_text,
            "total_pages": len(pages),
            "total_chars": len(full_text)
        })

    # Incremental indexing via document registry & Qdrant
    indexed_count = vector_store.build_index(uploaded_docs)
    return {
        "status": "success",
        "uploaded_files_count": len(files),
        "indexed_chunks_count": indexed_count,
        "total_vector_chunks": len(vector_store.chunks)
    }

@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose_issue(req: DiagnoseRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    copilot_graph = build_copilot_graph(vector_store)
    
    initial_state = {
        "query": req.query,
        "chat_history": req.chat_history or [],
        "doc_evidence": [],
        "net_evidence": [],
        "log_evidence": [],
        "inc_evidence": [],
        "validation_results": {},
        "confidence_score": 0,
        "executed_agents": [],
        "next_agent": "Supervisor",
        "final_response": "",
        "api_key": req.api_key,
        "model_choice": req.model_choice or "kimi-k2.7-code:cloud"
    }

    final_state = copilot_graph.invoke(initial_state)

    return DiagnoseResponse(
        query=req.query,
        final_response=final_state.get("final_response", "Diagnosis synthesis failed."),
        confidence_score=final_state.get("confidence_score", 85),
        executed_agents=final_state.get("executed_agents", []),
        validation_results=final_state.get("validation_results", {})
    )

@app.post("/clear")
def clear_knowledge_base():
    vector_store.clear_all()
    document_registry.clear()
    return {"status": "success", "message": "Vector index & document registry cleared."}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
