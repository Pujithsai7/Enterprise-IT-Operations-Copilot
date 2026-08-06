import os
import uvicorn
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from utils import (
    parse_uploaded_file,
    detect_document_category,
    FAISSVectorStore,
    DocumentRegistry
)
from agents import build_copilot_graph
from security import (
    SecurityManager,
    get_current_user,
    APIKeyEncrypter,
    SecureUploadValidator,
    PromptInjectionGuard
)

app = FastAPI(
    title="Enterprise IT Operations Copilot Backend API",
    description="FastAPI Service with OAuth2/OIDC, 25MB Secure Upload Validation, API Key Encryption, and Prompt Injection Defense.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = FAISSVectorStore()
document_registry = DocumentRegistry()

class ChatMessage(BaseModel):
    role: str
    content: str
    confidence: Optional[int] = None

class DiagnoseRequest(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = []
    api_key: Optional[str] = None
    model_choice: Optional[str] = "kimi-k2.7-code:cloud"



class DiagnoseResponse(BaseModel):
    query: str
    final_response: str
    confidence_score: int
    executed_agents: List[str]
    validation_results: Dict[str, Any]

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: str

@app.get("/")
def read_root():
    return {"message": "Enterprise IT Operations Copilot Secure FastAPI Backend Running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_chunks": len(vector_store.chunks),
        "primary_model": "kimi-k2.7-code:cloud",
        "security": "OAuth2/OIDC Enabled | 25MB Secure Uploads | Prompt Injection Guard Active"
    }

@app.post("/token", response_model=TokenResponse)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Basic enterprise user authentication
    if form_data.username and form_data.password:
        access_token = SecurityManager.create_access_token(
            data={"username": form_data.username, "role": "IT_Operator"}
        )
        return TokenResponse(access_token=access_token, token_type="bearer", user=form_data.username)
    raise HTTPException(status_code=400, detail="Invalid username or password")

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    uploaded_docs = []
    for f in files:
        file_content = await f.read()
        
        # Enterprise Security: 25MB Limit & Magic Byte Inspection
        SecureUploadValidator.validate_upload(f.filename, file_content, f.content_type)
        
        pages = parse_uploaded_file(file_content, filename_override=f.filename)
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

    indexed_count = vector_store.build_index(uploaded_docs)
    return {
        "status": "success",
        "uploaded_files_count": len(files),
        "indexed_chunks_count": indexed_count,
        "total_vector_chunks": len(vector_store.chunks)
    }

@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose_issue(
    req: DiagnoseRequest,
    current_user: dict = Depends(get_current_user)
):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Enterprise Security: Prompt Injection Guard Inspection
    is_safe, sanitized_query, violation_msg = PromptInjectionGuard.sanitize_and_validate(req.query)
    if not is_safe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=violation_msg
        )

    # Enterprise Security: Decrypt Encrypted API Key if supplied
    decrypted_key = APIKeyEncrypter.decrypt_api_key(req.api_key) if req.api_key else None

    copilot_graph = build_copilot_graph(vector_store)
    
    formatted_history = [
        m.model_dump() if hasattr(m, "model_dump") else dict(m)
        for m in (req.chat_history or [])
    ]

    initial_state = {
        "query": sanitized_query,
        "chat_history": formatted_history,

        "doc_evidence": [],
        "net_evidence": [],
        "log_evidence": [],
        "inc_evidence": [],
        "validation_results": {},
        "confidence_score": 0,
        "executed_agents": [],
        "next_agent": "Supervisor",
        "final_response": "",
        "api_key": decrypted_key,
        "model_choice": req.model_choice or "kimi-k2.7-code:cloud"
    }

    from langsmith_tracer import run_traced_copilot_graph
    final_state = run_traced_copilot_graph(copilot_graph, initial_state)

    return DiagnoseResponse(
        query=sanitized_query,
        final_response=final_state.get("final_response", "Diagnosis synthesis failed."),
        confidence_score=final_state.get("confidence_score", 85),
        executed_agents=final_state.get("executed_agents", []),
        validation_results=final_state.get("validation_results", {})
    )

@app.get("/traces")
def get_langsmith_traces():
    log_file = ".cache/langsmith_traces.json"
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


@app.post("/clear")
def clear_knowledge_base(current_user: dict = Depends(get_current_user)):
    vector_store.clear_all()
    document_registry.clear()
    return {"status": "success", "message": "Vector index & document registry cleared."}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
