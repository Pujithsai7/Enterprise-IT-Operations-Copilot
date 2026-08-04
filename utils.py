import re
import math
import numpy as np
import faiss
import streamlit as st
from collections import Counter

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

ERROR_KEYWORDS = [
    "error", "failed", "failure", "critical", "timeout", "denied",
    "down", "unreachable", "crc", "drop", "panic", "invalid",
    "exception", "err-disable", "link-flap", "hogging", "crash",
    "disconnect", "offline", "degraded", "reboot", "alarm"
]

def detect_fault_indicators(query, chunks):
    """
    Scans retrieved document chunks for explicit fault/error indicators.
    Returns (has_fault, max_fault_keyword_matches)
    """
    text_corpus = " ".join([c.get('content', '') for c in chunks]).lower()
    found_keywords = [kw for kw in ERROR_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', text_corpus)]
    return len(found_keywords) > 0, len(found_keywords)

def detect_document_category(filename, text_content):
    """
    Auto-detects document category based on filename and text content analysis.
    Categories: 'SOP / Manual', 'Network Configuration', 'Server Log / Alert', 'Incident Ticket'
    """
    fname = filename.lower()
    content = str(text_content).lower()[:3000]
    
    # 1. Incident Ticket Detection
    if any(k in fname for k in ["ticket", "inc-", "incident", "case", "resolution", "outage"]) or \
       any(k in content for k in ["ticket inc-", "incident summary", "past incident", "ticket #", "resolution:"]):
        return "Incident Ticket"

    # 2. Server Log / Alert Detection
    if fname.endswith(".log") or fname.endswith(".syslog") or \
       any(k in fname for k in ["log", "syslog", "alert", "trace", "audit", "event", "telemetry"]) or \
       any(k in content for k in ["syslog:", "%link-", "%ethport", "%sys-", "alert #", "timestamp", "ping timeout", "error code"]):
        return "Server Log / Alert"

    # 3. Network Configuration & Topology Detection
    if any(k in fname for k in ["config", "cfg", "topology", "switch", "router", "vlan", "bgp", "ospf"]) or \
       any(k in content for k in ["interface gigabitethernet", "vlan ", "router bgp", "switch-", "ip route", "duplex auto", "speed auto", "running-config"]):
        return "Network Configuration"

    # 4. SOP / Equipment Manual (Default for PDFs, Guides, Technical Documentation)
    return "SOP / Manual"

@st.cache_resource
def get_embedding_model():
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            return None
    return None

def chunk_text(text, chunk_size=250, overlap=50):
    """
    Chunks text using a sliding window with overlap.
    """
    if not text or not str(text).strip():
        return []
    words = str(text).split()
    if not words:
        return []
    
    chunks = []
    i = 0
    chunk_id = 1
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_str = " ".join(chunk_words)
        if chunk_str.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "content": chunk_str
            })
            chunk_id += 1
        i += (chunk_size - overlap) if len(chunk_words) == chunk_size else len(chunk_words)
    return chunks

class FAISSVectorStore:
    """
    FAISS Index + Embeddings Vector Store for RAG Document Retrieval with Page Number Tracking.
    """
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []
        self.model = get_embedding_model()

    def build_index(self, raw_documents):
        self.chunks = []
        all_texts = []
        
        for doc in raw_documents:
            pages = doc.get('pages', [])
            if not pages and doc.get('content'):
                pages = [{"page": 1, "content": doc.get('content', '')}]
                
            for idx, p_info in enumerate(pages):
                if isinstance(p_info, dict):
                    page_num = p_info.get("page", idx + 1)
                    text_content = p_info.get("content", "")
                elif isinstance(p_info, str):
                    page_num = idx + 1
                    text_content = p_info
                else:
                    page_num = idx + 1
                    text_content = str(p_info)

                text_chunks = chunk_text(text_content, chunk_size=250, overlap=50)
                if not text_chunks and text_content.strip():
                    text_chunks = [{"chunk_id": 1, "content": text_content.strip()}]
                    
                for ch in text_chunks:
                    citation_str = f"[{doc.get('title', 'Doc')} | Page #{page_num} | Chunk #{ch['chunk_id']}]"
                    chunk_entry = {
                        "id": doc.get("id"),
                        "source_type": doc.get('source_type', 'General'),
                        "title": doc.get('title', 'Document'),
                        "page_number": page_num,
                        "chunk_id": ch['chunk_id'],
                        "content": ch['content'],
                        "citation": citation_str
                    }
                    self.chunks.append(chunk_entry)
                    all_texts.append(ch['content'])
                
        if not all_texts:
            self.index = faiss.IndexFlatIP(self.dimension)
            return

        if self.model is not None:
            try:
                embeddings = self.model.encode(all_texts, convert_to_numpy=True)
                embeddings = embeddings.astype(np.float32)
                faiss.normalize_L2(embeddings)
                self.dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(embeddings)
                return
            except Exception:
                pass

        self._build_fallback_tfidf(all_texts)

    def _build_fallback_tfidf(self, all_texts):
        vocab = sorted(list(set(re.findall(r'\w+', " ".join(all_texts).lower()))))
        vocab_map = {word: idx for idx, word in enumerate(vocab[:384])}
        self.vocab_map = vocab_map
        self.dimension = len(vocab_map) if vocab_map else 384
        
        embeddings = []
        for text in all_texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            tokens = re.findall(r'\w+', text.lower())
            for t in tokens:
                if t in vocab_map:
                    vec[vocab_map[t]] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
            
        embeddings_np = np.array(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.dimension)
        if len(embeddings_np) > 0:
            self.index.add(embeddings_np)

    def search(self, query, source_types=None, top_k=4):
        if self.index.ntotal == 0 or not self.chunks:
            return []
            
        if self.model is not None:
            try:
                q_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
                faiss.normalize_L2(q_emb)
            except Exception:
                q_emb = None
        else:
            q_emb = None

        if q_emb is None:
            vocab_map = getattr(self, "vocab_map", {})
            q_emb = np.zeros((1, self.dimension), dtype=np.float32)
            tokens = re.findall(r'\w+', query.lower())
            for t in tokens:
                if t in vocab_map:
                    q_emb[0, vocab_map[t]] += 1.0
            norm = np.linalg.norm(q_emb)
            if norm > 0:
                q_emb = q_emb / norm

        scores, indices = self.index.search(q_emb, k=min(self.index.ntotal, 30))
        
        results = []
        for idx_pos, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            score = float(scores[0][idx_pos])
            chunk['score'] = score
            
            if source_types:
                if chunk['source_type'] in source_types:
                    results.append((score, chunk))
            else:
                results.append((score, chunk))
                
            if len(results) >= top_k:
                break
                
        if not results and source_types:
            for idx_pos, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                chunk = dict(self.chunks[idx])
                score = float(scores[0][idx_pos])
                chunk['score'] = score
                results.append((score, chunk))
                if len(results) >= top_k:
                    break

        return [r[1] for r in results]

def parse_uploaded_file(uploaded_file):
    filename = uploaded_file.name.lower()
    pages = []
    try:
        if filename.endswith(".pdf") and pypdf:
            reader = pypdf.PdfReader(uploaded_file)
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    pages.append({"page": idx + 1, "content": txt.strip()})
            if not pages:
                pages = [{"page": 1, "content": f"[File: {uploaded_file.name} - Empty or non-text PDF]"}]
        elif filename.endswith(".docx") and docx:
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs]).strip()
            pages = [{"page": 1, "content": text if text else f"[File: {uploaded_file.name} - Empty DOCX]"}]
        else:
            raw_bytes = uploaded_file.read()
            text = raw_bytes.decode("utf-8", errors="ignore").strip()
            pages = [{"page": 1, "content": text if text else f"[File: {uploaded_file.name} - Empty text file]"}]
    except Exception as e:
        pages = [{"page": 1, "content": f"[Error reading file {uploaded_file.name}: {str(e)}]"}]
    return pages
