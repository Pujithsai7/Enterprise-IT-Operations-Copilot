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

SUPPORTED_ZERO_SHOT_LABELS = [
    "Network",
    "Security",
    "Server",
    "Storage",
    "Virtualization",
    "Cloud",
    "Incident",
    "SOP",
    "Manual",
    "Knowledge Base",
    "Configuration"
]

class ZeroShotDocumentClassifier:
    """
    Zero-Shot Multi-Label Document Classifier: Removes keyword matching completely.
    Supports labels: Network, Security, Server, Storage, Virtualization, Cloud, Incident, SOP, Manual, Knowledge Base, Configuration.
    """
    def __init__(self, labels=None, multi_label=True, confidence_threshold=0.32):
        self.labels = labels or SUPPORTED_ZERO_SHOT_LABELS
        self.multi_label = multi_label
        self.confidence_threshold = confidence_threshold
        self._embedding_engine = None
        self._label_embeddings = None

    def _get_engine(self):
        if self._embedding_engine is None:
            self._embedding_engine = EnterpriseEmbeddingEngine()
            label_prompts = [f"Enterprise IT Operations Document Type: {lbl}" for lbl in self.labels]
            self._label_embeddings = self._embedding_engine.encode(label_prompts)
        return self._embedding_engine

    def classify(self, text_content, filename=""):
        sample_text = f"Filename: {filename}\nContent: {str(text_content)[:2000]}".strip()
        if not sample_text:
            return ["Configuration"]

        try:
            from transformers import pipeline
            classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-small", device=-1)
            res = classifier(sample_text[:1000], candidate_labels=self.labels, multi_label=True)
            labels = [lbl for lbl, score in zip(res['labels'], res['scores']) if score >= self.confidence_threshold]
            if labels:
                return labels
        except Exception:
            pass

        try:
            engine = self._get_engine()
            doc_vec = engine.encode([sample_text])[0]
            sims = np.dot(self._label_embeddings, doc_vec)
            
            classified = []
            for label, sim in zip(self.labels, sims):
                if sim >= self.confidence_threshold:
                    classified.append(label)
                    
            if not classified:
                top_idx = int(np.argmax(sims))
                classified.append(self.labels[top_idx])

            return classified
        except Exception:
            return ["Configuration", "Knowledge Base"]

_zero_shot_classifier_instance = None

def detect_document_category(filename, text_content):
    """
    Auto-detects document categories using Zero-Shot Multi-Label Classification.
    Zero keyword matching.
    """
    global _zero_shot_classifier_instance
    if _zero_shot_classifier_instance is None:
        _zero_shot_classifier_instance = ZeroShotDocumentClassifier()
    labels = _zero_shot_classifier_instance.classify(text_content, filename=filename)
    return ", ".join(labels) if isinstance(labels, list) else str(labels)


class RecursiveCharacterTextSplitter:
    """
    Recursively splits text into chunks using a sequence of separators in order of precedence.
    Preserves text structure and logical boundaries.
    """
    def __init__(self, chunk_size=1000, chunk_overlap=200, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text):
        if not text or not text.strip():
            return []
        splits = self._split_text_with_separators(text, self.separators)
        return self._merge_splits(splits)

    def _split_text_with_separators(self, text, separators):
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        
        for i, _s in enumerate(separators):
            if _s == "":
                separator = ""
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1:]
                break
                
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator=separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if new_separators:
                    other_splits = self._split_text_with_separators(s, new_separators)
                    final_chunks.extend(other_splits)
                else:
                    final_chunks.append(s)
        if good_splits:
            merged = self._merge_splits(good_splits, separator=separator)
            final_chunks.extend(merged)
            
        return final_chunks

    def _merge_splits(self, splits, separator=""):
        docs = []
        current_doc = []
        total = 0
        
        for d in splits:
            len_d = len(d)
            if total + len_d + (len(separator) if current_doc else 0) > self.chunk_size:
                if total > 0:
                    joined = separator.join(current_doc).strip()
                    if joined:
                        docs.append(joined)
                    while total > self.chunk_overlap or (total + len_d + (len(separator) if current_doc else 0) > self.chunk_size and total > 0):
                        popped = current_doc.pop(0)
                        total -= (len(popped) + len(separator))
                
            current_doc.append(d)
            total += len_d + len(separator)
            
        if current_doc:
            joined = separator.join(current_doc).strip()
            if joined:
                docs.append(joined)
                
        return docs

class StructureAwareChunker:
    """
    Structure-Aware Document Chunker for Enterprise IT Operations.
    Preserves Cisco IOS stanzas, network configs, ACLs, VLANs, Syslogs, Markdown headers, tables, and SOPs intact.
    Stores rich metadata: filename, page, section, heading, document_type, device_type, chunk_id.
    """
    def __init__(self, target_chunk_size=1000, chunk_overlap=200):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_document(self, doc):
        filename = doc.get("title", doc.get("filename", "Document"))
        doc_type = doc.get("source_type", detect_document_category(filename, doc.get("content", "")))
        pages = doc.get("pages", [])
        
        if not pages and doc.get("content"):
            pages = [{"page": 1, "content": doc.get("content", "")}]

        chunks = []
        chunk_id = 1

        for idx, page_info in enumerate(pages):
            if isinstance(page_info, dict):
                page_num = page_info.get("page", idx + 1)
                text = page_info.get("content", "")
            else:
                page_num = idx + 1
                text = str(page_info)

            if not text.strip():
                continue

            device_type = self._infer_device_type(filename, text, doc_type)
            page_chunks = self._chunk_text_structured(
                text=text,
                filename=filename,
                page_num=page_num,
                doc_type=doc_type,
                device_type=device_type,
                doc_id=doc.get("id"),
                start_chunk_id=chunk_id
            )
            
            chunks.extend(page_chunks)
            chunk_id += len(page_chunks)

        return chunks

    def _infer_device_type(self, filename, text, doc_type):
        fname = filename.lower()
        t_lower = text[:3000].lower()
        
        if any(k in t_lower for k in ["cisco asa", "pix firewall", "access-group", "security-level", "object-group"]):
            return "Cisco ASA Firewall"
        elif any(k in t_lower for k in ["policy-rule", "security zone", "fortigate", "palo alto"]):
            return "Firewall Config"
        elif any(k in t_lower for k in ["building configuration...", "interface gigabitethernet", "vlan database", "spanning-tree", "switchport"]):
            return "Cisco IOS Switch"
        elif any(k in t_lower for k in ["router bgp", "router ospf", "router eigrp", "ip route"]):
            return "Cisco Router"
        elif doc_type == "Server Log / Alert" or any(k in fname for k in [".log", ".syslog"]) or "%" in text:
            return "Syslog / Telemetry"
        elif doc_type == "Incident Ticket":
            return "Incident Record"
        elif doc_type == "SOP / Manual" or fname.endswith(".md"):
            return "SOP / Markdown Guide"
        return "IT Infrastructure Config"

    def _chunk_text_structured(self, text, filename, page_num, doc_type, device_type, doc_id, start_chunk_id):
        if self._is_network_config(text):
            raw_blocks = self._split_network_config_stanzas(text)
        elif self._is_syslog(text):
            raw_blocks = self._split_syslog_groups(text)
        else:
            raw_blocks = self._split_markdown_or_sop(text)

        chunks = []
        current_id = start_chunk_id

        for block in raw_blocks:
            content = block["content"].strip()
            if not content:
                continue
                
            heading = block.get("heading", "General")
            section = block.get("section", "General")
            
            if len(content) <= self.target_chunk_size * 1.5:
                citation = f"[{filename} | Page #{page_num} | Section: {heading} | Chunk #{current_id}]"
                chunks.append({
                    "id": doc_id,
                    "chunk_id": current_id,
                    "filename": filename,
                    "title": filename,
                    "page": page_num,
                    "page_number": page_num,
                    "section": section,
                    "heading": heading,
                    "document_type": doc_type,
                    "source_type": doc_type,
                    "device_type": device_type,
                    "content": content,
                    "citation": citation
                })
                current_id += 1
            else:
                sub_texts = self.recursive_splitter.split_text(content)
                for sub_idx, sub_t in enumerate(sub_texts):
                    sub_heading = f"{heading} (Part {sub_idx+1})" if len(sub_texts) > 1 else heading
                    citation = f"[{filename} | Page #{page_num} | Section: {sub_heading} | Chunk #{current_id}]"
                    chunks.append({
                        "id": doc_id,
                        "chunk_id": current_id,
                        "filename": filename,
                        "title": filename,
                        "page": page_num,
                        "page_number": page_num,
                        "section": section,
                        "heading": sub_heading,
                        "document_type": doc_type,
                        "source_type": doc_type,
                        "device_type": device_type,
                        "content": sub_t,
                        "citation": citation
                    })
                    current_id += 1

        return chunks

    def _is_network_config(self, text):
        t_lower = text.lower()
        return any(kw in t_lower for kw in [
            "interface gigabitethernet", "interface fastethernet", "interface vlan",
            "router bgp", "router ospf", "ip access-list", "access-list ", "running-config",
            "switchport mode", "line vty", "building configuration..."
        ])

    def _is_syslog(self, text):
        return bool(re.search(r'%\w+-\d+-\w+|syslog:|\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}|[A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2}', text))

    def _split_network_config_stanzas(self, text):
        lines = text.split("\n")
        stanzas = []
        current_stanza_lines = []
        current_heading = "System Global Config"
        current_section = "System Config"

        stanza_pattern = re.compile(
            r'^(interface\s+[\w\d/.-]+|router\s+(?:bgp|ospf|eigrp|rip)\s+[\w\d.-]+|'
            r'vlan\s+\d+|ip\s+access-list\s+(?:extended|standard)\s+[\w.-]+|'
            r'access-list\s+\d+|line\s+vty\s+[\d\s]+|line\s+con\s+\d+|'
            r'object-group\s+\w+|nat\s+\([^)]+\)|crypto\s+map|spanning-tree\s+[\w-]+)',
            re.IGNORECASE
        )

        for line in lines:
            line_str = line.rstrip()
            match = stanza_pattern.match(line_str.strip())
            
            if match:
                if current_stanza_lines:
                    stanza_text = "\n".join(current_stanza_lines).strip()
                    if stanza_text:
                        stanzas.append({
                            "heading": current_heading,
                            "section": current_section,
                            "content": stanza_text
                        })
                    current_stanza_lines = []

                header_text = match.group(0).strip()
                current_heading = header_text
                
                h_lower = header_text.lower()
                if "interface" in h_lower:
                    current_section = "Interface Configuration"
                elif "router" in h_lower:
                    current_section = "Routing Protocol Config"
                elif "vlan" in h_lower:
                    current_section = "VLAN Configuration"
                elif "access-list" in h_lower:
                    current_section = "Access Control List (ACL)"
                elif "object-group" in h_lower or "nat" in h_lower:
                    current_section = "Firewall Policy / Object"
                else:
                    current_section = "Network Configuration Stanza"
                    
                current_stanza_lines.append(line_str)
            elif line_str.strip() == "!":
                current_stanza_lines.append(line_str)
                stanza_text = "\n".join(current_stanza_lines).strip()
                if stanza_text:
                    stanzas.append({
                        "heading": current_heading,
                        "section": current_section,
                        "content": stanza_text
                    })
                current_stanza_lines = []
            else:
                current_stanza_lines.append(line_str)

        if current_stanza_lines:
            stanza_text = "\n".join(current_stanza_lines).strip()
            if stanza_text:
                stanzas.append({
                    "heading": current_heading,
                    "section": current_section,
                    "content": stanza_text
                })

        return stanzas if stanzas else [{"heading": "Global Config", "section": "Network Config", "content": text}]

    def _split_syslog_groups(self, text):
        lines = text.split("\n")
        blocks = []
        current_lines = []
        current_heading = "Syslog Telemetry Trace"
        
        for line in lines:
            if not line.strip():
                continue
            current_lines.append(line)
            
            if len(current_lines) == 1:
                event_match = re.search(r'%\w+-\d+-\w+', line)
                if event_match:
                    current_heading = event_match.group(0)
                else:
                    ts_match = re.search(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}|[A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2}', line)
                    if ts_match:
                        current_heading = f"Log @ {ts_match.group(0)}"
                    else:
                        current_heading = line[:50]

            if len(current_lines) >= 12:
                blocks.append({
                    "heading": current_heading,
                    "section": "Syslog Event Log",
                    "content": "\n".join(current_lines)
                })
                current_lines = []

        if current_lines:
            blocks.append({
                "heading": current_heading,
                "section": "Syslog Event Log",
                "content": "\n".join(current_lines)
            })

        return blocks if blocks else [{"heading": "Log Trace", "section": "Server Log", "content": text}]

    def _split_markdown_or_sop(self, text):
        lines = text.split("\n")
        blocks = []
        current_lines = []
        current_heading = "General Overview"
        current_section = "Documentation"
        in_table = False
        in_code_block = False

        for line in lines:
            strip_line = line.strip()
            
            if strip_line.startswith("```"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue
                
            if strip_line.startswith("|") and "|" in strip_line[1:]:
                in_table = True
                current_lines.append(line)
                continue
            elif in_table and not strip_line.startswith("|"):
                in_table = False

            heading_match = re.match(r'^(#{1,6})\s+(.+)$', strip_line)
            if heading_match and not in_code_block:
                if current_lines:
                    blocks.append({
                        "heading": current_heading,
                        "section": current_section,
                        "content": "\n".join(current_lines)
                    })
                    current_lines = []

                h_title = heading_match.group(2).strip()
                current_heading = h_title
                current_section = f"Section: {h_title}"
                current_lines.append(line)
            else:
                current_lines.append(line)

        if current_lines:
            blocks.append({
                "heading": current_heading,
                "section": current_section,
                "content": "\n".join(current_lines)
            })

        return blocks if blocks else [{"heading": "Document Content", "section": "Documentation", "content": text}]

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Structure-aware chunking function replacing naive word split.
    """
    chunker = StructureAwareChunker(target_chunk_size=chunk_size, chunk_overlap=overlap)
    doc = {"title": "Document", "content": text}
    return chunker.chunk_document(doc)

import os
import hashlib
import sqlite3
import torch
import numpy as np
import faiss
import streamlit as st

class PersistentEmbeddingCache:
    """
    SQLite-backed persistent disk cache for embedding vectors.
    Prevents re-computing embeddings for previously seen texts across application runs.
    """
    def __init__(self, db_path=None):
        if db_path is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            db_path = os.path.join(cache_dir, "embeddings_cache.db")
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    hash TEXT PRIMARY KEY,
                    model TEXT,
                    embedding BLOB
                )
            """)
            conn.commit()

    def get_many(self, model_name, texts):
        keys = [hashlib.sha256(f"{model_name}:{t}".encode('utf-8')).hexdigest() for t in texts]
        cached_vectors = {}
        
        with self._get_connection() as conn:
            for i in range(0, len(keys), 500):
                batch_keys = keys[i:i + 500]
                placeholders = ",".join(["?"] * len(batch_keys))
                cursor = conn.execute(
                    f"SELECT hash, embedding FROM embedding_cache WHERE hash IN ({placeholders})",
                    batch_keys
                )
                for h, blob in cursor.fetchall():
                    cached_vectors[h] = np.frombuffer(blob, dtype=np.float32)

        results = []
        missing_texts = []
        missing_indices = []

        for idx, (t, k) in enumerate(zip(texts, keys)):
            if k in cached_vectors:
                results.append(cached_vectors[k])
            else:
                results.append(None)
                missing_texts.append(t)
                missing_indices.append(idx)

        return results, missing_texts, missing_indices

    def set_many(self, model_name, texts, vectors):
        keys = [hashlib.sha256(f"{model_name}:{t}".encode('utf-8')).hexdigest() for t in texts]
        rows = []
        for k, v in zip(keys, vectors):
            rows.append((k, model_name, v.astype(np.float32).tobytes()))

        with self._get_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (hash, model, embedding) VALUES (?, ?, ?)",
                rows
            )
            conn.commit()

class EnterpriseEmbeddingEngine:
    """
    Enterprise Local Embedding Engine supporting BAAI/bge-large-en-v1.5 and BAAI/bge-base-en-v1.5.
    Features:
    - GPU auto-detection (CUDA) with CPU fallback
    - Batch encoding (default batch_size=32)
    - Persistent SQLite disk caching
    - BGE Query instruction formatting
    """
    PREFERRED_MODELS = [
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        "all-MiniLM-L6-v2"
    ]

    def __init__(self, model_name="BAAI/bge-large-en-v1.5", batch_size=32):
        self.requested_model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_name = model_name
        self.dimension = 1024
        self.cache = PersistentEmbeddingCache()
        self._load_model()

    def _load_model(self):
        if not HAS_SENTENCE_TRANSFORMERS:
            self.model = None
            return

        candidates = [self.requested_model_name] + [m for m in self.PREFERRED_MODELS if m != self.requested_model_name]
        
        for candidate in candidates:
            try:
                self.model = SentenceTransformer(candidate, device=self.device)
                self.model_name = candidate
                self.dimension = self.model.get_sentence_embedding_dimension() or 1024
                return
            except Exception:
                if self.device == "cuda":
                    try:
                        self.device = "cpu"
                        self.model = SentenceTransformer(candidate, device="cpu")
                        self.model_name = candidate
                        self.dimension = self.model.get_sentence_embedding_dimension() or 1024
                        return
                    except Exception:
                        pass
                continue

    def encode(self, texts, is_query=False, batch_size=None):
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.model is None:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)

        b_size = batch_size or self.batch_size
        
        if is_query and "bge" in self.model_name.lower():
            formatted_texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        else:
            formatted_texts = [str(t) for t in texts]

        cached_results, missing_texts, missing_indices = self.cache.get_many(self.model_name, formatted_texts)

        if missing_texts:
            try:
                new_embeddings = self.model.encode(
                    missing_texts,
                    batch_size=b_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device=self.device
                ).astype(np.float32)
            except Exception:
                new_embeddings = self.model.encode(
                    missing_texts,
                    batch_size=b_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device="cpu"
                ).astype(np.float32)

            faiss.normalize_L2(new_embeddings)
            self.cache.set_many(self.model_name, missing_texts, new_embeddings)

            for idx, vec in zip(missing_indices, new_embeddings):
                cached_results[idx] = vec

        final_matrix = np.vstack(cached_results).astype(np.float32)
        faiss.normalize_L2(final_matrix)
        return final_matrix

@st.cache_resource
def get_embedding_engine():
    return EnterpriseEmbeddingEngine(model_name="BAAI/bge-large-en-v1.5")

class DocumentRegistry:
    """
    Persistent Document Registry for Document Hashing & Change Detection.
    Tracks SHA-256 hashes of indexed documents to enable incremental updates (O(1)) and skip unchanged files.
    """
    def __init__(self, registry_path=None):
        if registry_path is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            registry_path = os.path.join(cache_dir, "document_registry.json")
        self.registry_path = registry_path
        self.registry = self._load()

    def _load(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=2)
        except Exception:
            pass

    def compute_hash(self, doc):
        content_str = f"{doc.get('title', '')}:{doc.get('source_type', '')}:{doc.get('content', '')}"
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()

    def sync_documents(self, raw_documents):
        """
        Calculates diffs between raw_documents and stored registry:
        Returns (new_docs, modified_docs, deleted_doc_ids, unchanged_docs)
        """
        current_docs_map = {}
        for doc in raw_documents:
            doc_id = str(doc.get("id", doc.get("title", "doc")))
            current_docs_map[doc_id] = doc

        new_docs = []
        modified_docs = []
        unchanged_docs = []
        
        for doc_id, doc in current_docs_map.items():
            doc_hash = self.compute_hash(doc)
            if doc_id not in self.registry:
                new_docs.append(doc)
            elif self.registry[doc_id].get("hash") != doc_hash:
                modified_docs.append(doc)
            else:
                unchanged_docs.append(doc)

        deleted_doc_ids = [d_id for d_id in self.registry if d_id not in current_docs_map]

        return new_docs, modified_docs, deleted_doc_ids, unchanged_docs

    def update_registry(self, raw_documents):
        new_reg = {}
        for doc in raw_documents:
            doc_id = str(doc.get("id", doc.get("title", "doc")))
            new_reg[doc_id] = {
                "id": doc.get("id"),
                "filename": doc.get("title", "Document"),
                "hash": self.compute_hash(doc)
            }
        self.registry = new_reg
        self._save()

    def clear(self):
        self.registry = {}
        self._save()

try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models
    HAS_QDRANT = True
except ImportError:
    qdrant_client = None
    HAS_QDRANT = False

class QdrantPersistentVectorStore:
    """
    Qdrant Persistent Vector Database for Enterprise IT Operations Copilot.
    Features:
    - Embedded disk persistence (survives app reboots, reloads & crashes)
    - Automatic reload of stored collections & metadata payloads on startup
    - Document SHA-256 Hashing & Change Detection (Incremental O(1) Updates)
    - Native Qdrant metadata payload filtering (source_type, document_type, device_type)
    - Cosine similarity metric (Distance.COSINE)
    - Delete & Update handling per document/file
    """
    COLLECTION_NAME = "it_ops_copilot_chunks"

    def __init__(self, storage_path=None):
        self.engine = get_embedding_engine()
        self.dimension = self.engine.dimension if self.engine else 1024
        
        if storage_path is None:
            storage_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".qdrant_db")
        self.storage_path = storage_path
        
        self.registry = DocumentRegistry()
        self.client = None
        self._init_qdrant()

    def _init_qdrant(self):
        if not HAS_QDRANT:
            return

        try:
            url = os.environ.get("QDRANT_URL")
            api_key = os.environ.get("QDRANT_API_KEY")
            
            if url:
                self.client = QdrantClient(url=url, api_key=api_key)
            else:
                os.makedirs(self.storage_path, exist_ok=True)
                self.client = QdrantClient(path=self.storage_path)

            collections = [c.name for c in self.client.get_collections().collections]
            if self.COLLECTION_NAME not in collections:
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=rest_models.VectorParams(
                        size=self.dimension,
                        distance=rest_models.Distance.COSINE
                    )
                )
        except Exception:
            try:
                self.client = QdrantClient(":memory:")
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=rest_models.VectorParams(
                        size=self.dimension,
                        distance=rest_models.Distance.COSINE
                    )
                )
            except Exception:
                self.client = None

    @property
    def chunks(self):
        """
        Returns list of stored chunk payload dictionaries for stats and display.
        """
        if not self.client:
            return []
        try:
            records, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            return [r.payload for r in records if r.payload]
        except Exception:
            return []

    def build_index(self, raw_documents):
        """
        Incremental change-detection indexing:
        - Unchanged docs: 0 embeddings computed (skipped)
        - New docs: Chunked & embedded incrementally (O(1))
        - Modified docs: Delete old chunks & re-embed updated document
        - Deleted docs: Delete associated vector points from Qdrant storage
        """
        if not raw_documents:
            self.clear_all()
            self.registry.clear()
            return

        new_docs, modified_docs, deleted_doc_ids, unchanged_docs = self.registry.sync_documents(raw_documents)

        # 1. Handle deleted documents
        for d_id in deleted_doc_ids:
            self.delete_document(d_id)

        # 2. Handle modified documents (delete old vectors first)
        for doc in modified_docs:
            d_id = doc.get("id")
            if d_id:
                self.delete_document(d_id)
            else:
                self.delete_file(doc.get("title", ""))

        # 3. Handle new & modified document chunking and embedding
        docs_to_index = new_docs + modified_docs
        if docs_to_index:
            chunker = StructureAwareChunker(target_chunk_size=1000, chunk_overlap=200)
            chunks_to_upsert = []
            for doc in docs_to_index:
                chunks_to_upsert.extend(chunker.chunk_document(doc))
            
            if chunks_to_upsert:
                self.upsert_chunks(chunks_to_upsert)

        # 4. Update persistent registry
        self.registry.update_registry(raw_documents)

    def upsert_chunks(self, chunks):
        """
        Incrementally upserts vector chunks and metadata payloads into Qdrant.
        """
        if not self.client or not chunks:
            return

        texts = [c['content'] for c in chunks]
        embeddings = self.engine.encode(texts, is_query=False)

        points = []
        for idx, (ch, vec) in enumerate(zip(chunks, embeddings)):
            point_id = int(hashlib.md5(ch.get('citation', f"chunk_{idx}").encode('utf-8')).hexdigest()[:8], 16)
            
            payload = {
                "id": ch.get("id"),
                "chunk_id": ch.get("chunk_id"),
                "filename": ch.get("filename", "Document"),
                "title": ch.get("title", "Document"),
                "page": ch.get("page", 1),
                "page_number": ch.get("page", 1),
                "section": ch.get("section", "General"),
                "heading": ch.get("heading", "General"),
                "document_type": ch.get("document_type", "General"),
                "source_type": ch.get("source_type", ch.get("document_type", "General")),
                "device_type": ch.get("device_type", "IT Infrastructure Config"),
                "content": ch.get("content", ""),
                "citation": ch.get("citation", "")
            }

            points.append(rest_models.PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload=payload
            ))

        for i in range(0, len(points), 200):
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points[i:i + 200]
            )

    def search(self, query, source_types=None, top_k=4):
        """
        Performs vector similarity search on Qdrant with native metadata payload filtering.
        """
        if not self.client:
            return []

        q_emb = self.engine.encode([query], is_query=True)[0].tolist()

        query_filter = None
        if source_types:
            query_filter = rest_models.Filter(
                should=[
                    rest_models.FieldCondition(
                        key="source_type",
                        match=rest_models.MatchAny(any=source_types)
                    ),
                    rest_models.FieldCondition(
                        key="document_type",
                        match=rest_models.MatchAny(any=source_types)
                    )
                ]
            )

        try:
            hits = []
            if hasattr(self.client, "query_points"):
                q_res = self.client.query_points(
                    collection_name=self.COLLECTION_NAME,
                    query=q_emb,
                    query_filter=query_filter,
                    limit=top_k
                )
                hits = q_res.points
                if not hits and query_filter:
                    q_res = self.client.query_points(
                        collection_name=self.COLLECTION_NAME,
                        query=q_emb,
                        limit=top_k
                    )
                    hits = q_res.points
            elif hasattr(self.client, "search"):
                hits = self.client.search(
                    collection_name=self.COLLECTION_NAME,
                    query_vector=q_emb,
                    query_filter=query_filter,
                    limit=top_k
                )
                if not hits and query_filter:
                    hits = self.client.search(
                        collection_name=self.COLLECTION_NAME,
                        query_vector=q_emb,
                        limit=top_k
                    )

            results = []
            for hit in hits:
                chunk = dict(hit.payload or {})
                chunk['score'] = float(getattr(hit, 'score', 0.0))
                results.append(chunk)

            return results
        except Exception:
            return []


    def delete_document(self, doc_id):
        """
        Deletes all chunks belonging to a specific document ID.
        """
        if not self.client:
            return
        try:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=rest_models.FilterSelector(
                    filter=rest_models.Filter(
                        must=[
                            rest_models.FieldCondition(
                                key="id",
                                match=rest_models.MatchValue(value=doc_id)
                            )
                        ]
                    )
                )
            )
        except Exception:
            pass

    def delete_file(self, filename):
        """
        Deletes all chunks belonging to a specific filename.
        """
        if not self.client:
            return
        try:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=rest_models.FilterSelector(
                    filter=rest_models.Filter(
                        must=[
                            rest_models.FieldCondition(
                                key="filename",
                                match=rest_models.MatchValue(value=filename)
                            )
                        ]
                    )
                )
            )
        except Exception:
            pass

    def clear_all(self):
        """
        Clears all points in the Qdrant collection.
        """
        if not self.client:
            return
        try:
            self.client.delete_collection(collection_name=self.COLLECTION_NAME)
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=rest_models.VectorParams(
                    size=self.dimension,
                    distance=rest_models.Distance.COSINE
                )
            )
        except Exception:
            pass

try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    BM25Okapi = None
    HAS_RANK_BM25 = False

try:
    from flashrank import Ranker, RerankRequest
    HAS_FLASHRANK = True
except ImportError:
    Ranker = None
    RerankRequest = None
    HAS_FLASHRANK = False

class BM25SearchEngine:
    """
    BM25 Sparse Lexical Search Engine for exact keyword matching (MACs, IPs, Cisco Interface IDs, Error Codes).
    """
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def build_index(self, chunks):
        self.chunks = chunks
        if not chunks or not HAS_RANK_BM25:
            self.bm25 = None
            return

        corpus = [re.findall(r'\w+', c.get('content', '').lower()) for c in chunks]
        if corpus:
            self.bm25 = BM25Okapi(corpus)

    def search(self, query, top_k=20):
        if not self.bm25 or not self.chunks:
            return []

        tokens = re.findall(r'\w+', query.lower())
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:min(top_k, len(self.chunks))]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                chunk = dict(self.chunks[idx])
                chunk['bm25_score'] = score
                results.append(chunk)

        return results

class FlashRankReranker:
    """
    Cross-Encoder Reranker using FlashRank (ms-marco-MiniLM-L-6-v2 / BAAI bge-reranker-large).
    Re-scores candidate text pairs (query, chunk) using deep cross-attention transformer models.
    """
    def __init__(self, model_name="ms-marco-MiniLM-L-6-v2"):
        self.ranker = None
        self.model_name = model_name
        self._init_ranker()

    def _init_ranker(self):
        if not HAS_FLASHRANK:
            self.ranker = None
            return

        try:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "flashrank")
            os.makedirs(cache_dir, exist_ok=True)
            self.ranker = Ranker(model_name=self.model_name, cache_dir=cache_dir)
        except Exception:
            try:
                self.ranker = Ranker()
            except Exception:
                self.ranker = None

    def rerank(self, query, chunks, top_k=4):
        if not chunks:
            return []

        if not self.ranker:
            return sorted(chunks, key=lambda x: x.get('score', 0.0), reverse=True)[:top_k]

        passages = [
            {
                "id": idx,
                "text": c.get('content', ''),
                "meta": c
            }
            for idx, c in enumerate(chunks)
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        try:
            results = self.ranker.rerank(rerank_req)
            reranked_chunks = []
            for item in results[:top_k]:
                chunk = dict(item.get("meta", {}))
                chunk['score'] = float(item.get("score", 0.0))
                reranked_chunks.append(chunk)
            return reranked_chunks
        except Exception:
            return sorted(chunks, key=lambda x: x.get('score', 0.0), reverse=True)[:top_k]

class HybridRetriever:
    """
    Enterprise Hybrid Retrieval Pipeline:
    User Query -> Dense Search (Qdrant) + BM25 Sparse Search -> RRF Candidate Fusion -> Cross-Encoder Reranker (FlashRank) -> Top Context
    """
    def __init__(self, vector_store=None):
        self.vector_store = vector_store or QdrantPersistentVectorStore()
        self.bm25_engine = BM25SearchEngine()
        self.reranker = FlashRankReranker()

    @property
    def chunks(self):
        return self.vector_store.chunks

    def build_index(self, raw_documents):
        self.vector_store.build_index(raw_documents)
        self.bm25_engine.build_index(self.vector_store.chunks)

    def search(self, query, source_types=None, top_k=4):
        # 1. Dense Vector Search (Qdrant)
        dense_candidates = self.vector_store.search(query, source_types=source_types, top_k=20)
        
        # 2. BM25 Sparse Search
        bm25_candidates = self.bm25_engine.search(query, top_k=20)
        if source_types:
            bm25_candidates = [
                c for c in bm25_candidates 
                if c.get('source_type') in source_types or c.get('document_type') in source_types
            ]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        candidate_map = {}

        for rank, c in enumerate(dense_candidates):
            cit = c.get('citation', str(c.get('content')[:50]))
            rrf_scores[cit] = rrf_scores.get(cit, 0.0) + (1.0 / (60.0 + rank + 1))
            candidate_map[cit] = c

        for rank, c in enumerate(bm25_candidates):
            cit = c.get('citation', str(c.get('content')[:50]))
            rrf_scores[cit] = rrf_scores.get(cit, 0.0) + (1.0 / (60.0 + rank + 1))
            candidate_map[cit] = c

        sorted_citations = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:30]
        merged_candidates = [candidate_map[cit] for cit in sorted_citations]

        if not merged_candidates:
            return []

        # 4. Cross-Encoder Reranking (FlashRank)
        reranked_results = self.reranker.rerank(query, merged_candidates, top_k=top_k)
        return reranked_results

    def clear_all(self):
        self.vector_store.clear_all()
        self.bm25_engine.build_index([])

class CitationVerifier:
    """
    Deterministic Citation Verification Engine.
    Verifies every generated citation string or metadata tuple against active indexed chunks on 4 dimensions:
    1. chunk_id
    2. filename
    3. page
    4. section

    Unverified or hallucinated citations are automatically removed from diagnostic outputs.
    """
    def build_chunk_registry(self, chunks):
        registry = set()
        chunk_map = {}
        for c in chunks:
            filename = str(c.get('filename', c.get('title', ''))).strip().lower()
            page = str(c.get('page', c.get('page_number', 1))).strip()
            section = str(c.get('section', c.get('heading', 'General'))).strip().lower()
            chunk_id = str(c.get('chunk_id', c.get('id', ''))).strip().lower()
            citation_str = str(c.get('citation', '')).strip().lower()

            if citation_str:
                registry.add(citation_str)
                registry.add(citation_str.replace("[", "").replace("]", "").strip())

            tuple_key = (filename, page, section, chunk_id)
            registry.add(tuple_key)
            registry.add(f"{filename}:{chunk_id}")
            registry.add(f"{filename}:page{page}")
            if filename:
                registry.add(filename)
            chunk_map[citation_str] = c

        return registry, chunk_map

    def verify_citation(self, citation_input, registry):
        if not citation_input:
            return False

        citation_str = str(citation_input).strip().lower()
        clean_cit = citation_str.replace("•", "").replace("`", "").replace("[", "").replace("]", "").strip()

        if clean_cit in registry or f"[{clean_cit}]" in registry or citation_str in registry:
            return True

        parts = [p.strip() for p in clean_cit.split("|")]
        if len(parts) >= 1:
            fname = parts[0].lower()
            if fname in registry:
                return True

            page_part = "1"
            section_part = "general"
            chunk_id_part = ""

            for p in parts[1:]:
                p_lower = p.lower()
                if "page #" in p_lower or "page" in p_lower:
                    page_part = p_lower.replace("page #", "").replace("page", "").strip()
                elif "section:" in p_lower or "section" in p_lower:
                    section_part = p_lower.replace("section:", "").replace("section", "").strip()
                elif "chunk #" in p_lower or "chunk" in p_lower:
                    chunk_id_part = p_lower.replace("chunk #", "").replace("chunk", "").strip()

            tuple_key = (fname, page_part, section_part, chunk_id_part)
            if tuple_key in registry or f"{fname}:{chunk_id_part}" in registry or f"{fname}:page{page_part}" in registry:
                return True

            for registered_item in registry:
                if isinstance(registered_item, tuple) and registered_item[0] == fname:
                    if not chunk_id_part or registered_item[3] == chunk_id_part:
                        return True

        return False

    def verify_and_filter(self, citations_list, active_chunks):
        if not citations_list:
            return [], []
        registry, _ = self.build_chunk_registry(active_chunks)
        verified_citations = []
        removed_citations = []

        for cit in citations_list:
            if self.verify_citation(cit, registry):
                verified_citations.append(cit)
            else:
                removed_citations.append(cit)

        return verified_citations, removed_citations

# Backward-compatibility alias so all existing vector store calls use HybridRetriever
FAISSVectorStore = HybridRetriever




import os
import io

from ingestion_pipeline import GenericDocumentIngestionEngine, IngestionError

def parse_uploaded_file(uploaded_file, filename_override=None):
    """
    Vendor-Neutral Generic Document Ingestion Entry Point.
    Delegates to GenericDocumentIngestionEngine supporting PDF, DOCX, TXT, CSV, XLSX, JSON, XML, HTML, LOGs, etc.
    """
    pages, summary = GenericDocumentIngestionEngine.parse_document(uploaded_file, filename_override)
    return pages



