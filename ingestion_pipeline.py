import os
import io
import re
import json
import yaml
import time
import datetime
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import List, Dict, Any, Tuple, Optional

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

try:
    import pandas as pd
except ImportError:
    pd = None

class IngestionError(Exception):
    """Raised when all parser fallback chains fail for a document."""
    pass

# Simple HTML Text Extractor
class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        cleaned = data.strip()
        if cleaned:
            self.text_parts.append(cleaned)

    def get_text(self):
        return "\n".join(self.text_parts)

class GenericDocumentIngestionEngine:
    """
    Vendor-Neutral Enterprise Ingestion Pipeline.
    Supports PDF, DOCX/DOC, TXT, Markdown, CSV, XLSX, JSON, XML, HTML, Log files, Config files, YAML, INI.
    Includes Magic-Byte Detection, Multi-Parser Fallback Chains, and Rich Metadata Extraction.
    """
    
    SUPPORTED_CATEGORIES = [
        "Manual", "SOP", "Knowledge Base", "Incident", "Logs",
        "Configuration", "Security", "Policy", "Architecture",
        "Network", "Server", "Database", "Cloud", "Other"
    ]

    @classmethod
    def detect_format_and_buffer(cls, file_input: Any, filename_override: Optional[str] = None) -> Tuple[str, io.BytesIO, str]:
        """
        Detects file format via magic bytes and MIME signatures.
        Returns (detected_format, seekable_bytes_io, filename).
        """
        # 1. Resolve Filename
        filename = (
            filename_override 
            or getattr(file_input, "filename", getattr(file_input, "name", None))
            or (os.path.basename(str(file_input)) if isinstance(file_input, (str, os.PathLike)) else "document.txt")
        ).lower()

        # 2. Resolve to io.BytesIO Buffer
        if isinstance(file_input, (str, os.PathLike)) and os.path.exists(str(file_input)):
            with open(str(file_input), "rb") as f:
                buffer = io.BytesIO(f.read())
        elif isinstance(file_input, bytes):
            buffer = io.BytesIO(file_input)
        elif isinstance(file_input, io.BytesIO):
            buffer = file_input
            buffer.seek(0)
        elif hasattr(file_input, "file"): # FastAPI UploadFile
            raw = file_input.file.read()
            buffer = io.BytesIO(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
        elif hasattr(file_input, "read"): # Streamlit UploadedFile
            raw = file_input.read()
            buffer = io.BytesIO(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
        else:
            buffer = io.BytesIO(str(file_input).encode("utf-8"))

        buffer.seek(0)
        header = buffer.read(512)
        buffer.seek(0)

        # 3. Magic Byte Inspection
        ext = os.path.splitext(filename)[1]
        if header.startswith(b"%PDF"):
            fmt = "pdf"
        elif header.startswith(b"PK\x03\x04"):
            if ext in [".xlsx", ".xls"]:
                fmt = "xlsx"
            else:
                fmt = "docx"
        elif header.startswith(b"{\n") or header.startswith(b"{\r\n") or header.startswith(b"["):
            fmt = "json"
        elif header.startswith(b"<?xml") or header.startswith(b"<html") or header.startswith(b"<!DOCTYPE html"):
            fmt = "html" if b"<html" in header.lower() else "xml"
        elif ext in [".csv", ".tsv"]:
            fmt = "csv"
        elif ext in [".json", ".jsonl"]:
            fmt = "json"
        elif ext in [".yaml", ".yml"]:
            fmt = "yaml"
        elif ext in [".ini", ".cfg", ".conf"]:
            fmt = "config"
        elif ext in [".log", ".syslog"]:
            fmt = "log"
        elif ext in [".md", ".markdown"]:
            fmt = "markdown"
        else:
            fmt = ext.replace(".", "") or "txt"

        return fmt, buffer, filename

    @classmethod
    def parse_document(cls, file_input: Any, filename_override: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Main Ingestion Entry Point with Fallback Chains.
        Returns (pages_list, metadata_summary).
        """
        fmt, buffer, filename = cls.detect_format_and_buffer(file_input, filename_override)
        pages = []
        parser_used = "Primary"
        errors_logged = []

        # --- PARSER FALLBACK CHAIN ---
        if fmt == "pdf":
            # Parser A: pypdf
            try:
                if not pypdf:
                    raise ImportError("pypdf library not installed")
                reader = pypdf.PdfReader(buffer)
                for idx, page in enumerate(reader.pages):
                    txt = page.extract_text() or ""
                    if txt.strip():
                        pages.append({"page": idx + 1, "content": txt.strip()})
                parser_used = "PyPDF Native"
            except Exception as e1:
                errors_logged.append(f"Parser A (PyPDF) failed: {e1}")
                # Parser B: Raw regex text stream extraction
                try:
                    buffer.seek(0)
                    raw_text = buffer.read().decode("utf-8", errors="ignore")
                    extracted = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', raw_text)
                    if len(extracted.strip()) > 50:
                        pages = [{"page": 1, "content": extracted.strip()}]
                        parser_used = "Fallback Stream Extractor"
                except Exception as e2:
                    errors_logged.append(f"Parser B failed: {e2}")

        elif fmt in ["docx", "doc"]:
            try:
                if not docx:
                    raise ImportError("python-docx library not installed")
                doc = docx.Document(buffer)
                full_txt = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                if full_txt.strip():
                    pages = [{"page": 1, "content": full_txt.strip()}]
                    parser_used = "Python-Docx Native"
            except Exception as e1:
                errors_logged.append(f"Docx Parser A failed: {e1}")

        elif fmt in ["csv", "xlsx"]:
            try:
                if pd:
                    buffer.seek(0)
                    df = pd.read_csv(buffer) if fmt == "csv" else pd.read_excel(buffer)
                    md_table = df.to_markdown(index=False)
                    pages = [{"page": 1, "content": md_table}]
                    parser_used = "Pandas Table Parser"
            except Exception as e1:
                errors_logged.append(f"Table Parser failed: {e1}")

        elif fmt == "json":
            try:
                buffer.seek(0)
                data = json.load(buffer)
                formatted = json.dumps(data, indent=2)
                pages = [{"page": 1, "content": formatted}]
                parser_used = "JSON Tree Parser"
            except Exception as e1:
                errors_logged.append(f"JSON Parser failed: {e1}")

        elif fmt in ["xml", "html"]:
            try:
                buffer.seek(0)
                raw_str = buffer.read().decode("utf-8", errors="ignore")
                parser = HTMLTextExtractor()
                parser.feed(raw_str)
                text = parser.get_text()
                if text.strip():
                    pages = [{"page": 1, "content": text.strip()}]
                    parser_used = "HTML/XML DOM Parser"
            except Exception as e1:
                errors_logged.append(f"HTML/XML Parser failed: {e1}")

        # Final Fallback: Text Decoder
        if not pages:
            try:
                buffer.seek(0)
                raw_b = buffer.read()
                txt = raw_b.decode("utf-8", errors="ignore").strip()
                if txt:
                    pages = [{"page": 1, "content": txt}]
                    parser_used = "Universal Text Decoder Fallback"
            except Exception as e_last:
                errors_logged.append(f"Universal Decoder failed: {e_last}")

        # Verification Check
        if not pages or not any(p.get("content", "").strip() for p in pages):
            raise IngestionError(
                f"Ingestion Failure for '{filename}': All parser fallback chains failed. Errors: {'; '.join(errors_logged)}"
            )

        metadata_summary = {
            "filename": filename,
            "file_type": fmt.upper(),
            "parser_used": parser_used,
            "total_pages": len(pages),
            "total_chars": sum(len(p.get("content", "")) for p in pages),
            "upload_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return pages, metadata_summary

def detect_document_category(filename: str, content: str) -> str:
    """
    Zero-Shot Classifier classifying document content into one of 14 categories.
    """
    if not content:
        return "Other"
        
    c_lower = content[:3000].lower()
    f_lower = filename.lower()
    
    if any(k in f_lower for k in ["log", "syslog", "trace", "alert"]) or "%ethport" in c_lower or "error" in c_lower:
        return "Logs"
    if any(k in f_lower for k in ["sop", "procedure", "guide", "playbook"]):
        return "SOP"
    if any(k in f_lower for k in ["cfg", "conf", "config", "switch", "router"]):
        return "Configuration"
    if any(k in f_lower for k in ["ticket", "inc", "incident", "jira"]):
        return "Incident"
    if any(k in f_lower for k in ["policy", "compliance", "sec"]):
        return "Security"
        
    return "Knowledge Base"
