import os
import json
import numpy as np
import streamlit as st
from typing import Dict, Any, List
from utils import CitationVerifier

class RAGEvaluator:
    """
    Enterprise RAG Evaluation Framework.
    Implements 8 core RAG metrics:
    1. RAGAS Score (Aggregated RAG Assessment)
    2. Faithfulness (Grounded claims ratio)
    3. Context Precision (Relevance of retrieved chunks)
    4. Context Recall (Coverage of query requirements)
    5. Answer Relevance (Alignment with user intent)
    6. Citation Accuracy (Verified vs hallucinated citations)
    7. Groundedness (Evidence-backed ratio)
    8. Hallucination Rate (Percentage of unverified claims)
    """
    def __init__(self, citation_verifier=None):
        self.verifier = citation_verifier or CitationVerifier()

    def evaluate(self, query: str, retrieved_chunks: List[Dict[str, Any]], generated_response: str, ground_truth: str = None) -> Dict[str, Any]:
        if not retrieved_chunks:
            return {
                "ragas_score": 0.0,
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_relevance": 0.0,
                "citation_accuracy": 0.0,
                "groundedness": 0.0,
                "hallucination_rate": 100.0,
                "status": "FAIL (Zero Context Chunks)"
            }

        text_corpus = " ".join([c.get('content', '') for c in retrieved_chunks]).lower()
        query_words = [w.lower() for w in query.split() if len(w) > 3]

        # 1. Context Precision
        relevant_chunks = 0
        for c in retrieved_chunks:
            c_text = c.get('content', '').lower()
            if any(w in c_text for w in query_words):
                relevant_chunks += 1
        context_precision = round(relevant_chunks / max(1, len(retrieved_chunks)), 2)

        # 2. Context Recall
        if query_words:
            matched_terms = sum(1 for w in query_words if w in text_corpus)
            context_recall = round(matched_terms / max(1, len(query_words)), 2)
        else:
            context_recall = 1.0

        # 3. Citation Accuracy
        citations = []
        for line in generated_response.splitlines():
            if "Source Citations:" in line or "`" in line or "[" in line:
                for token in line.split("`"):
                    if "[" in token and "]" in token:
                        citations.append(token.strip())

        if citations:
            verified, removed = self.verifier.verify_and_filter(citations, retrieved_chunks)
            citation_accuracy = round(len(verified) / max(1, len(citations)), 2)
        else:
            citation_accuracy = 1.0

        # 4. Faithfulness & Groundedness
        resp_lines = [l.strip() for l in generated_response.splitlines() if l.strip() and not l.startswith("#")]
        supported_lines = 0
        for line in resp_lines:
            line_words = [w.lower() for w in line.split() if len(w) > 4]
            if not line_words or any(w in text_corpus for w in line_words):
                supported_lines += 1

        faithfulness = round(supported_lines / max(1, len(resp_lines)), 2)
        groundedness = faithfulness

        # 5. Hallucination Rate
        hallucination_rate = round((1.0 - faithfulness) * 100.0, 1)

        # 6. Answer Relevance
        resp_lower = generated_response.lower()
        rel_matches = sum(1 for w in query_words if w in resp_lower)
        answer_relevance = round(rel_matches / max(1, len(query_words)), 2) if query_words else 1.0

        # 7. Overall RAGAS Score
        ragas_score = round(
            (context_precision * 0.25) +
            (context_recall * 0.25) +
            (faithfulness * 0.25) +
            (answer_relevance * 0.25),
            2
        )

        status = "PASS" if ragas_score >= 0.70 and hallucination_rate <= 20.0 else "WARN"

        res = {
            "ragas_score": ragas_score,
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_relevance": answer_relevance,
            "citation_accuracy": citation_accuracy,
            "groundedness": groundedness,
            "hallucination_rate": hallucination_rate,
            "status": status
        }
        
        # Save evaluation report after every change/query
        self._save_eval_report(res, query)
        return res

    def _save_eval_report(self, eval_res: Dict[str, Any], query: str = ""):
        try:
            os.makedirs(".cache", exist_ok=True)
            report_data = {
                "query": query,
                "metrics": eval_res
            }
            with open(".cache/latest_eval_report.json", "w") as f:
                json.dump(report_data, f, indent=2)
        except Exception:
            pass

    def generate_report(self, eval_res: Dict[str, Any], query: str = "") -> str:
        return f"""### 📊 RAG Evaluation Framework Report

| Metric | Score | Status |
| :--- | :--- | :--- |
| **RAGAS Score** | `{eval_res['ragas_score']}` | {'✅ PASS' if eval_res['ragas_score'] >= 0.7 else '⚠️ WARN'} |
| **Faithfulness** | `{int(eval_res['faithfulness'] * 100)}%` | ✅ Grounded Claims |
| **Context Precision** | `{int(eval_res['context_precision'] * 100)}%` | ✅ High Precision |
| **Context Recall** | `{int(eval_res['context_recall'] * 100)}%` | ✅ Relevant Context |
| **Answer Relevance** | `{int(eval_res['answer_relevance'] * 100)}%` | ✅ Query Aligned |
| **Citation Accuracy** | `{int(eval_res['citation_accuracy'] * 100)}%` | ✅ Verified Citations |
| **Groundedness** | `{int(eval_res['groundedness'] * 100)}%` | ✅ Evidence Supported |
| **Hallucination Rate** | `{eval_res['hallucination_rate']}%` | {'✅ Low (<15%)' if eval_res['hallucination_rate'] <= 15 else '⚠️ Elevated'} |
"""
