import re

ERROR_KEYWORDS = [
    "error", "failed", "failure", "critical", "timeout", "denied",
    "down", "unreachable", "crc", "drop", "panic", "invalid",
    "exception", "err-disable", "link-flap", "hogging", "crash",
    "disconnect", "offline", "degraded", "reboot", "alarm"
]

class DiagnosticValidator:
    """
    Diagnostic Validation Layer: Evaluates retrieved FAISS chunks before synthesis.
    Performs 4 validation checks:
    1. Is there an actual error in retrieved chunks? (is_actual_error)
    2. Is evidence sufficient? (is_evidence_sufficient)
    3. Confidence calculation (confidence_score)
    4. Error category determination (error_category)
    """
    def validate(self, query, retrieved_chunks):
        if not retrieved_chunks:
            return {
                "is_actual_error": False,
                "is_evidence_sufficient": False,
                "confidence_score": 25,
                "error_category": "Insufficient Evidence",
                "matched_keywords": []
            }

        # 1. Error Keyword Scan ONLY on retrieved document chunks
        chunks_text = " ".join([c.get('content', '') for c in retrieved_chunks]).lower()
        matched_keywords = [kw for kw in ERROR_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', chunks_text)]
        is_actual_error = len(matched_keywords) > 0

        # 2. Evidence Sufficiency Check
        total_length = sum(len(c.get('content', '')) for c in retrieved_chunks)
        is_evidence_sufficient = total_length >= 80

        # 3. Confidence Calculation
        scores = [c.get('score', 0.0) for c in retrieved_chunks if 'score' in c]
        top_score = max(scores) if scores else 0.0

        if not is_actual_error:
            confidence_score = 25
        elif not is_evidence_sufficient:
            confidence_score = 45
        elif len(matched_keywords) >= 2 and top_score > 0.4:
            confidence_score = int(min(98, max(90, 85 + top_score * 20)))
        elif len(matched_keywords) >= 1 or top_score > 0.3:
            confidence_score = int(min(89, max(75, 68 + top_score * 25)))
        else:
            confidence_score = 55

        # 4. Error Category Determination
        error_category = self._determine_category(is_actual_error, chunks_text)

        return {
            "is_actual_error": is_actual_error,
            "is_evidence_sufficient": is_evidence_sufficient,
            "confidence_score": confidence_score,
            "error_category": error_category,
            "matched_keywords": matched_keywords
        }

    def _determine_category(self, is_actual_error, text):
        if not is_actual_error:
            return "No Error Detected (Operational Status Normal)"
            
        categories = []
        if any(kw in text for kw in ["switch", "router", "vlan", "interface", "port", "err-disable", "link-flap", "link-down"]):
            categories.append("Network Interface / Topology Failure")
        if any(kw in text for kw in ["syslog", "alert", "cpu", "memory", "hogging", "process", "crash"]):
            categories.append("Syslog / Telemetry Alert")
        if any(kw in text for kw in ["ticket", "inc-", "outage", "past"]):
            categories.append("Historical Incident Record")
        if any(kw in text for kw in ["sop", "manual", "guide", "procedure"]):
            categories.append("SOP Recovery Procedure")
            
        return categories[0] if categories else "General Technical Fault"
