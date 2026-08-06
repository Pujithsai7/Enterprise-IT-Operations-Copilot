import pytest
from agents import (
    DocumentationAgent,
    NetworkAgent,
    LogAnalysisAgent,
    IncidentAgent,
    SupervisorAgent,
    PlannerAgent,
    build_copilot_graph
)
from agents.planner_agent import DiagnosticReport, deduplicate_evidence, resolve_conflicts
from validator import LLMValidatorAgent
from utils import CitationVerifier
from pydantic import ValidationError

def test_domain_agents_execution(vector_store):
    doc_node = DocumentationAgent()
    net_node = NetworkAgent()
    log_node = LogAnalysisAgent()
    inc_node = IncidentAgent()

    state = {"query": "Interface GigabitEthernet0/1 down", "executed_agents": []}
    
    res_doc = doc_node.run_node(state, vector_store)
    assert "doc_evidence" in res_doc
    
    res_net = net_node.run_node(state, vector_store)
    assert "net_evidence" in res_net

def test_supervisor_agent_routing():
    supervisor = SupervisorAgent()
    state = {"query": "How to fix interface down in SOP manual?", "executed_agents": []}
    next_agent = supervisor.route(state)
    assert next_agent in ["DocumentationAgent", "NetworkAgent", "LogAnalysisAgent", "IncidentAgent", "ValidatorAgent"]

def test_pydantic_diagnostic_report_schema():
    valid_data = {
        "cause": "Interface down due to shutdown command",
        "evidence": ["cisco_switch.cfg: shutdown"],
        "reasoning": ["Intent: Infrastructure Config Audit"],
        "commands": ["no shutdown"],
        "verification_steps": ["Check line protocol"],
        "resolution": ["Enable port with no shutdown"],
        "confidence": 95,
        "citations": ["cisco_switch.cfg | Page #1"]
    }
    report = DiagnosticReport.model_validate(valid_data)
    assert report.confidence == 95
    assert "### Cause:" in report.to_markdown()

    invalid_data = {"confidence": 80}
    with pytest.raises(ValidationError):
        DiagnosticReport.model_validate(invalid_data)

def test_llm_validator_agent():
    validator = LLMValidatorAgent()
    chunks = [{
        "citation": "[cisco_switch.cfg | Page #1]",
        "document_type": "Network Configuration",
        "content": "interface GigabitEthernet0/1\n shutdown",
        "score": 0.90
    }]
    res = validator.validate("Interface GigabitEthernet0/1 down", chunks)
    assert "confidence_score" in res
    assert "error_category" in res

def test_citation_verifier():
    verifier = CitationVerifier()
    chunks = [{
        "filename": "cisco_switch.cfg",
        "page": 1,
        "section": "Global",
        "chunk_id": "1",
        "citation": "[cisco_switch.cfg | Page #1 | Chunk #1]"
    }]
    valid_cits = ["[cisco_switch.cfg | Page #1 | Chunk #1]"]
    invalid_cits = ["[fake_doc.pdf | Page #99]"]

    v_ok, r_ok = verifier.verify_and_filter(valid_cits, chunks)
    assert len(v_ok) == 1

    v_bad, r_bad = verifier.verify_and_filter(invalid_cits, chunks)
    assert len(r_bad) == 1
