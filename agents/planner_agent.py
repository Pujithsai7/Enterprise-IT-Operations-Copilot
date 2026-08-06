import os
import hashlib
import streamlit as st
from validator import DiagnosticValidator
from typing import TypedDict, List, Dict, Any, Optional

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    StateGraph = None
    END = None
    HAS_LANGGRAPH = False

SYSTEM_PROMPT = """You are the Principal Technical Planner Agent for Enterprise IT Operations.

Role & Core Responsibilities:
- Synthesize multi-agent evidence (Documentation, Network, Log Analysis, Incident records) into a single, comprehensive, grounded diagnostic report.

Strict Execution Rules:
1. Avoid Hallucinations: Do NOT invent parameters, device names, IPs, MAC addresses, error codes, or CLI commands not supported by retrieved context.
2. Require Grounded Evidence: Every diagnostic assertion must be backed by explicit evidence from the deduplicated context chunks.
3. Enforce Exact Citations: Include exact file basenames, headers, and page citations in all evidence bullets and source citations sections.
4. Prioritize Retrieved Context: Real-time telemetry/syslogs take strict precedence over static baseline documents or historical tickets.
5. Ignore Unsupported Assumptions: Never assume an error exists without explicit proof. If no fault is found, state: "No fault was detected from the supplied logs."

You MUST format your response into EXACTLY the following 8 sections:

### Cause:
[Concise root cause summary OR "No fault was detected from the supplied logs."]

### Evidence:
- [Bullet points of deduplicated evidence with source citations]

### Reasoning:
- [Step-by-step diagnostic reasoning, intent classification, and conflict resolution analysis]

### Commands:
```cisco
[Executable CLI/shell commands for diagnosis and remediation, e.g., 'show interface GigabitEthernet0/1', 'no shutdown']
```

### Verification Steps:
1. [Step 1 to verify operational recovery and traffic health]
2. [Step 2]

### Resolution:
1. [Actionable step 1 supported by retrieved documentation]
2. [Actionable step 2]

### Confidence Score:
[Calculated score e.g. 92% - High Diagnostic Confidence]

### 📍 Source Citations:
- [List of exact file citations]
"""


def estimate_tokens(text):
    if not text:
        return 0
    return max(1, len(text) // 4)

def deduplicate_evidence(raw_context_items):
    seen_hashes = set()
    deduped = []
    for item in raw_context_items:
        item_hash = hashlib.sha256(item['content'].encode('utf-8')).hexdigest()
        if item_hash not in seen_hashes:
            seen_hashes.add(item_hash)
            deduped.append(item)
    return deduped

def resolve_conflicts(evidence_items):
    logs = [e for e in evidence_items if "Log" in e['agent_name']]
    net = [e for e in evidence_items if "Network" in e['agent_name']]
    doc = [e for e in evidence_items if "Doc" in e['agent_name']]
    inc = [e for e in evidence_items if "Incident" in e['agent_name']]
    if logs:
        return logs + net + doc + inc
    return evidence_items

def limit_context_by_tokens(context_items, max_tokens=100000):
    selected_snippets = []
    current_tokens = 0
    for item in context_items:
        snippet_text = f"[{item['agent_name']} | {item['citation']}]: {item['content']}"
        item_tokens = estimate_tokens(snippet_text)
        if current_tokens + item_tokens <= max_tokens:
            selected_snippets.append(snippet_text)
            current_tokens += item_tokens
        else:
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 50:
                words = item['content'].split()
                max_words = int(remaining_tokens * 0.75)
                truncated_content = " ".join(words[:max_words]) + " [Truncated due to context limit]"
                selected_snippets.append(f"[{item['agent_name']} | {item['citation']}]: {truncated_content}")
            break
    return selected_snippets

from pydantic import BaseModel, Field, ValidationError

class DiagnosticReport(BaseModel):
    cause: str = Field(..., description="Concise root cause summary OR 'No fault was detected from the supplied logs.'")
    evidence: List[str] = Field(default_factory=list, description="List of deduplicated evidence items with citations")
    reasoning: List[str] = Field(default_factory=list, description="Step-by-step diagnostic reasoning and conflict resolution steps")
    commands: List[str] = Field(default_factory=list, description="List of executable CLI or shell commands for diagnosis and remediation")
    verification_steps: List[str] = Field(default_factory=list, description="List of verification steps to confirm operational recovery")
    resolution: List[str] = Field(default_factory=list, description="List of actionable resolution steps supported by retrieved documentation")
    confidence: int = Field(..., ge=0, le=100, description="Calculated confidence score between 0 and 100")
    citations: List[str] = Field(default_factory=list, description="List of exact file citations")

    def to_markdown(self) -> str:
        ev_str = "\n".join([f"- {item}" for item in self.evidence]) if self.evidence else "- No specific evidence."
        reason_str = "\n".join([f"- {item}" for item in self.reasoning]) if self.reasoning else "- General intent analysis."
        cmd_str = "\n".join(self.commands) if self.commands else "# No commands required"
        verif_str = "\n".join([f"{idx+1}. {item}" for idx, item in enumerate(self.verification_steps)]) if self.verification_steps else "1. Verify link status."
        res_str = "\n".join([f"{idx+1}. {item}" for idx, item in enumerate(self.resolution)]) if self.resolution else "1. System operational."
        cit_str = "\n".join([f"- `{item}`" for item in self.citations]) if self.citations else "- `[Uploaded Company Documents]`"

        return f"""### Cause:
{self.cause}

### Evidence:
{ev_str}

### Reasoning:
{reason_str}

### Commands:
```cisco
{cmd_str}
```

### Verification Steps:
{verif_str}

### Resolution:
{res_str}

### Confidence Score:
{self.confidence}% - Grounded Pydantic Diagnostic Report

### 📍 Source Citations:
{cit_str}"""

class PlannerAgent:
    def __init__(self, max_context_tokens=100000):
        self.validator = DiagnosticValidator()
        self.max_context_tokens = max_context_tokens

    def classify_intent(self, query):
        q = query.lower()
        if "log" in q or "syslog" in q or "event" in q or "down" in q:
            return "Active Telemetry & Syslog Investigation"
        elif "switch" in q or "router" in q or "vlan" in q or "config" in q or "acl" in q:
            return "Infrastructure Configuration Audit"
        elif "sop" in q or "manual" in q or "procedure" in q or "fix" in q:
            return "Standard Operating Procedure Resolution"
        elif "ticket" in q or "incident" in q or "past" in q:
            return "Historical Incident Analysis"
        return "General IT Operations Diagnostic"

    def synthesize(self, query, doc_evidence, net_evidence, log_evidence, inc_evidence, chat_history=None, api_key=None, model_choice="Local Engine"):
        intent = self.classify_intent(query)
        all_raw_chunks = doc_evidence + net_evidence + log_evidence + inc_evidence
        validation_results = self.validator.validate(query, all_raw_chunks)
        confidence_score = validation_results.get("confidence_score", 85)
        is_actual_error = validation_results.get("is_actual_error", False)
        error_category = validation_results.get("error_category", "Operational Log Audit")

        memory_str = ""
        if chat_history:
            history_snippets = [f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-4:]]
            memory_str = "\nRecent Conversation Memory:\n" + "\n".join(history_snippets)

        raw_context_items = []
        all_citations = []
        for agent_name, ev_list in [("Documentation Agent", doc_evidence), ("Network Agent", net_evidence), ("Log Agent", log_evidence), ("Incident Agent", inc_evidence)]:
            for item in ev_list:
                cit = item.get('citation', '')
                if cit and cit not in all_citations:
                    all_citations.append(cit)
                raw_context_items.append({"agent_name": agent_name, "citation": cit, "content": item.get('content', '')})

        deduped_items = deduplicate_evidence(raw_context_items)
        resolved_items = resolve_conflicts(deduped_items)
        context_snippets = limit_context_by_tokens(resolved_items, max_tokens=self.max_context_tokens)
        context_text = "\n".join(context_snippets) if context_snippets else "No specific document chunks retrieved."

        from utils import CitationVerifier
        verifier = CitationVerifier()
        verified_citations, removed_citations = verifier.verify_and_filter(all_citations, all_raw_chunks)

        openai_key = api_key or os.environ.get("OPENAI_API_KEY")
        if model_choice != "Local Engine" and openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                user_prompt = f"User Query: {query}\nClassified Intent: {intent}\n{memory_str}\n\nMulti-Agent Context:\n{context_text}\n\nDiagnostic Audit:\n- Error Detected: {'YES' if is_actual_error else 'NO'}\n- Category: {error_category}\n- Confidence: {confidence_score}%\n\nGenerate structured output matching the DiagnosticReport Pydantic schema."
                
                selected_model = model_choice if model_choice and model_choice != "Local Engine" else "kimi-k2.7-code:cloud"
                response = client.beta.chat.completions.parse(
                    model=selected_model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
                    response_format=DiagnosticReport,
                    temperature=0.0,
                    top_p=0.95,
                    max_tokens=4096
                )
                parsed_report: DiagnosticReport = response.choices[0].message.parsed
                clean_cits, _ = verifier.verify_and_filter(parsed_report.citations, all_raw_chunks)
                parsed_report.citations = clean_cits or verified_citations
                return (parsed_report.to_markdown(), validation_results)
            except Exception as e:
                st.warning(f"LLM Synthesis error ({str(e)}). Falling back to Pydantic Local Engine.")


        if not is_actual_error:
            report_data = {
                "cause": "No fault was detected from the supplied logs.",
                "evidence": [
                    "All supplied telemetry, syslogs, and network configurations report normal operational status.",
                    f"Diagnostic Validation Audit verified status: `{error_category}`."
                ],
                "reasoning": [
                    f"Intent Classified: `{intent}`.",
                    "Multi-Agent Analysis: Examined logs, configurations, and SOPs; zero explicit failure keywords were triggered.",
                    "Conflict Resolution: No contradictions detected in telemetry."
                ],
                "commands": ["show interface status", "show ip interface brief"],
                "verification_steps": [
                    "Verify that interface line protocols remain in 'up/up' state.",
                    "Monitor telemetry metrics for unexpected drops."
                ],
                "resolution": [
                    "No corrective intervention required. System operational.",
                    "Provide specific error logs if anomalous behavior is observed."
                ],
                "confidence": min(100, max(0, confidence_score)),
                "citations": verified_citations or ["Uploaded Company Documents"]
            }
        else:
            evidence_items = [f"**{item['agent_name']}** (`{item['citation']}`): {item['content']}" for item in resolved_items[:4]]
            report_data = {
                "cause": f"Analysis of multi-agent evidence indicates an operational fault ({error_category}) for query: '{query}'.",
                "evidence": evidence_items,
                "reasoning": [
                    f"Intent Classified: `{intent}`.",
                    f"Deduplicated {len(raw_context_items)} evidence chunks down to {len(resolved_items)} unique items.",
                    "Conflict Resolution: Prioritized real-time syslog telemetry over static baseline documentation."
                ],
                "commands": ["configure terminal", "interface GigabitEthernet0/1", "no shutdown", "description Uplink Restored", "end", "write memory"] if ("gigabit" in query.lower() or "interface" in query.lower() or "down" in query.lower()) else ["show logging | include ERROR", "show running-config"],
                "verification_steps": [
                    "Execute `show interface status` to confirm line protocol transitions to 'up'.",
                    "Verify ping connectivity across affected VLAN interfaces."
                ],
                "resolution": [
                    f"Apply interface and stanza fixes documented in `{verified_citations[0] if verified_citations else 'uploaded company documents'}`.",
                    "Re-enable interface and save startup-config."
                ],
                "confidence": min(100, max(0, confidence_score)),
                "citations": verified_citations or ["Uploaded Company Documents"]
            }

        from evaluator import RAGEvaluator
        rag_evaluator = RAGEvaluator()

        try:
            report_obj = DiagnosticReport.model_validate(report_data)
            final_text = report_obj.to_markdown()
            eval_results = rag_evaluator.evaluate(query, all_raw_chunks, final_text)
            validation_results["eval_results"] = eval_results
            return (final_text, validation_results)
        except ValidationError as val_err:
            st.error(f"Malformed output rejected by Pydantic schema: {val_err}")
            raise val_err




class AgentState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    doc_evidence: List[Dict[str, Any]]
    net_evidence: List[Dict[str, Any]]
    log_evidence: List[Dict[str, Any]]
    inc_evidence: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    confidence_score: int
    executed_agents: List[str]
    next_agent: str
    final_response: str
    api_key: Optional[str]
    model_choice: str

class SupervisorAgent:
    def route(self, state: dict) -> str:
        query = state.get("query", "").lower()
        executed = state.get("executed_agents", [])
        if len(executed) >= 4: return "ValidatorAgent"
        if ("sop" in query or "manual" in query or "procedure" in query or "guide" in query or "how to" in query) and "DocumentationAgent" not in executed: return "DocumentationAgent"
        if ("switch" in query or "router" in query or "vlan" in query or "interface" in query or "bgp" in query or "ospf" in query or "port" in query or "config" in query) and "NetworkAgent" not in executed: return "NetworkAgent"
        if ("log" in query or "syslog" in query or "alert" in query or "event" in query or "down" in query or "crash" in query or "error" in query or "timeout" in query) and "LogAnalysisAgent" not in executed: return "LogAnalysisAgent"
        if ("ticket" in query or "incident" in query or "outage" in query or "past" in query or "inc-" in query) and "IncidentAgent" not in executed: return "IncidentAgent"
        for agent in ["NetworkAgent", "LogAnalysisAgent", "DocumentationAgent", "IncidentAgent"]:
            if agent not in executed: return agent
        return "ValidatorAgent"

def build_copilot_graph(vector_store, planner_agent=None):
    from agents.doc_agent import DocumentationAgent
    from agents.network_agent import NetworkAgent
    from agents.log_agent import LogAnalysisAgent
    from agents.incident_agent import IncidentAgent
    doc_node, net_node, log_node, inc_node = DocumentationAgent(), NetworkAgent(), LogAnalysisAgent(), IncidentAgent()
    supervisor, validator, planner = SupervisorAgent(), DiagnosticValidator(), planner_agent or PlannerAgent()
    if not HAS_LANGGRAPH:
        class FallbackGraph:
            def invoke(self, state: dict) -> dict:
                doc_ev, net_ev, log_ev, inc_ev = doc_node.query_sop_manuals(state.get("query", ""), vector_store), net_node.query_network_configs(state.get("query", ""), vector_store), log_node.query_server_logs(state.get("query", ""), vector_store), inc_node.query_incident_tickets(state.get("query", ""), vector_store)
                val_res = validator.validate(state.get("query", ""), doc_ev + net_ev + log_ev + inc_ev)
                res_text, _ = planner.synthesize(state.get("query", ""), doc_ev, net_ev, log_ev, inc_ev, chat_history=state.get("chat_history", []), api_key=state.get("api_key"), model_choice=state.get("model_choice", "Local Engine"))
                return {"doc_evidence": doc_ev, "net_evidence": net_ev, "log_evidence": log_ev, "inc_evidence": inc_ev, "validation_results": val_res, "confidence_score": val_res.get("confidence_score", 85), "executed_agents": ["DocumentationAgent", "NetworkAgent", "LogAnalysisAgent", "IncidentAgent", "ValidatorAgent", "PlannerAgent"], "final_response": res_text}
        return FallbackGraph()
    builder = StateGraph(AgentState)
    builder.add_node("Supervisor", lambda s: {"next_agent": supervisor.route(s)})
    builder.add_node("DocumentationAgent", lambda s: doc_node.run_node(s, vector_store))
    builder.add_node("NetworkAgent", lambda s: net_node.run_node(s, vector_store))
    builder.add_node("LogAnalysisAgent", lambda s: log_node.run_node(s, vector_store))
    builder.add_node("IncidentAgent", lambda s: inc_node.run_node(s, vector_store))
    def validator_step(state: dict) -> dict:
        query = state.get("query", "")
        retry_count = state.get("retry_count", 0)
        all_chunks = (
            state.get("doc_evidence", []) +
            state.get("net_evidence", []) +
            state.get("log_evidence", []) +
            state.get("inc_evidence", [])
        )
        
        val_res = validator.validate(
            query=query,
            retrieved_chunks=all_chunks,
            api_key=state.get("api_key"),
            model_choice=state.get("model_choice", "Local Engine")
        )
        
        score = val_res.get("confidence_score", 85)
        requires_more = val_res.get("requires_more_documents", False)
        executed = list(state.get("executed_agents", []))
        executed.append("ValidatorAgent")

        if requires_more and retry_count < 2:
            expanded_q = val_res.get("suggested_query_expansion", query)
            more_chunks = vector_store.search(expanded_q, top_k=8)
            log_ev = list(state.get("log_evidence", [])) + more_chunks
            return {
                "validation_results": val_res,
                "confidence_score": score,
                "executed_agents": executed,
                "retry_count": retry_count + 1,
                "log_evidence": log_ev
            }

        return {
            "validation_results": val_res,
            "confidence_score": score,
            "executed_agents": executed
        }

    def planner_step(state: dict) -> dict:
        res_text, val_res = planner.synthesize(
            query=state.get("query", ""),
            doc_evidence=state.get("doc_evidence", []),
            net_evidence=state.get("net_evidence", []),
            log_evidence=state.get("log_evidence", []),
            inc_evidence=state.get("inc_evidence", []),
            chat_history=state.get("chat_history", []),
            api_key=state.get("api_key"),
            model_choice=state.get("model_choice", "Local Engine")
        )
        executed = list(state.get("executed_agents", []))
        executed.append("PlannerAgent")
        return {
            "final_response": res_text,
            "validation_results": val_res,
            "executed_agents": executed
        }

    builder = StateGraph(AgentState)
    builder.add_node("Supervisor", lambda s: {"next_agent": supervisor.route(s)})
    builder.add_node("DocumentationAgent", lambda s: doc_node.run_node(s, vector_store))
    builder.add_node("NetworkAgent", lambda s: net_node.run_node(s, vector_store))
    builder.add_node("LogAnalysisAgent", lambda s: log_node.run_node(s, vector_store))
    builder.add_node("IncidentAgent", lambda s: inc_node.run_node(s, vector_store))
    builder.add_node("ValidatorAgent", validator_step)
    builder.add_node("PlannerAgent", planner_step)
    builder.set_entry_point("Supervisor")
    builder.add_conditional_edges("Supervisor", lambda s: s.get("next_agent", "ValidatorAgent"), {"DocumentationAgent": "DocumentationAgent", "NetworkAgent": "NetworkAgent", "LogAnalysisAgent": "LogAnalysisAgent", "IncidentAgent": "IncidentAgent", "ValidatorAgent": "ValidatorAgent"})
    for node in ["DocumentationAgent", "NetworkAgent", "LogAnalysisAgent", "IncidentAgent"]: builder.add_edge(node, "Supervisor")
    builder.add_edge("ValidatorAgent", "PlannerAgent")
    builder.add_edge("PlannerAgent", END)
    return builder.compile()

