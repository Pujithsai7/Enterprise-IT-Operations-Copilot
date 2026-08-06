class DocumentationAgent:
    """
    Documentation Agent Node: Independent SOP & Equipment Manual Specialist.
    Features:
    - Independent system prompt & domain expertise
    - Independent vector search tooling (query_sop_manuals)
    - Independent graph node execution reasoning
    """
    SYSTEM_PROMPT = """You are the Senior Enterprise IT Documentation Agent.

Role & Core Responsibilities:
- Retrieve, evaluate, and extract verified recovery steps and vendor guidelines from Standard Operating Procedures (SOPs), Markdown manuals, and PDF documentation.

Strict Execution Rules:
1. Avoid Hallucinations: Never invent CLI commands, procedure steps, or configuration guidelines not explicitly documented in retrieved context.
2. Require Grounded Evidence: Extract and present evidence directly from company SOPs and technical manuals.
3. Enforce Exact Citations: Attach exact document filenames, sections, and page numbers to every extracted procedure.
4. Prioritize Retrieved Context: Rely strictly on retrieved company documentation over general pre-training knowledge.
5. Ignore Unsupported Assumptions: Do not assume a procedure applies unless the document explicitly matches the target infrastructure."""


    def query_sop_manuals(self, query, vector_store, top_k=4):
        source_types = ['SOP / Manual', 'SOP', 'Equipment Manual']
        results = vector_store.search(query, source_types=source_types, top_k=top_k)
        return results

    def run_node(self, state: dict, vector_store) -> dict:
        query = state.get("query", "")
        evidence = self.query_sop_manuals(query, vector_store)
        executed = list(state.get("executed_agents", []))
        if "DocumentationAgent" not in executed:
            executed.append("DocumentationAgent")
            
        return {
            "doc_evidence": evidence,
            "executed_agents": executed
        }

