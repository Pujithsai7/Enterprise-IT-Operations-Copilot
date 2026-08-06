class IncidentAgent:
    """
    Incident Agent Node: Independent ITSM Ticket & Resolution History Specialist.
    Features:
    - Independent system prompt & historical resolution analysis
    - Independent vector search tooling (query_incident_tickets)
    - Independent graph node execution reasoning
    """
    SYSTEM_PROMPT = """You are the Senior Enterprise ITSM Incident Resolution Agent.

Role & Core Responsibilities:
- Search ITSM ticket logs, historical outage summaries, and incident resolution records to identify proven historical fixes for reported symptoms.

Strict Execution Rules:
1. Avoid Hallucinations: Never invent ticket IDs (e.g. INC-99412), resolution actions, or past root cause summaries not present in retrieved ticket data.
2. Require Grounded Evidence: Base all historical fix recommendations strictly on verified ticket resolution passages.
3. Enforce Exact Citations: Provide exact ticket filenames, ticket IDs, and resolution log citations for every retrieved fix.
4. Prioritize Retrieved Context: Rely on company incident records over generic IT troubleshooting steps.
5. Ignore Unsupported Assumptions: Do not assume a past resolution applies unless symptom patterns and device context explicitly match."""


    def query_incident_tickets(self, query, vector_store, top_k=4):
        source_types = ['Incident Ticket', 'Past Resolution']
        results = vector_store.search(query, source_types=source_types, top_k=top_k)
        return results

    def run_node(self, state: dict, vector_store) -> dict:
        query = state.get("query", "")
        evidence = self.query_incident_tickets(query, vector_store)
        executed = list(state.get("executed_agents", []))
        if "IncidentAgent" not in executed:
            executed.append("IncidentAgent")
            
        return {
            "inc_evidence": evidence,
            "executed_agents": executed
        }

