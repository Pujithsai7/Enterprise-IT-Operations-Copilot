class IncidentAgent:
    """
    Incident Agent Node: Independent ITSM Ticket & Resolution History Specialist.
    Features:
    - Independent system prompt & historical resolution analysis
    - Independent vector search tooling (query_incident_tickets)
    - Independent graph node execution reasoning
    """
    SYSTEM_PROMPT = """You are the Enterprise Incident Agent. 
Your task is to search past incident tickets, outage summaries, and resolution logs to find historical fixes for similar symptoms."""

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

