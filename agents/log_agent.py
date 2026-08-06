class LogAnalysisAgent:
    """
    Log Analysis Agent Node: Independent Syslog & Telemetry Trace Specialist.
    Features:
    - Independent system prompt & syslog trace analysis
    - Independent vector search tooling (query_server_logs)
    - Independent graph node execution reasoning
    """
    SYSTEM_PROMPT = """You are the Senior Enterprise Log Analysis & Telemetry Agent.

Role & Core Responsibilities:
- Scan system logs, alert messages, telemetry traces, and syslog event codes (e.g. %ETHPORT-5-IF_DOWN, %LINK-3-UPDOWN) to detect real-time operational faults.

Strict Execution Rules:
1. Avoid Hallucinations: Never invent timestamps, syslog error codes, process IDs, or log messages not present in the retrieved telemetry files.
2. Require Grounded Evidence: Base all alert findings strictly on exact syslog entries and telemetry strings.
3. Enforce Exact Citations: Include exact log filenames, timestamps, and line citations for every detected alert.
4. Prioritize Retrieved Context: Real-time telemetry log events take strict precedence over static baseline documents.
5. Ignore Unsupported Assumptions: Do not declare a system down unless explicit error logs or state-change entries confirm it."""


    def query_server_logs(self, query, vector_store, top_k=4):
        source_types = ['Server Log', 'Server Log / Alert']
        results = vector_store.search(query, source_types=source_types, top_k=top_k)
        return results

    def run_node(self, state: dict, vector_store) -> dict:
        query = state.get("query", "")
        evidence = self.query_server_logs(query, vector_store)
        executed = list(state.get("executed_agents", []))
        if "LogAnalysisAgent" not in executed:
            executed.append("LogAnalysisAgent")
            
        return {
            "log_evidence": evidence,
            "executed_agents": executed
        }

