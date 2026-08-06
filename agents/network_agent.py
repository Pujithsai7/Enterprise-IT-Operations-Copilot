class NetworkAgent:
    """
    Network Agent Node: Independent Cisco Network Infrastructure & Topology Specialist.
    Features:
    - Independent system prompt & Cisco IOS / switch / router expertise
    - Independent vector search tooling (query_network_configs)
    - Independent graph node execution reasoning
    """
    SYSTEM_PROMPT = """You are the Senior Enterprise Network Infrastructure Agent.

Role & Core Responsibilities:
- Inspect Cisco IOS configurations, switch/router stanzas, VLAN definitions, ACL access lists, and BGP/OSPF routing topologies to diagnose network faults.

Strict Execution Rules:
1. Avoid Hallucinations: Never invent interface names (e.g. GigabitEthernet0/1), VLAN IDs, IP subnets, or CLI commands not present in the retrieved configurations.
2. Require Grounded Evidence: Base all topology findings directly on verified configuration stanzas.
3. Enforce Exact Citations: Provide exact file basenames, interface headers, and section names for every configuration evidence snippet.
4. Prioritize Retrieved Context: Give strict priority to the retrieved device configs over standard generic networking defaults.
5. Ignore Unsupported Assumptions: Do not assume a port or VLAN is misconfigured without explicit stanza evidence."""


    def query_network_configs(self, query, vector_store, top_k=4):
        source_types = ['Network Configuration', 'Topology']
        results = vector_store.search(query, source_types=source_types, top_k=top_k)
        return results

    def run_node(self, state: dict, vector_store) -> dict:
        query = state.get("query", "")
        evidence = self.query_network_configs(query, vector_store)
        executed = list(state.get("executed_agents", []))
        if "NetworkAgent" not in executed:
            executed.append("NetworkAgent")
            
        return {
            "net_evidence": evidence,
            "executed_agents": executed
        }

