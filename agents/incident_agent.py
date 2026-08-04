class IncidentAgent:
    """
    Incident Agent: Uses FAISS embeddings to retrieve Incident Ticket & Past Resolution chunks.
    """
    def execute(self, query, vector_store, top_k=4):
        source_types = ['Incident Ticket', 'Past Resolution']
        results = vector_store.search(query, source_types=source_types, top_k=top_k)
        return results
