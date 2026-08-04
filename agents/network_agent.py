class NetworkAgent:
    """
    Network Agent: Uses FAISS embeddings to retrieve Network Configuration & Topology chunks.
    """
    def execute(self, query, vector_store, top_k=4):
        source_types = ['Network Configuration', 'Topology']
        results = vector_store.search(query, source_types=source_types, top_k=top_k)
        return results
