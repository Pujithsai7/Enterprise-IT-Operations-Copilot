class LogAnalysisAgent:
    """
    Log Analysis Agent: Uses FAISS embeddings to retrieve Server Log & Alert chunks.
    """
    def execute(self, query, vector_store, top_k=4):
        source_types = ['Server Log', 'Server Log / Alert']
        results = vector_store.search(query, source_types=source_types, top_k=top_k)
        return results
