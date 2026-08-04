class DocumentationAgent:
    """
    Documentation Agent: Uses FAISS embeddings to retrieve SOP and Equipment Manual chunks.
    """
    def execute(self, query, vector_store, top_k=4):
        source_types = ['SOP / Manual', 'SOP', 'Equipment Manual']
        results = vector_store.search(query, source_types=source_types, top_k=top_k)
        return results
