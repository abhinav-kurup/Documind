from vectorstore.chroma import VectorStoreManager
from core.state import AgentState
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class RetrievalAgent:
    def __init__(self):
        self.vector_store = VectorStoreManager()

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        if not query:
            return {"audit_log": [{"step": "RetrievalAgent", "status": "Skipped"}]}

        logger.info(f"RetrievalAgent: Searching for '{query}'")
        
        try:
            results = self.vector_store.similarity_search(query, k=5)
            
            docs = []
            for doc in results:
                docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            logger.info(f"RetrievalAgent: Found {len(docs)} documents")
            
            audit_logger = state.get("audit_logger")
            query_id = state.get("query_id")
            if audit_logger and query_id:
                audit_logger.log_step(
                    query_id=query_id,
                    step_name="RetrievalAgent",
                    status="Success",
                    retrieved_count=len(docs),
                    query=query
                )

            return {
                "retrieved_docs": docs,
                "audit_log": [{
                    "step": "RetrievalAgent", 
                    "status": "Success", 
                    "retrieved_count": len(docs),
                    "query": query
                }]
            }
        except Exception as e:
            logger.error(f"RetrievalAgent Error: {e}")
            return {
                "audit_log": [{
                    "step": "RetrievalAgent", 
                    "status": "Error", 
                    "error": str(e)
                }]
            }
