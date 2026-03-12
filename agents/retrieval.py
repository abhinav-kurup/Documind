from core.state import AgentState
from utils.helpers import log_agent_step
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class RetrievalAgent:
    def __init__(self,vector_store):
        self.vector_store = vector_store

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
            
            log_agent_step(
                state=state,
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
