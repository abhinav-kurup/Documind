from core.state import AgentState
from core.config import Config
from typing import Dict, Any
import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

logger = logging.getLogger(__name__)

class ExtractionAgent:
    def __init__(self, model_name: str = None):
        model_name = model_name or Config.MODEL_NAME
        base_url = Config.OLLAMA_BASE_URL
        self.llm = ChatOllama(
            model=model_name, 
            base_url=base_url, 
            temperature=0,
            timeout=30  
        )

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        logger.info("ExtractionAgent: Process started")
        query = state.get("query", "")
        docs = state.get("retrieved_docs", [])
        
        if not docs:
            self._log_skip(state, "No docs found")
            return {"audit_log": [{"step": "ExtractionAgent", "status": "Skipped", "reason": "No docs found"}]}

        if "extract" not in query.lower() and "table" not in query.lower():
            self._log_skip(state, "Query does not request extraction")
            return {"audit_log": [{"step": "ExtractionAgent", "status": "Skipped", "reason": "Query does not request extraction"}]}

        context = "\n\n".join([d.get("content", "") for d in docs])
        
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert data extraction agent.
            Extract the specific information requested in the query from the context below.
            Output the result as a structured JSON or Markdown table.
            
            Query: {query}
            
            Context:
            {context}
            
            Extracted Data:
            """
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({"query": query, "context": context})
            
            # Progressive logging
            audit_logger = state.get("audit_logger")
            query_id = state.get("query_id")
            if audit_logger and query_id:
                audit_logger.log_step(query_id, "ExtractionAgent", "Success", extracted_length=len(result))
            
            return {
                "extracted_data": {"content": result},
                "audit_log": [{
                    "step": "ExtractionAgent", 
                    "status": "Success", 
                    "extracted_length": len(result)
                }]
            }
        except Exception as e:
            logger.error(f"ExtractionAgent Error: {e}")
            audit_logger = state.get("audit_logger")
            query_id = state.get("query_id")
            if audit_logger and query_id:
                audit_logger.log_step(query_id, "ExtractionAgent", "Error", error=str(e))
            return {
                "audit_log": [{
                    "step": "ExtractionAgent", 
                    "status": "Error", 
                    "error": str(e)
                }]
            }
    
    def _log_skip(self, state: AgentState, reason: str):
        audit_logger = state.get("audit_logger")
        query_id = state.get("query_id")
        if audit_logger and query_id:
            audit_logger.log_step(query_id, "ExtractionAgent", "Skipped", reason=reason)
