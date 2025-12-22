from core.state import AgentState
from core.config import Config
from typing import Dict, Any
import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

logger = logging.getLogger(__name__)

class AnalysisAgent:
    def __init__(self, model_name: str = None):
        model_name = model_name or Config.MODEL_NAME
        base_url = Config.OLLAMA_BASE_URL
        logger.info(f"AnalysisAgent initialized with Base URL: {base_url}, Model: {model_name}")
        self.llm = ChatOllama(
            model=model_name, 
            base_url=base_url, 
            temperature=0.2,
            timeout=30  
        )

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        logger.info("AnalysisAgent: Process started")
        query = state.get("query", "")
        docs = state.get("retrieved_docs", [])
        extracted = state.get("extracted_data", {}).get("content", "")
        
        context_parts = []
        for i, doc in enumerate(docs):
            filename = doc['metadata'].get('source', f'Document {i+1}')
            page = doc['metadata'].get('page_number', '?')
            source = f"{filename} (Page {page})"
            context_parts.append(f"[{source}]: {doc.get('content', '')}")
            
        context_str = "\n\n".join(context_parts)
        
        if extracted:
            context_str += f"\n\n[Extracted Data]:\n{extracted}"

        prompt = ChatPromptTemplate.from_template(
            """
            You are an intelligent document analysis assistant. 
            Answer the user's query based ONLY on the provided context.
            If the answer is not in the context, state that you don't know.
            
            You must cite your sources using the format [Source X (Page Y)].
            
            Query: {query}
            
            Context:
            {context}
            
            Answer:
            """
        )

        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            logger.info("AnalysisAgent: Invoking LLM for generation...")
            answer = chain.invoke({"query": query, "context": context_str})
            logger.info("AnalysisAgent: Generation successful")
            
            audit_logger = state.get("audit_logger")
            query_id = state.get("query_id")
            if audit_logger and query_id:
                audit_logger.log_step(query_id, "AnalysisAgent", "Success", response_length=len(answer))
            
            return {
                "final_response": answer,
                "audit_log": [{
                    "step": "AnalysisAgent", 
                    "status": "Success", 
                    "response_length": len(answer)
                }]
            }
        except Exception as e:
            logger.error(f"AnalysisAgent Error: {e}")
            
            # Progressive logging for errors
            audit_logger = state.get("audit_logger")
            query_id = state.get("query_id")
            if audit_logger and query_id:
                audit_logger.log_step(query_id, "AnalysisAgent", "Error", error=str(e))
            
            return {
                "final_response": f"I encountered an error during analysis: {str(e)}. \n\nEnsure your LLM (Ollama) is running and accessible.",
                "audit_log": [{
                    "step": "AnalysisAgent", 
                    "status": "Error", 
                    "error": str(e)
                }]
            }
