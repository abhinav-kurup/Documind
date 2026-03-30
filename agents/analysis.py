from core.state import AgentState
from core.config import Config
from typing import Dict, Any
import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.helpers import log_agent_step

logger = logging.getLogger(__name__)

class AnalysisAgent:
    def __init__(self, model_identifier: str = None):
        from core.llm import get_llm
        model_identifier = model_identifier or Config.MODEL_NAME
        logger.info(f"AnalysisAgent initialized with Model: {model_identifier}")
        self.llm = get_llm(model_identifier, temperature=0.2)

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        from utils.helpers import dump_agent_state
        dump_agent_state(state, "AnalysisAgent")

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
            
            IMPORTANT RULES:
            - If [Extracted Data] is provided in the context, it contains the exact, pre-processed information the user requested. 
            - You MUST present this extracted data to the user directly and clearly.
            - Do not be overly literal; if the user asks for a "table" and the extracted data is JSON, format that JSON into a beautifully formatted Markdown table.
            - Do not claim information is missing if it is present in the [Extracted Data] section.
            
            CITATION RULES:
            - You must cite your sources using the format [Source X (Page Y)].
            - NEVER cite "[Extracted Data]" as a source. If you are using information from the [Extracted Data] block, cite the original document name provided above it in the context (e.g., [document.pdf (Page 1)]).
            
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
            
            log_agent_step(state, "AnalysisAgent", "Success", response_length=len(answer))
            
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
            
            log_agent_step(state, "AnalysisAgent", "Error", error=str(e))
            
            return {
                "final_response": "I apologize, but I encountered a system issue while analyzing the documents. Please check the Audit Logs or verify the AI model is correctly configured.",
                "audit_log": [{
                    "step": "AnalysisAgent", 
                    "status": "Error", 
                    "error": str(e)
                }]
            }
