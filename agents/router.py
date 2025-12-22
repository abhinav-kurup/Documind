from core.state import AgentState
from core.config import Config
from typing import Dict, Any
import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class RouterAgent:
    """
    Routes queries using a micro LLM classifier to determine if they need 
    document retrieval or should be rejected as out-of-scope.
    """
    
    def __init__(self):
        self.llm = ChatOllama(
            model=Config.MODEL_NAME,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0, 
            timeout=10  
        )
        
    # ============================================================================
    # KEYWORD-BASED FALLBACK (COMMENTED - Uncomment if LLM classification fails)
    # ============================================================================
    
    # # High-confidence conversational patterns (greetings, pleasantries)
    # CONVERSATIONAL_KEYWORDS = [
    #     'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
    #     'how are you', 'how do you do', 'nice to meet', 'pleased to meet',
    #     'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'talk to you later',
    #     'who are you', 'what is your name', 'introduce yourself'
    # ]
    # 
    # # Strong document-related keywords
    # STRONG_DOCUMENT_KEYWORDS = [
    #     'revenue', 'profit', 'loss', 'sales', 'income', 'expenses', 'costs',
    #     'earnings', 'margin', 'growth', 'decline', 'increase', 'decrease',
    #     'report', 'document', 'page', 'section', 'chapter', 'paragraph',
    #     'table', 'chart', 'figure', 'graph', 'diagram', 'image',
    #     'according to', 'mentioned', 'stated', 'written', 'says', 'contains',
    #     'extract', 'summarize', 'summary', 'analyze', 'analysis', 'compare',
    #     'quarter', 'year', 'fy', 'q1', 'q2', 'q3', 'q4', 'fiscal'
    # ]
    # 
    # # Question words that need context (weak indicators)
    # WEAK_DOCUMENT_KEYWORDS = [
    #     'what', 'when', 'where', 'who', 'how', 'why', 'which',
    #     'show', 'tell', 'find', 'get', 'give', 'list', 'explain', 'describe'
    # ]
    # 
    # # Meta/system questions (should be rejected)
    # META_KEYWORDS = [
    #     'what can you do', 'what are your capabilities', 'how do you work',
    #     'help', 'how to use', 'instructions', 'guide'
    # ]
    
    # ============================================================================
    
    def invoke(self, state: AgentState) -> Dict[str, Any]:
        """
        Classifies query using LLM into: 'document' or 'conversational'
        """
        query = state.get("query", "").strip()
        
        # LLM Classification
        prompt = ChatPromptTemplate.from_template(
            """You are a query classifier for a document analysis system.

Classify the user's query into ONE of these categories:

1. "document" - Questions about specific information, data, or analysis that would require searching through documents
   Examples: "What is the revenue?", "Summarize page 5", "Extract sales data"

2. "conversational" - Greetings, pleasantries, or questions about the system itself (not about documents)
   Examples: "Hello", "How are you?", "What can you do?", "Thanks"

Query: "{query}"

Respond with ONLY one word: "document" or "conversational"
"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            logger.info(f"RouterAgent: Classifying query with LLM...")
            classification = chain.invoke({"query": query}).strip().lower()
            
            # Parse LLM response
            if "document" in classification:
                route = "document"
            elif "conversational" in classification:
                route = "conversational"
            else:
                logger.warning(f"Unclear classification '{classification}', defaulting to 'document'")
                route = "document"
            
            logger.info(f"RouterAgent: LLM classified as '{route}'")
            
            # Progressive logging
            audit_logger = state.get("audit_logger")
            query_id = state.get("query_id")
            if audit_logger and query_id:
                audit_logger.log_step(
                    query_id=query_id,
                    step_name="RouterAgent",
                    status="Success",
                    route=route,
                    method="llm",
                    classification=classification
                )
            
            return {
                "route": route,
                "audit_log": [{
                    "step": "RouterAgent",
                    "status": "Success",
                    "route": route,
                    "method": "llm",
                    "classification": classification
                }]
            }
            
        except Exception as e:
            logger.error(f"RouterAgent LLM Error: {e}") 
            route = self._fallback_classification(query)
            
            logger.info(f"RouterAgent: Fallback classified as '{route}'")
            
            # Progressive logging
            audit_logger = state.get("audit_logger")
            query_id = state.get("query_id")
            if audit_logger and query_id:
                audit_logger.log_step(
                    query_id=query_id,
                    step_name="RouterAgent",
                    status="Success",
                    route=route,
                    method="fallback",
                    error=str(e)
                )
            
            return {
                "route": route,
                "audit_log": [{
                    "step": "RouterAgent",
                    "status": "Success",
                    "route": route,
                    "method": "fallback",
                    "error": str(e)
                }]
            }
    
    def _fallback_classification(self, query: str) -> str:
        query_lower = query.lower()
        
        greetings = ['hello', 'hi', 'hey', 'thanks', 'thank', 'bye']
        for greeting in greetings:
            if greeting in query_lower:
                return "conversational"
        
        if len(query.split()) < 3 and '?' in query:
            return "conversational"
        
        return "document"

def reject_query(state):
    response = (
        "I'm sorry, but I can only answer questions about the documents you've uploaded. "
        "Your query appears to be conversational or out of scope.\n\n"
        "Please ask me questions about your documents, such as:\n"
        "- What is the revenue mentioned in the report?\n"
        "- Summarize the key findings\n"
        "- Extract data from tables"
    )
    
    audit_logger = state.get("audit_logger")
    query_id = state.get("query_id")
    if audit_logger and query_id:
        audit_logger.log_step(query_id, "RejectionHandler", "Out-of-Scope", query=state.get("query"))
    
    return {
        "final_response": response,
        "audit_log": [{"step": "RejectionHandler", "status": "Out-of-Scope"}]
    }
