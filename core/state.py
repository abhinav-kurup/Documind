from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
import operator
from typing import Annotated

def merge_lists(a: list, b: list) -> list:
    return a + b

class AgentState(TypedDict):
    query: str
    messages: Annotated[List[BaseMessage], operator.add]
    route: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    extracted_data: Dict[str, Any]
    final_response: Optional[str]
    audit_log: Annotated[List[Dict[str, Any]], merge_lists]
