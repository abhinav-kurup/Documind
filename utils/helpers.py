from core.state import AgentState

def log_agent_step(state: AgentState, step_name: str, status: str, **kwargs):
    """Safely logs an agent step if the logger exists in state."""
    audit_logger = state.get("audit_logger")
    query_id = state.get("query_id")
    if audit_logger and query_id:
        audit_logger.log_step(query_id, step_name, status, **kwargs)
