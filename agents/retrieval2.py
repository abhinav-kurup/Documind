import logging
import hashlib
from core.config import Config
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from core.state import AgentState
from utils.helpers import log_agent_step, dump_agent_state
from core.llm import get_llm

logger = logging.getLogger(__name__)


class SearchPlan(BaseModel):
    sub_queries: List[str] = Field(
        description="1-3 optimized search queries derived from user query"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters"
    )



class RetrievalAgent:
    def __init__(self, vector_store, model_identifier: str = "gemini/gemini-2.5-flash-lite"):
        self.vector_store = vector_store
        self.llm = get_llm(model_identifier, temperature=0.1)

        try:
            print("Structured")
            self.structured_llm = self.llm.with_structured_output(SearchPlan)
            logger.info(f"Initialized with structured output: {model_identifier}")
        except NotImplementedError:
            logger.warning("Structured output not supported, fallback mode")
            self.structured_llm = None


    def invoke(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "AdvancedRetrievalAgent")

        query = state.get("query", "")
        if not query:
            return {"audit_log": [{"step": "AdvancedRetrievalAgent", "status": "Skipped"}]}

        logger.info(f"AdvancedRetrievalAgent: Query -> '{query}'")

        try:
            search_plan = self._create_search_plan(query)

            docs = self._retrieve_documents(query, search_plan)

            logger.info(f"AdvancedRetrievalAgent: Final docs count {len(docs)}")

            log_agent_step(
                state=state,
                step_name="AdvancedRetrievalAgent",
                status="Success",
                retrieved_count=len(docs),
                query=query
            )

            return {
                "retrieved_docs": docs,
                "audit_log": [{
                    "step": "AdvancedRetrievalAgent",
                    "status": "Success",
                    "retrieved_count": len(docs),
                    "query": query
                }]
            }

        except Exception as e:
            logger.error(f"AdvancedRetrievalAgent Error: {e}")
            return {
                "audit_log": [{
                    "step": "AdvancedRetrievalAgent",
                    "status": "Error",
                    "error": str(e)
                }]
            }

    def _create_search_plan(self, query: str) -> SearchPlan:
        """
        Smart query planning:
        - Use LLM to breakdown complex queries
        """

        if len(query.split()) <= 5:
            return SearchPlan(sub_queries=[query], filters=None)

        if self.structured_llm:
            try:
                prompt = (
                    "Break the user query into 1-3 semantic search queries.\n"
                    "Extract filters if present.\n\n"
                    f"Query: {query}"
                )
                plan = self.structured_llm.invoke(prompt)
                plan.sub_queries = list(set([query] + plan.sub_queries))
                return plan

            except Exception as e:
                logger.warning(f"Search plan failed, fallback: {e}")

        return SearchPlan(sub_queries=[query], filters=None)


    def _retrieve_documents(self, query: str, plan: SearchPlan) -> List[Dict]:
        """
        Retrieval pipeline:
        - multi-query search
        - deduplication
        - ranking
        """

        all_docs = []
        seen_hashes = set()

        for sub_q in plan.sub_queries:
            logger.info(f"Sub-query: {sub_q}")

            results = self.vector_store.similarity_search_with_score(
                sub_q,
                k=4,
                filter=plan.filters
            )

            for doc, score in results:
                content = doc.page_content.strip()

                content_hash = hashlib.md5(content.encode()).hexdigest()
                if content_hash in seen_hashes:
                    continue

                seen_hashes.add(content_hash)

                all_docs.append({
                    "content": content,
                    "metadata": doc.metadata,
                    "score": score
                })

        ranked_docs = sorted(
            all_docs,
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        final_docs = [
            {
                "content": d["content"],
                "metadata": d["metadata"]
            }
            for d in ranked_docs[:5]
        ]

        return final_docs