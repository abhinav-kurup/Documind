from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import uuid

class DocumentChunker:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Approx characters per chunk (aiming for ~500 tokens)
            chunk_overlap: Approx characters overlap
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_documents(self, pages: List[Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
        """
        Splits pages into chunks with metadata.
        """
        chunks = []
        chunk_counter = 0

        for page in pages:
            text = page.get("text", "")
            if not text:
                continue

            page_chunks = self.splitter.split_text(text)
            
            for chunk_text in page_chunks:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "page_number": page.get("page_number"),
                    "chunk_index": chunk_counter,
                    "text": chunk_text,
                    "metadata": {
                        "source": "pdf", # Could add filename if passed
                        **page.get("metadata", {})
                    }
                })
                chunk_counter += 1
                
        return chunks
