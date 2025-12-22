import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

class PDFLoader:
    def __init__(self):
        pass

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a PDF file and extracts content from each page.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of dictionaries containing page content and metadata.
            [
                {
                    "page_number": int,
                    "text": str,
                    "is_scanned": bool,
                    "images": List[Any],
                    "metadata": Dict
                },
                ...
            ]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {file_path}: {e}")
            raise e

        processed_pages = []

        for i, page in enumerate(doc):
            text = page.get_text()
            is_scanned_page = self.is_scanned(page, text)
            
            page_data = {
                "page_number": i + 1,
                "text": text,
                "is_scanned": is_scanned_page,
                "images": page.get_images(full=True),
                "metadata": doc.metadata
            }
            processed_pages.append(page_data)

        doc.close()
        return processed_pages

    def is_scanned(self, page: fitz.Page, text: str) -> bool:
        """
        Determines if a page is likely scanned based on text density and presence of images.
        """
        # If very little text but has images covering the page, it's likely scanned.
        text_len = len(text.strip())
        if text_len < 50:
            images = page.get_images()
            if len(images) > 0:
                # Could check image size coverage here for more accuracy
                return True
        return False
