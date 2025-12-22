import cv2
import numpy as np
import fitz
from typing import List, Dict, Any
import io
from PIL import Image

class TableExtractor:
    def __init__(self):
        pass

    def extract_tables(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """
        Detects and extracts tables from a page using OpenCV for structure detection.
        Returns a list of dictionaries representing tables (bbox, data, etc.)
        """
        pix = page.get_pixmap()
        img = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Combine to find table grid
        mask = detect_horizontal + detect_vertical
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tables = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter small arbitrary contours
            if w > 50 and h > 50: 
                # This could be a table
                # Extract text from this region using PyMuPDF rect
                # Note: Coordinates need scaling if pixmap was scaled. Here default is 1.0 (72 DPI) usually?
                # PyMuPDF default pixmap is size of page.
                
                rect = fitz.Rect(x, y, x+w, y+h)
                # Ensure rect is within page bounds
                
                table_text = page.get_text("text", clip=rect)
                tables.append({
                    "bbox": [x, y, w, h],
                    "content": table_text,
                    "confidence": 0.8 # Placeholder
                })
        
        return tables
