import pytesseract
from PIL import Image
import io
import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

# Ensure pytesseract can find the binary if not in PATH (though Dockerfile installs it)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # Typical linux path

class OCRProcessor:
    def __init__(self, lang: str = 'eng'):
        self.lang = lang

    def process_page(self, page: fitz.Page) -> str:
        """
        Renders the PDF page as an image and performs OCR.
        
        Args:
            page: fitz.Page object
            
        Returns:
            Extracted text string
        """
        try:
            # Zoom to improve OCR quality (2.0 = 200% DPI approx)
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            text = pytesseract.image_to_string(image, lang=self.lang)
            return text
        except Exception as e:
            logger.error(f"OCR failed for page {page.number}: {e}")
            return ""

    def process_image(self, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=self.lang)
            return text
        except Exception as e:
            logger.error(f"OCR failed for image {image_path}: {e}")
            return ""
