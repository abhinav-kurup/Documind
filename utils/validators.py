from typing import Any

def validate_file_type(filename: str, allowed_extensions: set = {'pdf'}) -> bool:
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)
