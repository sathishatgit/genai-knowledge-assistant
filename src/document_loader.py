"""Document loader module for GenAI Knowledge Assistant.

Loads plain text documents from the specified directory, capturing file content
and source metadata in a beginner-friendly structure.
"""

from pathlib import Path
from typing import List, Dict, Any
from src.config import DOCUMENTS_DIR


def load_documents(directory_path: Path | str = DOCUMENTS_DIR) -> List[Dict[str, Any]]:
    """Load all .txt files from the target directory.
    
    Args:
        directory_path: Path to the folder containing document files.
        
    Returns:
        List of dictionaries with keys:
            - id: Unique document ID (e.g., 'doc_rag.txt')
            - text: Text content of the file
            - metadata: Dictionary containing source filename and other info
    """
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print(f"Warning: Document directory '{directory}' does not exist.")
        return []

    documents: List[Dict[str, Any]] = []
    
    # Sort files for deterministic loading order
    txt_files = sorted(directory.glob("*.txt"))
    if not txt_files:
        print(f"No .txt documents found in '{directory}'.")
        return []

    for file_path in txt_files:
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            # Ignore empty documents
            if not content:
                continue

            doc_id = f"doc_{file_path.name}"
            documents.append({
                "id": doc_id,
                "text": content,
                "metadata": {
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "char_count": len(content),
                }
            })
        except Exception as e:
            print(f"Error reading file {file_path.name}: {e}")

    return documents
