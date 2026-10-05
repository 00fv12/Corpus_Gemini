from pypdf import PdfReader
from typing import List
from corpus_ai.ingestion.models import Document

def load_pdf(file_path: str) -> List[Document]:
    reader = PdfReader(file_path)
    documents = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            doc = Document(
                id=f"pdf:{file_path}:page-{i+1}",
                source_type="pdf",
                source_name=file_path,
                text=text.strip(),
                metadata={"page_number": i + 1}
            )
            documents.append(doc)
            
    return documents