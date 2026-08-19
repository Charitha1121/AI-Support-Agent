import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    filename: str
    title: str
    section: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)


def get_default_knowledge_base_dir() -> Path:
    """Resolve the absolute path to the data/knowledge_base directory."""
    backend_dir = Path(__file__).resolve().parent.parent.parent
    kb_dir = backend_dir.parent / "data" / "knowledge_base"
    if not kb_dir.exists():
        # Fallback to local data/knowledge_base if running from workspace root
        kb_dir = Path("data/knowledge_base").resolve()
    return kb_dir


def load_raw_documents(kb_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    """Load all markdown documents from the knowledge base directory."""
    if kb_dir is None:
        kb_dir = get_default_knowledge_base_dir()

    documents = []
    if not kb_dir.exists():
        return documents

    for file_path in sorted(kb_dir.glob("*.md")):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        documents.append({
            "filename": file_path.name,
            "filepath": str(file_path),
            "content": content
        })

    return documents


def extract_title_and_sections(markdown_text: str) -> tuple[str, List[tuple[str, str]]]:
    """
    Extract the main title (# H1) and sections (## H2 + section content) from markdown text.
    """
    lines = markdown_text.strip().split("\n")
    title = "NovaTech Policy"
    
    # Extract main title
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    # Split into sections based on ## headers
    sections: List[tuple[str, str]] = []
    current_section_title = "Overview"
    current_section_lines: List[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_section_lines:
                section_text = "\n".join(current_section_lines).strip()
                if section_text:
                    sections.append((current_section_title, section_text))
            current_section_title = line[3:].strip()
            current_section_lines = []
        else:
            # Skip the H1 header line from body text to avoid duplication
            if not line.startswith("# "):
                current_section_lines.append(line)

    if current_section_lines:
        section_text = "\n".join(current_section_lines).strip()
        if section_text:
            sections.append((current_section_title, section_text))

    if not sections and markdown_text.strip():
        sections.append(("Overview", markdown_text.strip()))

    return title, sections


def chunk_documents(kb_dir: Optional[Path] = None, max_chunk_chars: int = 800) -> List[DocumentChunk]:
    """
    Load all knowledge base documents and produce structured, semantically chunked DocumentChunks.
    """
    raw_docs = load_raw_documents(kb_dir)
    chunks: List[DocumentChunk] = []

    for doc_idx, doc in enumerate(raw_docs):
        filename = doc["filename"]
        doc_id = filename.replace(".md", "")
        title, sections = extract_title_and_sections(doc["content"])

        for sec_idx, (section_title, section_content) in enumerate(sections):
            # If section content is within max size, keep it as single chunk
            if len(section_content) <= max_chunk_chars:
                chunk_id = f"{doc_id}_s{sec_idx}_c0"
                formatted_content = f"[{title} - {section_title}]\n{section_content}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        filename=filename,
                        title=title,
                        section=section_title,
                        content=formatted_content,
                        metadata={
                            "doc_id": doc_id,
                            "filename": filename,
                            "title": title,
                            "section": section_title,
                        }
                    )
                )
            else:
                # Split large sections by paragraphs
                paragraphs = section_content.split("\n\n")
                sub_chunk_idx = 0
                buffer = ""

                for para in paragraphs:
                    if len(buffer) + len(para) + 2 <= max_chunk_chars:
                        buffer = f"{buffer}\n\n{para}".strip() if buffer else para
                    else:
                        if buffer:
                            chunk_id = f"{doc_id}_s{sec_idx}_c{sub_chunk_idx}"
                            formatted_content = f"[{title} - {section_title}]\n{buffer}"
                            chunks.append(
                                DocumentChunk(
                                    chunk_id=chunk_id,
                                    doc_id=doc_id,
                                    filename=filename,
                                    title=title,
                                    section=section_title,
                                    content=formatted_content,
                                    metadata={
                                        "doc_id": doc_id,
                                        "filename": filename,
                                        "title": title,
                                        "section": section_title,
                                    }
                                )
                            )
                            sub_chunk_idx += 1
                        buffer = para

                if buffer:
                    chunk_id = f"{doc_id}_s{sec_idx}_c{sub_chunk_idx}"
                    formatted_content = f"[{title} - {section_title}]\n{buffer}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            filename=filename,
                            title=title,
                            section=section_title,
                            content=formatted_content,
                            metadata={
                                "doc_id": doc_id,
                                "filename": filename,
                                "title": title,
                                "section": section_title,
                            }
                        )
                    )

    return chunks
