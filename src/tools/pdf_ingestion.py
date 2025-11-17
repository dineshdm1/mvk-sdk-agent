"""PDF ingestion and processing."""

import os
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import mvk_sdk as mvk
from mvk_sdk import Metric

from ..utils.config import config


class PDFIngestor:
    """Handle PDF document ingestion and chunking."""

    def __init__(self, pdf_path: str = None, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize PDF ingestor.

        Args:
            pdf_path: Path to PDF file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.pdf_path = pdf_path or config.PDF_PATH
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def ingest(self) -> List[Document]:
        """
        Load and process PDF into chunks.

        Returns:
            List of Document chunks

        Raises:
            FileNotFoundError: If PDF file doesn't exist
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(
                f"PDF file not found at {self.pdf_path}. "
                f"Please place 'mvk_sdk_documentation.pdf' in the docs/ directory."
            )

        # Stage 1: Load PDF
        with mvk.create_signal(
            name="tool.pdf_loader",
            step_type="TOOL",
            operation="load"
        ):
            print(f"📄 Loading PDF from {self.pdf_path}...")
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            print(f"✅ Loaded {len(documents)} pages")

            # Track PDF loading cost
            mvk.add_metered_usage(
                Metric(
                    name="pdf.pages_loaded",
                    value=len(documents),
                    unit="page",
                    estimated_cost=config.TOOL_PRICES["pdf_ingestion"] * len(documents),
                    metadata={
                        "file_path": self.pdf_path,
                        "pages_loaded": len(documents)
                    }
                )
            )

        # Stage 2: Split into chunks
        with mvk.create_signal(
            name="tool.pdf_splitter",
            step_type="TOOL",
            operation="split"
        ):
            print(f"✂️  Splitting into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})...")
            chunks = self.text_splitter.split_documents(documents)
            print(f"✅ Created {len(chunks)} chunks")

            # Track PDF parsing cost
            mvk.add_metered_usage(
                Metric(
                    name="pdf.chunks_created",
                    value=len(chunks),
                    unit="chunk",
                    estimated_cost=config.TOOL_PRICES["pdf_parsing"],
                    metadata={
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "total_chunks": len(chunks)
                    }
                )
            )

        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["source"] = "mvk_sdk_documentation.pdf"

        return chunks

    def get_stats(self) -> dict:
        """Get PDF statistics."""
        if not os.path.exists(self.pdf_path):
            return {"exists": False}

        file_size = os.path.getsize(self.pdf_path)

        return {
            "exists": True,
            "path": self.pdf_path,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
        }


# Export singleton instance
pdf_ingestor = PDFIngestor()
