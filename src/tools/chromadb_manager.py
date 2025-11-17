"""ChromaDB vector store management."""

import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import mvk_sdk as mvk

from ..utils.config import config
from ..utils.mvk_tracker import tracker


class ChromaDBManager:
    """Manage ChromaDB vector store operations."""

    def __init__(self):
        """Initialize ChromaDB manager."""
        self.collection_name = config.CHROMA_COLLECTION
        self.persist_directory = config.CHROMA_PERSIST_DIR
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        self._vectorstore: Optional[Chroma] = None

    @property
    def vectorstore(self) -> Chroma:
        """Get or create vectorstore instance."""
        if self._vectorstore is None:
            self._vectorstore = self._load_or_create_vectorstore()
        return self._vectorstore

    def _load_or_create_vectorstore(self) -> Chroma:
        """Load existing vectorstore or create new one."""
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        try:
            # Try to load existing vectorstore
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

            # Check if it has documents
            if vectorstore._collection.count() > 0:
                print(f"✅ Loaded existing ChromaDB collection '{self.collection_name}' with {vectorstore._collection.count()} documents")
                return vectorstore
            else:
                print(f"⚠️  ChromaDB collection '{self.collection_name}' is empty")
                return vectorstore

        except Exception as e:
            print(f"⚠️  Could not load existing vectorstore: {e}")
            print(f"Creating new vectorstore...")

            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents into ChromaDB.

        Args:
            documents: List of Document chunks to index
        """
        if not documents:
            print("⚠️  No documents to index")
            return

        print(f"🔄 Indexing {len(documents)} documents into ChromaDB...")

        with mvk.create_signal(
            name="tool.chromadb_indexing",
            step_type="TOOL",
            operation="index"
        ):
            # Create embeddings and store
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )

            # Persist to disk
            self._vectorstore.persist()

            count = self._vectorstore._collection.count()

            # Track ChromaDB indexing cost
            tracker.track_operation_with_cost(
                metric_name="chromadb.documents_indexed",
                operation_key="chromadb_indexing",
                quantity=len(documents),
                unit="document",
                provider="chromadb",
                additional_metadata={
                    "total_documents": count,
                    "batch_size": len(documents)
                }
            )

        print(f"✅ Indexed {count} documents in ChromaDB")

    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Similarity search in ChromaDB.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant Documents
        """
        k = k or config.TOP_K_RESULTS

        # Perform search (tracking is done by caller in sdk_agent.py)
        results = self.vectorstore.similarity_search(query, k=k)

        return results

    def search_with_score(self, query: str, k: int = None) -> List[tuple[Document, float]]:
        """
        Similarity search with relevance scores.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (Document, score) tuples
        """
        k = k or config.TOP_K_RESULTS

        # Perform search with score (tracking is done by caller if needed)
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return results

    def is_indexed(self) -> bool:
        """Check if documents are indexed."""
        try:
            count = self.vectorstore._collection.count()
            return count > 0
        except:
            return False

    def get_document_count(self) -> int:
        """Get number of indexed documents."""
        try:
            return self.vectorstore._collection.count()
        except:
            return 0

    def reset(self) -> None:
        """Reset (delete) the ChromaDB collection."""
        print(f"🗑️  Resetting ChromaDB collection '{self.collection_name}'...")

        try:
            if self._vectorstore:
                self._vectorstore._client.delete_collection(self.collection_name)
            self._vectorstore = None
            print("✅ Collection reset successfully")
        except Exception as e:
            print(f"⚠️  Error resetting collection: {e}")

    def get_stats(self) -> dict:
        """Get ChromaDB statistics."""
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "document_count": self.get_document_count(),
            "is_indexed": self.is_indexed(),
        }


# Export singleton instance
chromadb_manager = ChromaDBManager()
