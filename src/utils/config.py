"""Configuration management for Mavvrik SDK Assistant."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Tavily Configuration
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # MVK SDK Configuration
    MVK_API_KEY: str = os.getenv("MVK_API_KEY", "")
    MVK_AGENT_ID: str = os.getenv("MVK_AGENT_ID", "mavvrik-sdk-assistant")
    MVK_TENANT_ID: str = os.getenv("MVK_TENANT_ID", "mavvrik-internal")

    # ChromaDB Configuration
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "mvk_sdk_docs")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma/data")

    # Document Processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5

    # Chainlit Configuration
    CHAINLIT_PORT: int = int(os.getenv("CHAINLIT_PORT", "8000"))

    # Authentication
    AUTH_PASSWORD: str = os.getenv("AUTH_PASSWORD", "mavvrik@123")

    # PDF Documentation Path
    PDF_PATH: str = "./docs/mvk_sdk_documentation.pdf"

    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = 20

    # Performance
    LLM_TEMPERATURE_INTENT: float = 0.0
    LLM_TEMPERATURE_SDK: float = 0.1
    LLM_TEMPERATURE_FRAMEWORK: float = 0.2
    LLM_TEMPERATURE_CODE: float = 0.3

    # MVK SDK Tool Pricing (for custom cost tracking)
    TOOL_PRICES = {
        # Vector database operations
        "chromadb_search": 0.0001,          # Per search operation
        "chromadb_indexing": 0.0005,        # Per document indexed

        # Web search operations
        "tavily_search": 0.001,             # Per search request

        # PDF processing operations
        "pdf_ingestion": 0.002,             # Per page loaded
        "pdf_parsing": 0.001,               # Per document parsed
    }

    @classmethod
    def validate(cls) -> list[str]:
        """Validate required configuration and return list of errors."""
        errors = []

        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required. Get one at https://platform.openai.com/api-keys")

        if not cls.TAVILY_API_KEY:
            errors.append("TAVILY_API_KEY is required. Get one at https://app.tavily.com")

        if not cls.MVK_API_KEY:
            errors.append("MVK_API_KEY is required for tracking")

        if not os.path.exists(cls.PDF_PATH):
            errors.append(f"PDF documentation not found at {cls.PDF_PATH}. Please add mvk_sdk_documentation.pdf")

        return errors

    @classmethod
    def is_valid(cls) -> bool:
        """Check if all required configuration is present."""
        return len(cls.validate()) == 0

    @classmethod
    def get_error_message(cls) -> str:
        """Get formatted error message for missing configuration."""
        errors = cls.validate()
        if not errors:
            return ""

        message = "❌ Configuration errors:\n\n"
        for error in errors:
            message += f"  • {error}\n"

        message += "\nPlease check your .env file and ensure all required values are set."
        return message


# Export singleton instance
config = Config()
