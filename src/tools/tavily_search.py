"""Tavily web search integration."""

from typing import List, Dict, Optional
from tavily import TavilyClient
import mvk_sdk as mvk
from mvk_sdk import Metric

from ..utils.config import config


class TavilySearch:
    """Web search using Tavily API."""

    def __init__(self):
        """Initialize Tavily client."""
        self.client = TavilyClient(api_key=config.TAVILY_API_KEY)

    def search(
        self,
        query: str,
        max_results: int = 3,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Perform web search.

        Args:
            query: Search query
            max_results: Maximum number of results
            search_depth: Search depth ("basic" or "advanced")
            include_domains: List of domains to include

        Returns:
            List of search results with title, url, content
        """
        with mvk.create_signal(
            name="tool.tavily_search",
            step_type="TOOL",
            operation="web_search"
        ):
            try:
                # Build search parameters
                search_params = {
                    "query": query,
                    "max_results": max_results,
                    "search_depth": search_depth,
                }

                if include_domains:
                    search_params["include_domains"] = include_domains

                # Perform search
                response = self.client.search(**search_params)

                # Extract results
                results = []
                for item in response.get("results", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", ""),
                        "score": item.get("score", 0.0)
                    })

                # Track Tavily search cost
                mvk.add_metered_usage(
                    Metric(
                        name="tavily.search",
                        value=1,
                        unit="search",
                        estimated_cost=config.TOOL_PRICES["tavily_search"],
                        metadata={
                            "max_results": max_results,
                            "search_depth": search_depth,
                            "results_returned": len(results),
                            "domains": include_domains if include_domains else "all"
                        }
                    )
                )

                return results

            except Exception as e:
                print(f"❌ Tavily search error: {e}")
                return []

    def search_framework(
        self,
        framework_name: str,
        query: str,
        max_results: int = 3
    ) -> List[Dict[str, str]]:
        """
        Search for framework-specific information.

        Args:
            framework_name: Framework name (langchain, llamaindex, etc.)
            query: Search query
            max_results: Maximum results

        Returns:
            List of search results
        """
        # Construct framework-specific query
        full_query = f"{framework_name} {query}"

        # Prefer official docs and GitHub
        domains = self._get_framework_domains(framework_name)

        return self.search(
            query=full_query,
            max_results=max_results,
            search_depth="advanced",
            include_domains=domains if domains else None
        )

    def _get_framework_domains(self, framework_name: str) -> Optional[List[str]]:
        """Get preferred domains for each framework."""
        framework_domains = {
            "langchain": ["python.langchain.com", "github.com/langchain-ai"],
            "llamaindex": ["docs.llamaindex.ai", "github.com/run-llama"],
            "crewai": ["docs.crewai.com", "github.com/joaomdmoura/crewai"],
            "autogen": ["microsoft.github.io/autogen", "github.com/microsoft/autogen"],
            "haystack": ["docs.haystack.deepset.ai", "github.com/deepset-ai/haystack"],
        }

        return framework_domains.get(framework_name.lower())

    def format_results(self, results: List[Dict[str, str]]) -> str:
        """
        Format search results as text.

        Args:
            results: List of search results

        Returns:
            Formatted text
        """
        if not results:
            return "No results found."

        formatted = "Search Results:\n\n"

        for i, result in enumerate(results, 1):
            formatted += f"{i}. **{result['title']}**\n"
            formatted += f"   URL: {result['url']}\n"
            formatted += f"   {result['content'][:300]}...\n\n"

        return formatted

    def get_combined_context(self, results: List[Dict[str, str]]) -> str:
        """
        Combine search results into single context string.

        Args:
            results: List of search results

        Returns:
            Combined context
        """
        if not results:
            return ""

        context = ""
        for result in results:
            context += f"Source: {result['title']} ({result['url']})\n"
            context += f"{result['content']}\n\n"

        return context


# Export singleton instance
tavily_search = TavilySearch()
