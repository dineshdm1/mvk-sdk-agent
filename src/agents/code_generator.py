"""Code Generator Agent - Generates working code examples."""

from typing import Dict, Optional
from langchain_openai import ChatOpenAI
import mvk_sdk as mvk

from ..utils.config import config
from ..utils.mvk_tracker import tracker
from ..prompts import CODE_GENERATOR_PROMPT


class CodeGenerator:
    """Agent for generating code examples."""

    def __init__(self):
        """Initialize Code Generator."""
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE_CODE,
            openai_api_key=config.OPENAI_API_KEY
        )

    @mvk.signal(step_type="AGENT", operation="code_generation")
    def generate(
        self,
        user_query: str,
        sdk_context: Optional[str] = None,
        framework_context: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate code example.

        Args:
            user_query: User's requirements
            sdk_context: Context from SDK Agent
            framework_context: Context from Framework Specialist

        Returns:
            Dictionary with code, explanation, cost estimate, and gotchas
        """
        try:
            # Build comprehensive context
            sdk_ctx = sdk_context or "No specific SDK context provided."
            framework_ctx = framework_context or "No specific framework context provided."

            # Stage 1: Code generation
            with mvk.context(name="stage.generation"):
                with mvk.create_signal(
                    name="tool.llm_code_gen",
                    step_type="TOOL",
                    operation="generate"
                ):
                    prompt = CODE_GENERATOR_PROMPT.format(
                        user_query=user_query,
                        sdk_context=sdk_ctx,
                        framework_context=framework_ctx
                    )

                    # LLM call is auto-tracked by MVK SDK
                    response = self.llm.invoke([
                        {"role": "system", "content": "You are an expert code generator for MVK SDK integration."},
                        {"role": "user", "content": prompt}
                    ])

                    raw_response = response.content

            # Stage 2: Response parsing
            with mvk.context(name="stage.parsing"):
                parsed = self._parse_response(raw_response)

            # Track metrics
            tracker.track_metric("code_generator.generations", 1, "generation")

            return {
                **parsed,
                "success": True
            }

        except Exception as e:
            print(f"❌ Code Generator error: {e}")
            tracker.track_metric("code_generator.errors", 1, "error")

            return {
                "code": f"# Error generating code: {str(e)}",
                "explanation": "",
                "cost_estimate": "",
                "gotchas": "",
                "success": False
            }

    def _parse_response(self, response: str) -> Dict[str, str]:
        """
        Parse LLM response to extract code, explanation, cost, and gotchas.

        Args:
            response: Raw LLM response

        Returns:
            Dictionary with parsed components
        """
        code = self._extract_code_block(response)
        explanation = self._extract_section(response, "Explanation")
        cost_estimate = self._extract_section(response, "Estimated Cost")
        gotchas = self._extract_section(response, "Gotchas")

        return {
            "code": code,
            "explanation": explanation,
            "cost_estimate": cost_estimate,
            "gotchas": gotchas
        }

    def _extract_code_block(self, text: str) -> str:
        """Extract code from markdown code block."""
        # Find ```python or ``` code blocks
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                return code
        elif "```" in text:
            parts = text.split("```")
            if len(parts) > 2:
                code = parts[1].strip()
                # Remove language identifier if present
                lines = code.split("\n")
                if lines and lines[0].strip().lower() in ["python", "py"]:
                    code = "\n".join(lines[1:])
                return code

        return text.strip()

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from the response."""
        # Try to find section with **SectionName:** or **SectionName**:
        markers = [
            f"**{section_name}:**",
            f"**{section_name}**:",
            f"{section_name}:",
        ]

        for marker in markers:
            if marker in text:
                parts = text.split(marker)
                if len(parts) > 1:
                    # Get text until next section or end
                    section_text = parts[1]

                    # Stop at next markdown header or section
                    for stop_marker in ["**Explanation", "**Estimated Cost", "**Gotchas", "**", "\n\n##"]:
                        if stop_marker in section_text and stop_marker != marker:
                            section_text = section_text.split(stop_marker)[0]
                            break

                    return section_text.strip()

        return ""

    def get_stats(self) -> Dict[str, any]:
        """Get Code Generator statistics."""
        return {
            "model": config.LLM_MODEL,
            "temperature": config.LLM_TEMPERATURE_CODE
        }


# Export singleton instance
code_generator = CodeGenerator()
