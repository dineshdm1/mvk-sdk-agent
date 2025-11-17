"""Code Generator Agent - Generates working code examples."""

from typing import Dict, Optional
from langchain_openai import ChatOpenAI
import mvk_sdk as mvk

from ..utils.config import config
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
        # Identify this agent
        with mvk.context(agent_name="code_generator"):
            try:
                # Build comprehensive context
                sdk_ctx = sdk_context or "No specific SDK context provided."
                framework_ctx = framework_context or "No specific framework context provided."

                # Stage 1: Code generation
                with mvk.context(name="stage.generation"):
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

                return {
                    **parsed,
                    "success": True
                }

            except Exception as e:
                print(f"❌ Code Generator error: {e}")

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
        # Initialize result
        result = {
            "code": "",
            "explanation": "",
            "cost_estimate": "",
            "gotchas": ""
        }

        # Extract code blocks
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                result["code"] = code_part.strip()

        # Extract explanation
        if "**Explanation:**" in response or "Explanation:" in response:
            explanation_marker = "**Explanation:**" if "**Explanation:**" in response else "Explanation:"
            parts = response.split(explanation_marker)
            if len(parts) > 1:
                explanation_text = parts[1].split("**")[0] if "**" in parts[1] else parts[1]
                result["explanation"] = explanation_text.strip()

        # Extract cost estimate
        if "**Estimated Cost:**" in response or "Estimated Cost:" in response or "**Cost Estimate:**" in response:
            cost_marker = "**Estimated Cost:**" if "**Estimated Cost:**" in response else \
                         "**Cost Estimate:**" if "**Cost Estimate:**" in response else "Estimated Cost:"
            parts = response.split(cost_marker)
            if len(parts) > 1:
                cost_text = parts[1].split("**")[0] if "**" in parts[1] else parts[1]
                result["cost_estimate"] = cost_text.strip()

        # Extract gotchas
        if "**Gotchas:**" in response or "Gotchas:" in response:
            gotchas_marker = "**Gotchas:**" if "**Gotchas:**" in response else "Gotchas:"
            parts = response.split(gotchas_marker)
            if len(parts) > 1:
                gotchas_text = parts[1].split("**")[0] if "**" in parts[1] else parts[1]
                result["gotchas"] = gotchas_text.strip()

        # If code wasn't extracted, use the whole response
        if not result["code"] and "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                result["code"] = parts[1].strip()

        return result

    def get_stats(self) -> Dict[str, any]:
        """Get Code Generator statistics."""
        return {
            "model": config.LLM_MODEL,
            "temperature": config.LLM_TEMPERATURE_CODE
        }


# Export singleton instance
code_generator = CodeGenerator()
