"""Chat Orchestrator - Main agent that routes to specialists."""

import json
from typing import Dict, Optional
from langchain_openai import ChatOpenAI
import mvk_sdk as mvk

from ..utils.config import config
from ..prompts import INTENT_CLASSIFICATION_PROMPT, RESPONSE_SYNTHESIS_PROMPT
from .sdk_agent import sdk_agent
from .framework_router import framework_router
from .code_generator import code_generator


class ChatOrchestrator:
    """Main orchestrator that routes queries to specialist agents."""

    def __init__(self):
        """Initialize Chat Orchestrator and MVK SDK instrumentation."""
        # Initialize MVK SDK once for entire application
        mvk.instrument(
            agent_id=config.MVK_AGENT_ID,
            api_key=config.MVK_API_KEY,
            enable_batching=True,
            batch_size=10,
            flush_interval_seconds=5
        )

        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE_INTENT,
            openai_api_key=config.OPENAI_API_KEY
        )

    @mvk.signal(step_type="AGENT", operation="orchestrate")
    def process_query(self, query: str, conversation_history: Optional[str] = None) -> Dict[str, any]:
        """
        Process user query by routing to appropriate agents.

        Args:
            query: User's question
            conversation_history: Optional conversation context

        Returns:
            Dictionary with answer and metadata
        """
        # Identify this agent
        with mvk.context(agent_name="orchestrator"):
            try:
                # Stage 1: Intent classification
                with mvk.context(name="stage.intent_classification"):
                    intent = self._classify_intent(query)

                # Stage 2: Agent routing
                with mvk.context(name="stage.agent_routing"):
                    agent_responses = self._route_to_agents(query, intent)

                # Stage 3: Response synthesis
                with mvk.context(name="stage.response_synthesis"):
                    final_response = self._synthesize_response(query, agent_responses, intent)

                return {
                    "answer": final_response,
                    "intent": intent,
                    "agent_responses": agent_responses,
                    "success": True
                }

            except Exception as e:
                print(f"❌ Orchestrator error: {e}")

                return {
                    "answer": f"❌ An error occurred: {str(e)}\n\nPlease try rephrasing your question.",
                    "intent": {},
                    "agent_responses": {},
                    "success": False
                }

    def _classify_intent(self, query: str) -> Dict[str, any]:
        """
        Classify user intent.

        Args:
            query: User's question

        Returns:
            Intent classification dictionary
        """
        try:
            prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)

            # LLM call is auto-tracked by MVK SDK
            response = self.llm.invoke([
                {"role": "system", "content": "You are an intent classification expert."},
                {"role": "user", "content": prompt}
            ])

            # Parse JSON response
            intent_json = response.content.strip()

            # Remove markdown code blocks if present
            if "```json" in intent_json:
                intent_json = intent_json.split("```json")[1].split("```")[0].strip()
            elif "```" in intent_json:
                intent_json = intent_json.split("```")[1].split("```")[0].strip()

            intent = json.loads(intent_json)

            return intent

        except Exception as e:
            print(f"⚠️  Intent classification error: {e}, defaulting to SDK query")
            # Default to SDK query if classification fails
            return {
                "needs_sdk": True,
                "needs_framework": False,
                "needs_code": False,
                "framework_name": None
            }

    def _route_to_agents(self, query: str, intent: Dict[str, any]) -> Dict[str, any]:
        """
        Route query to appropriate specialist agents based on intent.

        Args:
            query: User's question
            intent: Intent classification

        Returns:
            Dictionary of agent responses
        """
        responses = {}

        # Query SDK Agent if needed
        if intent.get("needs_sdk", False):
            # Agent is already instrumented with @mvk.signal()
            sdk_response = sdk_agent.query(query)
            responses["sdk"] = sdk_response

        # Query Framework Specialist if needed
        if intent.get("needs_framework", False):
            framework_name = intent.get("framework_name")
            # Agent is already instrumented with @mvk.signal()
            framework_response = framework_router.query(query, framework_name)
            responses["framework"] = framework_response

        # Query Code Generator if needed
        if intent.get("needs_code", False):
            # Prepare context from previous agents
            sdk_context = responses.get("sdk", {}).get("answer", "")
            framework_context = responses.get("framework", {}).get("answer", "")

            # Agent is already instrumented with @mvk.signal()
            code_response = code_generator.generate(
                user_query=query,
                sdk_context=sdk_context if sdk_context else None,
                framework_context=framework_context if framework_context else None
            )
            responses["code"] = code_response

        return responses

    def _synthesize_response(self, query: str, agent_responses: Dict[str, any], intent: Dict[str, any]) -> str:
        """
        Synthesize final response from multiple agent outputs.

        Args:
            query: User's question
            agent_responses: Dictionary of agent responses
            intent: Intent classification

        Returns:
            Synthesized response string
        """
        # If only one agent responded, return its answer directly
        if len(agent_responses) == 1:
            agent_name = list(agent_responses.keys())[0]
            response = agent_responses[agent_name]

            if agent_name == "code":
                return self._format_code_response(response)
            else:
                return response.get("answer", "")

        # If multiple agents, synthesize
        if len(agent_responses) > 1:
            # Build agent responses text
            responses_text = ""

            if "sdk" in agent_responses:
                responses_text += f"**SDK Agent Response:**\n{agent_responses['sdk'].get('answer', '')}\n\n"

            if "framework" in agent_responses:
                framework_name = intent.get("framework_name", "framework")
                responses_text += f"**{framework_name.capitalize()} Specialist Response:**\n{agent_responses['framework'].get('answer', '')}\n\n"

            if "code" in agent_responses:
                responses_text += f"**Code Generator Response:**\n{self._format_code_response(agent_responses['code'])}\n\n"

            # Use LLM to synthesize (optional - can be disabled for cost saving)
            # For now, just concatenate intelligently
            final_response = "🔄 **Multi-Agent Response:**\n\n"
            final_response += responses_text

            # Add sources if available
            final_response += self._add_sources(agent_responses)

            return final_response

        # Fallback
        return "I couldn't process your query. Please try rephrasing."

    def _format_code_response(self, code_response: Dict[str, any]) -> str:
        """Format code generator response."""
        formatted = ""

        if code_response.get("code"):
            formatted += "```python\n"
            formatted += code_response["code"]
            formatted += "\n```\n\n"

        if code_response.get("explanation"):
            formatted += f"**Explanation:**\n{code_response['explanation']}\n\n"

        if code_response.get("cost_estimate"):
            formatted += f"**Estimated Cost:**\n{code_response['cost_estimate']}\n\n"

        if code_response.get("gotchas"):
            formatted += f"**Gotchas:**\n{code_response['gotchas']}\n\n"

        return formatted

    def _add_sources(self, agent_responses: Dict[str, any]) -> str:
        """Add source citations to response."""
        sources_text = ""

        # SDK sources
        if "sdk" in agent_responses and agent_responses["sdk"].get("sources"):
            sources_text += "**SDK Documentation Sources:**\n"
            for i, source in enumerate(agent_responses["sdk"]["sources"][:3], 1):
                sources_text += f"{i}. Page {source['page']}\n"
            sources_text += "\n"

        # Framework sources
        if "framework" in agent_responses and agent_responses["framework"].get("sources"):
            sources_text += "**Framework Sources:**\n"
            for i, source in enumerate(agent_responses["framework"]["sources"][:3], 1):
                sources_text += f"{i}. [{source['title']}]({source['url']})\n"
            sources_text += "\n"

        return sources_text


# Export singleton instance
chat_orchestrator = ChatOrchestrator()
