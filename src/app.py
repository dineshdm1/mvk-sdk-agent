"""Mavvrik SDK Assistant - Main Chainlit Application."""

import chainlit as cl
from chainlit.input_widget import TextInput
import time
import mvk_sdk as mvk

from utils.config import config
from utils.session_manager import session_manager
from agents.orchestrator import chat_orchestrator
from prompts import (
    WELCOME_MESSAGE,
    AUTH_USERNAME_PROMPT,
    AUTH_PASSWORD_PROMPT,
    AUTH_SUCCESS_MESSAGE,
    AUTH_FAILED_MESSAGE,
    ERROR_GENERAL
)


# Authentication state
AUTH_STATE_USERNAME = "awaiting_username"
AUTH_STATE_PASSWORD = "awaiting_password"
AUTH_STATE_AUTHENTICATED = "authenticated"


@cl.on_chat_start
async def start():
    """Initialize chat session."""
    # Validate configuration
    if not config.is_valid():
        error_msg = config.get_error_message()
        await cl.Message(content=error_msg).send()
        return

    # Ask for username
    await cl.Message(content=AUTH_USERNAME_PROMPT).send()

    # Set initial authentication state
    cl.user_session.set("auth_state", AUTH_STATE_USERNAME)
    cl.user_session.set("authenticated", False)


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages."""
    auth_state = cl.user_session.get("auth_state", AUTH_STATE_USERNAME)
    user_content = message.content.strip()

    # Handle authentication flow
    if auth_state == AUTH_STATE_USERNAME:
        await handle_username_input(user_content)
        return

    elif auth_state == AUTH_STATE_PASSWORD:
        await handle_password_input(user_content)
        return

    elif auth_state == AUTH_STATE_AUTHENTICATED:
        await handle_query(user_content)
        return

    else:
        await cl.Message(content="❌ Invalid session state. Please refresh the page.").send()


async def handle_username_input(username: str):
    """
    Handle username input.

    Args:
        username: Entered username
    """
    if not username or len(username) < 2:
        await cl.Message(content="⚠️ Please enter a valid username (at least 2 characters).").send()
        await cl.Message(content=AUTH_USERNAME_PROMPT).send()
        return

    # Store username
    cl.user_session.set("username", username)

    # Create session
    user_session = session_manager.create_session(user_id=username)
    cl.user_session.set("session_id", user_session.session_id)

    # Ask for password
    await cl.Message(content=AUTH_PASSWORD_PROMPT).send()
    cl.user_session.set("auth_state", AUTH_STATE_PASSWORD)


async def handle_password_input(password: str):
    """
    Handle password input.

    Args:
        password: Entered password
    """
    username = cl.user_session.get("username")
    session_id = cl.user_session.get("session_id")

    # Validate password
    if password == config.AUTH_PASSWORD:
        # Authenticate session
        session_manager.authenticate_session(session_id, password, config.AUTH_PASSWORD)

        # Update state
        cl.user_session.set("authenticated", True)
        cl.user_session.set("auth_state", AUTH_STATE_AUTHENTICATED)

        # Send success message
        success_msg = AUTH_SUCCESS_MESSAGE.format(username=username)
        await cl.Message(content=success_msg).send()

        # Send welcome message
        await cl.Message(content=WELCOME_MESSAGE).send()

    else:
        # Authentication failed
        await cl.Message(content=AUTH_FAILED_MESSAGE).send()
        await cl.Message(content=AUTH_PASSWORD_PROMPT).send()


async def handle_query(query: str):
    """
    Handle user query after authentication.

    Args:
        query: User's question
    """
    # Get session info
    username = cl.user_session.get("username")
    session_id = cl.user_session.get("session_id")

    # Create conversation ID for this query
    conversation_id = f"conv-{int(time.time() * 1000)}"

    # Show loading message
    loading_msg = await cl.Message(content="🔍 Processing your query...").send()

    try:
        # Track query with MVK SDK context
        start_time = time.time()

        # Set top-level business context for entire request
        with mvk.context(
            user_id=username,
            session_id=session_id,
            tenant_id=config.MVK_TENANT_ID
        ):
            with mvk.context(conversation_id=conversation_id):
                # Process query through orchestrator
                result = chat_orchestrator.process_query(query)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Update loading message with result
        await loading_msg.update(content=result["answer"])

        # Store conversation in session
        session_manager.add_conversation(
            session_id=session_id,
            user_message=query,
            assistant_message=result["answer"],
            conversation_id=conversation_id
        )

        # Store conversation_id for feedback
        cl.user_session.set("last_conversation_id", conversation_id)

        # Add feedback actions
        actions = [
            cl.Action(name="feedback_helpful", value="helpful", label="👍 Helpful"),
            cl.Action(name="feedback_not_helpful", value="not_helpful", label="👎 Not Helpful")
        ]

        await cl.Message(
            content="Was this response helpful?",
            actions=actions
        ).send()

    except Exception as e:
        error_msg = ERROR_GENERAL.format(error=str(e))
        await loading_msg.update(content=error_msg)


@cl.action_callback("feedback_helpful")
async def on_feedback_helpful(action: cl.Action):
    """Handle thumbs up feedback."""
    session_id = cl.user_session.get("session_id")
    conversation_id = cl.user_session.get("last_conversation_id")

    if session_id and conversation_id:
        session_manager.add_feedback(session_id, conversation_id, "helpful")

    await cl.Message(content="✅ Thank you for your feedback!").send()

    # Remove the action
    await action.remove()


@cl.action_callback("feedback_not_helpful")
async def on_feedback_not_helpful(action: cl.Action):
    """Handle thumbs down feedback."""
    session_id = cl.user_session.get("session_id")
    conversation_id = cl.user_session.get("last_conversation_id")

    if session_id and conversation_id:
        session_manager.add_feedback(session_id, conversation_id, "not_helpful")

    await cl.Message(content="✅ Thank you for your feedback! We'll work on improving.").send()

    # Remove the action
    await action.remove()


@cl.on_chat_end
async def end():
    """Handle chat session end."""
    session_id = cl.user_session.get("session_id")
    if session_id:
        user_session = session_manager.get_session(session_id)
        if user_session:
            print(f"📊 Session ended: {user_session.to_dict()}")


if __name__ == "__main__":
    # This allows running with: python app.py
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
