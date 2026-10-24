import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GoToSleep(Tool):
    """Put Reachy to sleep and keep listening until asked to wake."""

    name = "go_to_sleep"
    description = (
        "Put Reachy to sleep until the user clearly asks him to wake up. "
        "He stays in this app and keeps listening, but he does not chat or move. "
        "Use only for an explicit request to sleep or end the conversation. "
        "Do not use for idle turns, sleepy emotions, silence, or ambiguous requests."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Put Reachy to sleep without stopping the conversation."""
        if deps.go_to_sleep is None:
            return {"error": "go_to_sleep is unavailable in this runtime"}

        logger.info("Tool call: go_to_sleep")
        try:
            return await asyncio.to_thread(deps.go_to_sleep)
        except Exception as e:
            logger.error("go_to_sleep failed: %s", e)
            return {"error": f"go_to_sleep failed: {type(e).__name__}: {e}"}
