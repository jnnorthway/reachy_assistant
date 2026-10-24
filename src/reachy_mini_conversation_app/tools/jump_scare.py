import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.sleep_mode import sleep_mode
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class JumpScare(Tool):
    """Enter jump scare mode and stay asleep until a wake phrase."""

    name = "jump_scare"
    description = (
        "Enter jump scare mode when the user asks for a jump scare or scare mode. "
        "Kevin goes to sleep, stays still, and keeps listening. Do not speak before or after calling this."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Arm jump scare and move to the sleep pose."""
        logger.info("Tool call: jump_scare")
        if deps.go_to_sleep is None:
            return {"error": "jump_scare is unavailable in this runtime"}
        sleep_mode.enter_jump_scare()
        try:
            result = await asyncio.to_thread(deps.go_to_sleep)
        except Exception as exc:
            logger.error("jump_scare failed: %s", exc)
            return {"error": f"jump_scare failed: {type(exc).__name__}: {exc}"}
        if isinstance(result, dict) and result.get("error"):
            logger.error("jump_scare sleep pose failed: %s", result.get("error"))
            return {"status": "jump_scare", "listening": True, "error": result["error"]}
        return {"status": "jump_scare", "listening": True}
