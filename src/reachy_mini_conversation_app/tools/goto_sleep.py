import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GotoSleep(Tool):
    """Put Reachy Mini into sleep posture and disable motors."""

    name = "goto_sleep"
    description = "Put Reachy Mini to sleep posture and relax motors."
    parameters_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional short reason for putting Reachy to sleep.",
            }
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Call the Reachy SDK sleep helper."""
        reason = kwargs.get("reason", "unspecified")
        logger.info("Tool call: goto_sleep reason=%s", reason)

        try:
            movement_manager = deps.movement_manager
            movement_manager.clear_move_queue()

            deps.reachy_mini.goto_sleep()
            deps.reachy_mini.disable_motors()
            return {"status": "success", "action": "goto_sleep", "reason": reason}
        except Exception as e:
            logger.error("goto_sleep failed")
            return {"error": f"goto_sleep failed: {type(e).__name__}: {e}"}
