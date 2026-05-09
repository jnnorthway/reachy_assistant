import logging
import time
from typing import Any, Dict

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class WakeUp(Tool):
    """Wake Reachy Mini and enable motors."""

    name = "wake_up"
    description = "Wake Reachy Mini and enable motors."
    parameters_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional short reason for waking Reachy up.",
            }
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Call the Reachy SDK wake helper."""
        reason = kwargs.get("reason", "unspecified")
        logger.info("Tool call: wake_up reason=%s", reason)

        try:
            movement_manager = deps.movement_manager
            movement_manager.set_listening(False)
            movement_manager.clear_move_queue()

            # Explicitly re-enable torque first so wake motion can execute.
            deps.reachy_mini.enable_motors()
            # Torque enable is dispatched asynchronously; let it settle first.
            time.sleep(0.2)

            deps.reachy_mini.wake_up()

            # Ensure we end in a stable neutral/idle pose after wake animation.
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            deps.reachy_mini.goto_target(
                head=neutral_head_pose,
                antennas=[-0.1745, 0.1745],
                duration=0.8,
                body_yaw=0.0,
            )

            # Keep moving state active briefly so idle logic doesn't interrupt recovery.
            movement_manager.set_moving_state(3.6)
            return {"status": "success", "action": "wake_up", "reason": reason}
        except Exception as e:
            logger.error("wake_up failed")
            return {"error": f"wake_up failed: {type(e).__name__}: {e}"}
