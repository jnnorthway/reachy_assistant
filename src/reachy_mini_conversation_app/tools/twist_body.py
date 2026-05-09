import logging
from typing import Any, Dict, Literal

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

Direction = Literal["left", "right", "center"]

# Body yaw angles in degrees (positive = left, negative = right based on SDK convention)
BODY_YAW: Dict[str, float] = {
    "left": 30.0,
    "right": -30.0,
    "center": 0.0,
}


class TwistBody(Tool):
    """Twist the body left, right, or back to center."""

    name = "twist_body"
    description = "Rotate your body (torso) left, right, or back to center."
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "center"],
                "description": "Direction to twist the body: left, right, or center to return to neutral.",
            },
        },
        "required": ["direction"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Twist the body in a given direction."""
        direction_raw = kwargs.get("direction")
        if not isinstance(direction_raw, str) or direction_raw not in BODY_YAW:
            return {"error": "direction must be one of: left, right, center"}

        direction: Direction = direction_raw  # type: ignore[assignment]
        target_yaw = BODY_YAW[direction]
        logger.info("Tool call: twist_body direction=%s yaw=%.1f", direction, target_yaw)

        try:
            movement_manager = deps.movement_manager

            current_head_pose = deps.reachy_mini.get_current_head_pose()
            _, current_antennas = deps.reachy_mini.get_current_joint_positions()

            goto_move = GotoQueueMove(
                target_head_pose=current_head_pose,
                start_head_pose=current_head_pose,
                target_antennas=(current_antennas[0], current_antennas[1]),
                start_antennas=(current_antennas[0], current_antennas[1]),  # Skip body_yaw
                target_body_yaw=target_yaw,
                start_body_yaw=current_antennas[0],  # body_yaw is first in joint positions
                duration=deps.motion_duration_s,
            )

            movement_manager.queue_move(goto_move)
            movement_manager.set_moving_state(deps.motion_duration_s)

            return {"status": f"body twisted {direction}"}

        except Exception as e:
            logger.error("twist_body failed: %s", e)
            return {"error": f"twist_body failed: {type(e).__name__}: {e}"}
