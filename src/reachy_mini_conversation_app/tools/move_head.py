import math
import logging
from typing import Any, Dict, Literal

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

Direction = Literal["left", "right", "up", "down", "front"]

# Left and right yaw the base. The head pose uses the same yaw so the neck stays
# square to the body instead of doing the turn by itself. The base can go to about
# ±160 degrees; 50 is a clear look without approaching the stop.
_LOOK_BODY_YAW_RAD = {
    "left": math.radians(50),
    "right": math.radians(-50),
    "front": 0.0,
}
_PITCH_DEG = {"up": -30, "down": 30}


class MoveHead(Tool):
    """Look in a direction, turning the base for left and right."""

    name = "move_head"
    DIRECTIONS: tuple[Direction, ...] = ("left", "right", "up", "down", "front")
    description = (
        "Look left, right, up, down, or front. Left and right rotate the lower body "
        "so the whole robot turns. Up and down tilt only the head and keep the current body angle. "
        "Front returns the head and the body to center."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "up", "down", "front"],
            },
        },
        "required": ["direction"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Look in a direction, turning the base when looking left or right."""
        direction_raw = kwargs.get("direction")
        if not isinstance(direction_raw, str):
            return {"error": "direction must be a string"}
        direction: Direction = direction_raw  # type: ignore[assignment]
        logger.info("Tool call: move_head direction=%s", direction)

        try:
            movement_manager = deps.movement_manager
            current_head_pose = deps.reachy_mini.get_current_head_pose()
            current_head_joints, current_antennas = deps.reachy_mini.get_current_joint_positions()
            current_body_yaw = float(current_head_joints[0])
            # Up and down keep the base where it is. Anything else unknown returns to front.
            if direction in _LOOK_BODY_YAW_RAD or direction in _PITCH_DEG:
                body_yaw = _LOOK_BODY_YAW_RAD.get(direction, current_body_yaw)
                pitch_deg = _PITCH_DEG.get(direction, 0)
            else:
                body_yaw = 0.0
                pitch_deg = 0
            target = create_head_pose(0, 0, 0, 0, pitch_deg, math.degrees(body_yaw), degrees=True)

            goto_move = GotoQueueMove(
                target_head_pose=target,
                start_head_pose=current_head_pose,
                target_antennas=(0, 0),
                start_antennas=(current_antennas[0], current_antennas[1]),
                target_body_yaw=body_yaw,
                start_body_yaw=current_body_yaw,
                duration=deps.motion_duration_s,
            )

            movement_manager.queue_move(goto_move)
            movement_manager.set_moving_state(deps.motion_duration_s)

            return {"status": f"looking {direction}"}

        except Exception as e:
            logger.error("move_head failed: %s", e)
            return {"error": f"move_head failed: {type(e).__name__}: {e}"}
