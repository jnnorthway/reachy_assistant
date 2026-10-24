import base64
import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.tools.move_head import MoveHead, queue_aligned_look
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Settle covers camera lag. The extra hold keeps the pose so breathing cannot restore the forward view first.
LOOK_SETTLE_S = 0.4
LOOK_CAPTURE_HOLD_S = 0.75


class LookDirection(Tool):
    """Look in a direction, then photograph whatever the camera faces after the motion."""

    name = "look_direction"
    description = (
        "Turn to look left, right, up, down, or front, wait until that motion finishes, "
        "then take a picture of the new view. "
        "Use this for any question about what is to the left, right, up, down, or off to the side, "
        "including behind to one side. "
        "Do not answer those questions from memory, and do not use the camera tool for them: "
        "camera only photographs the current forward view and does not turn. "
        "Describe only the picture this tool returns."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "up", "down", "front"],
                "description": "Which way to look before taking the picture.",
            },
        },
        "required": ["direction"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Look, wait until the motion has settled, then return a JPEG of that view."""
        direction_raw = kwargs.get("direction")
        if not isinstance(direction_raw, str) or direction_raw not in MoveHead.DIRECTIONS:
            return {"error": "direction must be one of: left, right, up, down, front"}
        direction = direction_raw

        if not deps.camera_enabled:
            logger.error("Camera is disabled")
            return {"error": "Camera is disabled"}

        wait_s = deps.motion_duration_s + LOOK_SETTLE_S
        logger.info("Tool call: look_direction direction=%s wait_s=%.2f", direction, wait_s)

        try:
            # A move already in the queue would still be playing when this wait ends.
            deps.movement_manager.clear_move_queue()
            queue_aligned_look(deps, direction, hold_s=LOOK_SETTLE_S + LOOK_CAPTURE_HOLD_S)
            await asyncio.sleep(wait_s)
            jpeg_bytes = deps.reachy_mini.media.get_frame_jpeg()
        except Exception as e:
            logger.error("look_direction failed: %s", e)
            return {"error": f"look_direction failed: {type(e).__name__}: {e}"}

        if not isinstance(jpeg_bytes, bytes) or not jpeg_bytes:
            logger.error("No frame available from camera")
            return {"error": "No frame available"}

        return {
            "b64_im": base64.b64encode(jpeg_bytes).decode("utf-8"),
            "direction": direction,
            "captured_after_look": True,
            "note": "Picture taken after the look finished. Describe only this picture.",
        }
