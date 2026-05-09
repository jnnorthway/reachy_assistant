import logging
import asyncio
import urllib.request
import json
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

_GEOLOCATION_TIMEOUT_S = 5


def _fetch_location() -> Dict[str, Any]:
    """Fetch approximate location via IP geolocation. No API key required."""
    url = "https://ipapi.co/json/"
    with urllib.request.urlopen(url, timeout=_GEOLOCATION_TIMEOUT_S) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    result: Dict[str, Any] = {}
    for key in ("city", "region", "country_name", "latitude", "longitude", "timezone"):
        if key in data:
            result[key] = data[key]
    return result


class GetLocation(Tool):
    """Return the approximate current location based on the device's IP address."""

    name = "get_location"
    description = (
        "Get the approximate current location of the robot (city, region, country) "
        "based on the device's IP address."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        logger.info("Tool call: get_location")
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_fetch_location),
                timeout=_GEOLOCATION_TIMEOUT_S + 1,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("get_location timed out")
            return {"error": "location lookup timed out"}
        except Exception as e:
            logger.warning("get_location failed: %s", e)
            return {"error": f"location lookup failed: {type(e).__name__}"}
