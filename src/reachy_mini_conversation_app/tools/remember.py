import json
import logging
from pathlib import Path
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

MEMORY_FILE = Path.home() / ".reachy_memory.json"
MAX_FACTS = 100


def load_facts() -> list[str]:
    """Load persisted facts from disk."""
    try:
        if MEMORY_FILE.exists():
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [f for f in data if isinstance(f, str)]
    except Exception as e:
        logger.warning("Could not load memory file: %s", e)
    return []


def save_facts(facts: list[str]) -> None:
    """Persist facts to disk."""
    try:
        MEMORY_FILE.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save memory file: %s", e)


class Remember(Tool):
    """Store an important fact so Kevin can recall it in future conversations."""

    name = "remember"
    description = (
        "Remember an important fact for future conversations. "
        "Use this when you learn something worth keeping, such as a person's name, "
        "a pet's name, someone's preference, or any detail the user would expect you to recall later."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": (
                    "The fact to store, written as a short statement. "
                    "Examples: 'User's name is James', 'There is a cat named Max', "
                    "'User prefers quiet music'."
                ),
            },
        },
        "required": ["fact"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        fact = (kwargs.get("fact") or "").strip()
        if not fact:
            return {"error": "fact must be a non-empty string"}

        facts = load_facts()

        # Avoid exact duplicates
        if fact in facts:
            logger.info("remember: fact already stored: %s", fact[:120])
            return {"status": "already_known", "fact": fact}

        facts.append(fact)

        # Cap to avoid unbounded growth
        if len(facts) > MAX_FACTS:
            facts = facts[-MAX_FACTS:]

        save_facts(facts)
        logger.info("remember: stored fact: %s", fact[:120])
        return {"status": "remembered", "fact": fact}
