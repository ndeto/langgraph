import os
from datetime import datetime, time
from zoneinfo import ZoneInfo


WORK_ROUTES = {
    ("POST", "/invoke"),
    ("POST", "/api/v1/documents"),
}


def operating_hours_enforced() -> bool:
    """Return whether production operating hours are enabled."""

    return os.getenv("ATLASAI_ENFORCE_OPERATING_HOURS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_new_work_route(method: str, path: str) -> bool:
    """Identify endpoints that start model or ingestion work."""

    if (method.upper(), path) in WORK_ROUTES:
        return True
    return (
        method.upper() == "POST"
        and path.startswith("/api/v1/threads/")
        and path.endswith("/messages")
    )


def is_within_operating_hours(now: datetime | None = None) -> bool:
    """Check the configured daily operating window."""

    timezone = ZoneInfo(os.getenv("ATLASAI_OPERATING_TIMEZONE", "Africa/Nairobi"))
    local_now = now.astimezone(timezone) if now is not None else datetime.now(timezone)
    opens_at = time.fromisoformat(os.getenv("ATLASAI_OPERATING_START", "08:00"))
    closes_at = time.fromisoformat(os.getenv("ATLASAI_OPERATING_END", "22:00"))
    return opens_at <= local_now.time().replace(tzinfo=None) < closes_at
