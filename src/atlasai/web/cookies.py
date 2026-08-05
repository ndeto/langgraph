from fastapi import Response

from atlasai.application.sessions import SessionService


def set_demo_session_cookie(
    response: Response,
    session_service: SessionService,
    *,
    cookie_value: str,
) -> None:
    """Write the signed demo session cookie."""

    response.set_cookie(
        key=session_service.settings.name,
        value=cookie_value,
        max_age=session_service.settings.max_age_seconds,
        expires=session_service.settings.max_age_seconds,
        httponly=True,
        secure=session_service.settings.secure,
        samesite=session_service.settings.same_site,
        path=session_service.settings.path,
    )
