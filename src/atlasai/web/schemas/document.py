from datetime import datetime

from pydantic import BaseModel


class DocumentAcceptedResponse(BaseModel):
    """Accepted document response."""

    document_id: str
    job_id: str
    status: str


class DocumentResponse(BaseModel):
    """Document response."""

    document_id: str
    filename: str
    size_bytes: int
    status: str
    created_at: datetime
    expires_at: datetime
