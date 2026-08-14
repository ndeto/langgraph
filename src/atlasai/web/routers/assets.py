import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from atlasai.infrastructure.postgres_repositories import PostgresAssetRepository
from atlasai.web.dependencies import RequestContext, get_asset_repository, get_request_context

router = APIRouter(prefix="/api/v1/assets", tags=["assets"])
logger = logging.getLogger(__name__)


@router.get("/{asset_id}")
def get_asset(
    asset_id: str,
    request_context: Annotated[RequestContext, Depends(get_request_context)],
    assets: Annotated[PostgresAssetRepository, Depends(get_asset_repository)],
) -> Response:
    """Return an image only to its owning anonymous session."""

    asset = assets.get_owned_asset(
        asset_id=asset_id,
        user_id=request_context.session.session.user_id,
    )
    if asset is None:
        logger.warning(
            "asset_request_not_found trace_id=%s user_id=%s asset_id=%s",
            request_context.trace_id,
            request_context.session.session.user_id,
            asset_id,
        )
        raise HTTPException(status_code=404, detail="Asset not found.")
    logger.info(
        "asset_request_served trace_id=%s user_id=%s asset_id=%s document_id=%s "
        "mime_type=%s size_bytes=%s",
        request_context.trace_id,
        request_context.session.session.user_id,
        asset.asset_id,
        asset.document_id,
        asset.mime_type,
        asset.size_bytes,
    )
    return Response(
        content=asset.payload,
        media_type=asset.mime_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )
