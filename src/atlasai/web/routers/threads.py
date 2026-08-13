import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse

from atlasai.application.quotas import QuotaService
from atlasai.application.sessions import SessionService
from atlasai.application.threads import ThreadService
from atlasai.application.usage import UsageService
from atlasai.infrastructure.telemetry import usage_record_from_callback
from atlasai.infrastructure.telemetry import log_usage_resolution
from atlasai.service.contracts import GraphRunner, InvokePayload
from atlasai.web.cookies import set_demo_session_cookie
from atlasai.web.dependencies import (
    RequestContext,
    get_graph_service,
    get_quota_service,
    get_request_context,
    get_session_service,
    get_thread_service,
    get_usage_service,
)
from atlasai.web.schemas.thread import (
    ThreadCreateRequest,
    ThreadMessageRequest,
    ThreadMessageResponse,
    ThreadResponse,
)
from atlasai.web.streaming import (
    build_usage_event,
    encode_ndjson_event,
    extract_usage_payload,
    format_structured_ndjson_chunk,
    stream_with_keepalive,
)

router = APIRouter(prefix="/api/v1", tags=["threads"])
logger = logging.getLogger(__name__)


def _build_thread_response(thread) -> ThreadResponse:
    return ThreadResponse(
        thread_id=thread.thread_id,
        created_at=thread.created_at,
        expires_at=thread.expires_at,
        document_id=thread.document_id,
        messages=[],
    )


def _build_thread_message_response(message) -> ThreadMessageResponse:
    return ThreadMessageResponse(
        message_id=message.message_id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        assets=message.assets,
        status=message.status,
    )


@router.post("/threads", response_model=ThreadResponse, status_code=201)
def create_thread(
    response: Response,
    request_context: Annotated[RequestContext, Depends(get_request_context)],
    thread_service: Annotated[ThreadService, Depends(get_thread_service)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    input: ThreadCreateRequest = ThreadCreateRequest(),
):
    thread = thread_service.create_thread(
        user_id=request_context.session.session.user_id,
        expires_at=request_context.session.session.expires_at,
        document_id=input.document_id,
    )
    updated_session = session_service.with_active_thread(
        request_context.session.session,
        thread.thread_id,
    )
    set_demo_session_cookie(
        response,
        session_service,
        cookie_value=session_service.serialize_session(updated_session),
    )
    return _build_thread_response(thread)


@router.get("/threads/{thread_id}", response_model=ThreadResponse)
def get_thread(
    thread_id: str,
    request_context: Annotated[RequestContext, Depends(get_request_context)],
    thread_service: Annotated[ThreadService, Depends(get_thread_service)],
) -> ThreadResponse:
    thread = thread_service.get_owned_thread(
        user_id=request_context.session.session.user_id,
        thread_id=thread_id,
    )
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found.")
    messages = thread_service.list_owned_messages(
        user_id=request_context.session.session.user_id,
        thread_id=thread.thread_id,
    )
    return ThreadResponse(
        thread_id=thread.thread_id,
        created_at=thread.created_at,
        expires_at=thread.expires_at,
        document_id=thread.document_id,
        messages=[
            _build_thread_message_response(message)
            for message in (messages or [])
        ],
    )


@router.post("/threads/{thread_id}/messages")
async def post_thread_message(
    thread_id: str,
    input: ThreadMessageRequest,
    graph_service: Annotated[GraphRunner, Depends(get_graph_service)],
    request_context: Annotated[RequestContext, Depends(get_request_context)],
    thread_service: Annotated[ThreadService, Depends(get_thread_service)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    quota_service: Annotated[QuotaService, Depends(get_quota_service)],
    usage_service: Annotated[UsageService, Depends(get_usage_service)],
):
    thread = thread_service.get_owned_thread(
        user_id=request_context.session.session.user_id,
        thread_id=thread_id,
    )
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found.")

    admission = quota_service.claim_question(
        user_id=request_context.session.session.user_id
    )
    if not admission.allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": admission.message, "code": admission.code},
        )

    payload: InvokePayload = {
        "user_input": input.user_input,
        "thread_id": thread.thread_id,
        "user_id": request_context.session.session.user_id,
        "document_id": thread.document_id,
        "trace_id": request_context.trace_id,
    }
    cookie_value = session_service.serialize_session(
        session_service.with_active_thread(
            request_context.session.session,
            thread.thread_id,
        )
    )
    thread_service.append_owned_message(
        user_id=request_context.session.session.user_id,
        thread_id=thread.thread_id,
        role="user",
        content=input.user_input,
    )

    async def response_stream():
        usage_emitted = False
        assistant_text_parts: list[str] = []
        assistant_assets: list[dict[str, str]] = []
        try:
            async for chunk in graph_service.stream(payload):
                if isinstance(chunk, dict) and chunk.get("type") == "token":
                    token = chunk.get("data")
                    if isinstance(token, str):
                        assistant_text_parts.append(token)
                if isinstance(chunk, dict) and chunk.get("type") == "sources":
                    assets = chunk.get("data")
                    if isinstance(assets, list):
                        assistant_assets = list(
                            {
                                asset.get("asset_id"): asset
                                for asset in assets
                                if isinstance(asset, dict)
                                and isinstance(asset.get("asset_id"), str)
                            }.values()
                        )
                if isinstance(chunk, dict) and chunk.get("type") == "final":
                    usage_payload = extract_usage_payload(chunk.get("data"))
                    record = usage_record_from_callback(
                        operation="agent",
                        payload=usage_payload,
                        trace_id=request_context.trace_id,
                    )
                    log_usage_resolution(
                        source="threads.final",
                        payload=usage_payload,
                        record=record,
                    )
                    usage_service.record_usage(
                        user_id=request_context.session.session.user_id,
                        record=record,
                    )
                    if assistant_text_parts or assistant_assets:
                        thread_service.append_owned_message(
                            user_id=request_context.session.session.user_id,
                            thread_id=thread.thread_id,
                            role="assistant",
                            content="".join(assistant_text_parts).strip(),
                            assets=assistant_assets or None,
                        )
                    yield encode_ndjson_event(build_usage_event(record))
                    yield encode_ndjson_event(
                        {"type": "done", "thread_id": thread.thread_id}
                    )
                    usage_emitted = True
                    continue

                formatted_chunk = format_structured_ndjson_chunk(chunk)
                if formatted_chunk is None and isinstance(chunk, dict):
                    if chunk.get("type") == "rag_images":
                        markdown = chunk.get("data")
                        if isinstance(markdown, str) and markdown:
                            yield encode_ndjson_event(
                                {"type": "rag_images", "markdown": markdown}
                            )
                if formatted_chunk is not None:
                    yield formatted_chunk

            if not usage_emitted:
                record = usage_record_from_callback(
                    operation="agent",
                    payload=None,
                    trace_id=request_context.trace_id,
                    run_id=request_context.trace_id,
                )
                log_usage_resolution(
                    source="threads.missing-final",
                    payload=None,
                    record=record,
                )
                usage_service.record_usage(
                    user_id=request_context.session.session.user_id,
                    record=record,
                )
                if assistant_text_parts or assistant_assets:
                    thread_service.append_owned_message(
                        user_id=request_context.session.session.user_id,
                        thread_id=thread.thread_id,
                        role="assistant",
                        content="".join(assistant_text_parts).strip(),
                        assets=assistant_assets or None,
                    )
                yield encode_ndjson_event(build_usage_event(record))
                yield encode_ndjson_event({"type": "done", "thread_id": thread.thread_id})
                usage_emitted = True
        except Exception:
            record = usage_record_from_callback(
                operation="agent",
                payload=None,
                trace_id=request_context.trace_id,
                run_id=request_context.trace_id,
            )
            log_usage_resolution(
                source="threads.exception",
                payload=None,
                record=record,
            )
            usage_service.record_usage(
                user_id=request_context.session.session.user_id,
                record=record,
            )
            thread_service.append_owned_message(
                user_id=request_context.session.session.user_id,
                thread_id=thread.thread_id,
                role="assistant",
                content="Request failed.",
                status="error",
            )
            yield encode_ndjson_event(build_usage_event(record))
            yield encode_ndjson_event({"type": "error", "text": "Request failed."})
            usage_emitted = True
        finally:
            quota_service.release_agent_run(
                user_id=request_context.session.session.user_id
            )

    response = StreamingResponse(
        stream_with_keepalive(response_stream()),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Trace-Id": request_context.trace_id,
        },
    )
    set_demo_session_cookie(
        response,
        session_service,
        cookie_value=cookie_value,
    )
    return response
