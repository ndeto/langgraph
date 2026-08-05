from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

from atlasai.application.quotas import QuotaService
from atlasai.application.sessions import SessionService
from atlasai.application.threads import ThreadService
from atlasai.application.usage import UsageService
from atlasai.infrastructure.telemetry import usage_record_from_callback
from atlasai.service.contracts import GraphRunner
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
from atlasai.web.schemas.thread import ThreadMessageRequest, ThreadResponse
from atlasai.web.streaming import (
    build_usage_event,
    encode_ndjson_event,
    extract_usage_payload,
    format_structured_ndjson_chunk,
)

router = APIRouter(prefix="/api/v1", tags=["threads"])


def _build_thread_response(thread) -> ThreadResponse:
    return ThreadResponse(
        thread_id=thread.thread_id,
        created_at=thread.created_at,
        expires_at=thread.expires_at,
        document_id=thread.document_id,
    )


@router.post("/threads", response_model=ThreadResponse, status_code=201)
def create_thread(
    response: Response,
    request_context: Annotated[RequestContext, Depends(get_request_context)],
    thread_service: Annotated[ThreadService, Depends(get_thread_service)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
):
    thread = thread_service.create_thread(
        user_id=request_context.session.session.user_id,
        expires_at=request_context.session.session.expires_at,
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
    return _build_thread_response(thread)


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
        raise HTTPException(status_code=429, detail=admission.reason)

    payload = {
        "user_input": input.user_input,
        "thread_id": thread.thread_id,
        "user_id": request_context.session.session.user_id,
    }
    cookie_value = session_service.serialize_session(
        session_service.with_active_thread(
            request_context.session.session,
            thread.thread_id,
        )
    )

    async def response_stream():
        usage_emitted = False
        try:
            async for chunk in graph_service.stream(payload):
                if isinstance(chunk, dict) and chunk.get("type") == "final":
                    record = usage_record_from_callback(
                        operation="agent",
                        payload=extract_usage_payload(chunk.get("data")),
                        trace_id=request_context.trace_id,
                        run_id=request_context.trace_id,
                    )
                    usage_service.record_usage(
                        user_id=request_context.session.session.user_id,
                        record=record,
                    )
                    yield encode_ndjson_event(build_usage_event(record))
                    yield encode_ndjson_event(
                        {"type": "done", "thread_id": thread.thread_id}
                    )
                    usage_emitted = True
                    continue

                formatted_chunk = format_structured_ndjson_chunk(chunk)
                if formatted_chunk is not None:
                    yield formatted_chunk

            if not usage_emitted:
                record = usage_record_from_callback(
                    operation="agent",
                    payload=None,
                    trace_id=request_context.trace_id,
                    run_id=request_context.trace_id,
                )
                usage_service.record_usage(
                    user_id=request_context.session.session.user_id,
                    record=record,
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
            usage_service.record_usage(
                user_id=request_context.session.session.user_id,
                record=record,
            )
            yield encode_ndjson_event(build_usage_event(record))
            yield encode_ndjson_event({"type": "error", "text": "Request failed."})
            usage_emitted = True
        finally:
            quota_service.release_agent_run(
                user_id=request_context.session.session.user_id
            )

    response = StreamingResponse(
        response_stream(),
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
