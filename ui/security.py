"""Bound waveform request streams and keep all browser resources local."""

from __future__ import annotations

from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_BODY_BYTES = 4 * 1024 * 1024
HEADERS = {
    "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; "
                               "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
}


class BodyTooLargeError(ValueError):
    """A streamed request exceeded the fixed canonical-waveform size ceiling."""


async def bounded_body(receive: Receive) -> bytes | None:
    """Read at most four MiB, refusing a chunk before extending the bounded buffer."""
    body = bytearray()
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        chunk = message.get("body", b"")
        if len(body) + len(chunk) > MAX_BODY_BYTES:
            raise BodyTooLargeError("Waveform request exceeds four MiB")
        body.extend(chunk)
        if not message.get("more_body", False):
            return bytes(body)


class LocalSecurityMiddleware:
    """Apply fixed headers and replay a bounded body through the standard ASGI interface."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the local application without modifying its body parser or private request state."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject oversize waveform streams before validation or inference reads their content."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).update(HEADERS)
            await send(message)

        receiver = receive
        if scope["method"] == "POST":
            try:
                body = await bounded_body(receive)
            except BodyTooLargeError:
                response = JSONResponse(
                    {"detail": "Request exceeds the 4 MiB waveform limit."}, status_code=413,
                )
                await response(scope, receive, send_headers)
                return
            if body is None:
                return
            pending = True

            async def replay() -> Message:
                nonlocal pending
                if pending:
                    pending = False
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            receiver = replay
        await self.app(scope, receiver, send_headers)
