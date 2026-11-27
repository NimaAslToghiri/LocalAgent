"""Serve the local chat interface and its conversation-scoped WebSocket."""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .app import build_agent
from .config import MAX_INPUT_CHARS, PORT
from .llm.qwen import ContextLimitError, GenerationError


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load one model and serialize work against the laptop's GPU."""
    app.state.agent = build_agent()
    app.state.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="signal-agent")
    app.state.turn_lock = asyncio.Lock()
    try:
        yield
    finally:
        app.state.worker.shutdown(wait=True, cancel_futures=True)
        app.state.agent.search.close()


app = FastAPI(title="Signal Agent", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])


@app.get("/")
async def serve_ui():
    """Serve the browser app from the package directory."""
    return FileResponse(Path(__file__).with_name("ui.html"), media_type="text/html")


def conversation_id(value) -> str:
    """Resume a valid random browser ID, or issue one for a new conversation."""
    if value is None:
        return str(uuid4())
    if not isinstance(value, str) or len(value) != 36:
        raise ValueError("Invalid conversation ID.")
    parsed = UUID(value)
    if parsed.version != 4:
        raise ValueError("Invalid conversation ID.")
    return str(parsed)


def parse_message(raw: str) -> dict:
    """Accept only object-shaped messages with a string type."""
    if len(raw) > 65_536:
        raise ValueError("Message is too large.")
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("type"), str):
        raise ValueError("Invalid message format.")
    return data


@app.websocket("/ws")
async def ws_chat(websocket: WebSocket):
    """Resume, clear, or send turns for this browser conversation only."""
    protocol = "https" if websocket.url.scheme == "wss" else "http"
    if websocket.headers.get("origin") != f"{protocol}://{websocket.url.netloc}":
        await websocket.close(code=1008)
        return
    await websocket.accept()
    state = websocket.app.state
    loop = asyncio.get_running_loop()
    try:
        try:
            hello = parse_message(await asyncio.wait_for(websocket.receive_text(), timeout=10))
            if hello["type"] != "hello":
                raise ValueError("Expected a conversation handshake.")
            session = conversation_id(hello.get("conversation_id"))
        except (ValueError, asyncio.TimeoutError):
            await websocket.close(code=1008)
            return

        try:
            turns = await loop.run_in_executor(state.worker, state.agent.recent, session)
        except Exception as error:
            logger.error("Could not restore conversation: %s", type(error).__name__)
            turns = []
        await websocket.send_json({"type": "ready", "conversation_id": session, "turns": turns})

        while True:
            raw = await websocket.receive_text()
            try:
                data = parse_message(raw)
                if data["type"] == "clear":
                    await loop.run_in_executor(state.worker, state.agent.clear, session)
                    result = {"type": "cleared"}
                elif data["type"] == "message":
                    content = data.get("content")
                    if not isinstance(content, str) or not content.strip() or len(content) > MAX_INPUT_CHARS:
                        result = {"type": "error", "content": f"Enter a message of up to {MAX_INPUT_CHARS:,} characters."}
                    elif state.turn_lock.locked():
                        result = {"type": "error", "content": "Signal is answering another request. Please try again shortly."}
                    else:
                        async with state.turn_lock:
                            await websocket.send_json({"type": "status", "status": "thinking"})
                            reply = await loop.run_in_executor(state.worker, state.agent.ask, content.strip(), session)
                        result = {"type": "response", **reply}
                else:
                    result = {"type": "error", "content": "Unsupported message type."}
            except WebSocketDisconnect:
                raise
            except (ContextLimitError, GenerationError) as error:
                result = {"type": "error", "content": str(error)}
            except ValueError:
                result = {"type": "error", "content": "Invalid message format."}
            except Exception as error:
                logger.error("Chat operation failed: %s", type(error).__name__)
                result = {"type": "error", "content": "That operation failed. Please try again."}
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, ws_max_size=65_536)
