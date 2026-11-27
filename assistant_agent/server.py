"""
FastAPI backend with WebSocket support for the Agent Chat UI.
Run with: python server.py
Then open http://localhost:8000 in your browser.
"""

import asyncio
import json
import os
import logging
import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from llm.qwen import QwenAssistant
from agent.graph_router import LangGraphAgent
from tools.internet_search_tool import SearchTool

os.environ["CUDA_PATH"] = "/usr/local/cuda-12.6"


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        os.makedirs("logs", exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fh = logging.FileHandler(f"logs/{current_time}.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


logger = setup_logger("AgentLogger")

agent = None
executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("Initializing agent...")
    model_path = os.environ.get(
        "MODEL_PATH",
        "/home/nimaasltoghiri/Documents/self_projects/Agentic_Signal/models/Qwen3-4b-AWQ"
    )
    assistant = QwenAssistant(model_path=model_path)
    tools = [SearchTool()]
    agent = LangGraphAgent(llm=assistant, tools=tools)
    logger.info("Agent ready")
    yield
    executor.shutdown(wait=False)


app = FastAPI(title="Signal Agent", lifespan=lifespan)


@app.get("/")
async def serve_ui():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected via WebSocket")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if data.get("type") != "message":
                continue

            user_input = data.get("content", "").strip()
            if not user_input:
                continue

            logger.info(f"User: {user_input[:120]}...")
            await websocket.send_json({"type": "status", "status": "thinking"})

            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(executor, agent.ask, user_input)
                logger.info(f"Agent response: {len(response)} chars")
                await websocket.send_json({"type": "response", "content": response})
            except Exception as e:
                logger.error(f"Agent error: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "content": str(e)
                })
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)