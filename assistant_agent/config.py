"""Shared settings for the browser, terminal, and model downloader."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def configure_cuda() -> None:
    """Expose the selected CUDA toolkit before PyTorch or vLLM is imported."""
    toolkit = os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH")
    if not toolkit:
        return

    os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"] = toolkit
    toolkit_bin = str(Path(toolkit) / "bin")
    paths = os.environ.get("PATH", os.defpath).split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join([toolkit_bin, *(path for path in paths if path != toolkit_bin)])


configure_cuda()


def project_path(value: str) -> Path:
    """Resolve relative settings from the repository, regardless of launch directory."""
    return (ROOT / Path(value).expanduser()).resolve()


MODEL_REPO = "Qwen/Qwen3-4B-AWQ"
MODEL_PATH = project_path(os.getenv("MODEL_PATH", "models/Qwen3-4b-AWQ"))
DATA_DIR = project_path(os.getenv("SIGNAL_DATA_DIR", ".signal"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
MAX_INPUT_CHARS = 12_000
PORT = int(os.getenv("SIGNAL_PORT", "8000"))


def configure_logging() -> None:
    """Keep a small rotating log without recording prompts or search content."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("assistant_agent")
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in (logging.StreamHandler(), RotatingFileHandler(
        DATA_DIR / "signal.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
    )):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
