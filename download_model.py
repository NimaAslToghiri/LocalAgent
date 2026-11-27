"""Download the model used by both Signal entry points."""

import os

from huggingface_hub import snapshot_download

from assistant_agent.config import MODEL_PATH, MODEL_REPO


def main() -> None:
    """Fetch Qwen3 4B AWQ into the configured directory using an optional token."""
    print(f"Downloading {MODEL_REPO} to {MODEL_PATH}")
    snapshot_download(
        repo_id=MODEL_REPO, local_dir=MODEL_PATH,
        token=os.getenv("HUGGINGFACE_TOKEN"), ignore_patterns=["*.gguf", "*.onnx"],
    )
    print("Download complete.")


if __name__ == "__main__":
    main()
