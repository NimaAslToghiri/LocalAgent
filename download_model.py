import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


load_dotenv(".env")

# Explicitly defining cache dir to keep things structured
model_dir = os.path.expanduser("models/Qwen3-8B-AWQ")

print("Starting download to:", model_dir)
snapshot_download(
    repo_id="Qwen/Qwen3-8B-AWQ",  # Calling the dedicated AWQ repository directly
    local_dir=model_dir,
    token=os.getenv("HUGGINGFACE_TOKEN"),
    local_dir_use_symlinks=False,
    ignore_patterns=["*.gguf", "*.onnx"] # Ensure we only get what vLLM needs
)
print("Download complete!")
