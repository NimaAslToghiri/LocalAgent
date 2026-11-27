# Signal Agent

Signal is a local LangGraph learning project using Qwen3 4B AWQ. It answers directly, asks for clarification, or researches a question with DDGS search snippets.

## Run locally

Use Linux, Python 3.10, `uv`, and a working NVIDIA driver. Run `nvidia-smi` first. Install the versions in `requirements.txt` with a [vLLM wheel and PyTorch CUDA build compatible with your driver](https://docs.vllm.ai/en/v0.21.0/getting_started/installation/gpu/).

From the repository root:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install --torch-backend=auto -r requirements.txt
python download_model.py          # Skip if the 4B AWQ files are already present.
python -m assistant_agent.server
```

Open `http://127.0.0.1:8000`. The browser server binds only to the local machine. For the terminal interface, use `python -m assistant_agent.app` instead (running both loads two models). Type `/clear` to forget the terminal conversation.

Both entry points use the same optional root `.env` settings; see `.env.example`. An optional `HUGGINGFACE_TOKEN` stays in `.env`. The embedding model (`all-MiniLM-L6-v2`) loads on the CPU and downloads on first use if it is absent from the Hugging Face cache. If vLLM cannot reserve enough GPU memory, close other GPU apps or try `MAX_MODEL_LEN=4096`. The browser falls back to plain text if its Markdown or sanitization library cannot load.

If Signal needs a specific CUDA toolkit, add `CUDA_HOME=/usr/local/cuda-12.6` to the root `.env` (adjust the path for your installation). Both launch commands load it before PyTorch or vLLM, and Signal sets `CUDA_PATH` and puts the toolkit's `bin` directory on `PATH` for that process. Restart Signal after changing the setting. If PyTorch or vLLM is already loaded in a notebook or IDE session, restart that Python process too.

## Run with Docker

1. **Expose the GPU to Docker.** On Ubuntu or Pop!_OS, follow NVIDIA's Apt instructions to add the repository and install `nvidia-container-toolkit`. Then configure Docker and verify that a test container reports your GPU. Restarting Docker also restarts or stops any containers already running. [NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
   ```

2. **Prepare the mounts.** From the repository root, run `mkdir -p .signal` and check that `models/Qwen3-4b-AWQ/config.json` exists. If the model is missing, run `python download_model.py` in the host Python environment. Compose mounts the weights read-only and uses `.signal` for existing chats, logs, and caches. Set `MODEL_PATH` or `SIGNAL_DATA_DIR` in `.env` for custom host paths. If your Linux user ID or group ID differs from 1000, set `SIGNAL_UID` and `SIGNAL_GID` to the values reported by `id -u` and `id -g`.

3. **Build the image.** `Dockerfile` starts from vLLM 0.21.0 with its CUDA stack and installs the project's pinned packages. `.dockerignore` lets the build receive just the application and requirements. The model, chats, and Hugging Face token stay on the host.

   ```bash
   docker compose config --quiet
   docker compose build
   ```

4. **Start and verify.** Stop the native Signal process first so that Docker can use the GPU and port. Compose makes the GPU and mounts available and publishes the browser port on localhost. Open `http://127.0.0.1:8000` once startup completes. The first message may download the public embedding model into the mounted cache.

   ```bash
   docker compose up -d
   docker compose logs -f signal
   ```

5. **Stop or rebuild.** Press Ctrl+C to exit the log view. Run `docker compose down` to stop the container. The bind-mounted chats and weights persist. After changing the code or dependencies, use `docker compose up -d --build`.

## How a turn runs

```text
Route ─┬─ Chat or clarify → Done
       ├─ Read or count local chat history → Done
       └─ Resolve question → Lookup → Review snippets → Cited answer or explanation
                             ↑                  │
                             └── Retry once ────┘
```

The router classifies the output requested by the current turn. The chat assistant receives recent dialogue, relevant older exchanges, and trusted runtime context. The system clock supplies local and UTC timestamps for any task that uses the current date or time. Exact quotes and message counts use the saved transcript.

Before searching, Signal uses recent dialogue to resolve the question. A follow-up lookup starts from the resolved question, and every evidence review uses it even when the search phrasing changes on retry. This follows research on rewriting follow-up questions for conversational search. [QReCC paper](https://aclanthology.org/2021.naacl-main.44/)

Signal waits up to seven seconds per lookup and starts at most two lookups within its 20-second retry budget. Model inference can take additional time. It checks that each cited excerpt appears in the provider's snippet. Search results are static excerpts, so they cannot verify a live clock reading.

Conversation memory lives in `.signal/memory` by default. The browser remembers its conversation in the current tab and restores its last four exchanges on reconnect or refresh. Older, relevant exchanges in the same conversation may be recalled as labeled background. Clear chat removes that conversation from active memory. Logs in `.signal/signal.log` record routing and lookup outcomes without saving message text.

The chronological transcript is separate from the context supplied to the model. [Google ADK](https://adk.dev/sessions/) distinguishes session events from searchable memory, [OpenAI's Conversations API](https://developers.openai.com/api/docs/guides/conversation-state) stores messages and other items under a durable conversation ID, and [LangGraph](https://docs.langchain.com/oss/python/langgraph/persistence) separates thread state from longer-term stores. Signal applies these ideas using its existing local storage. After the router selects history, a separate model call selects the author, order, and limit for that operation. The history node reads the current conversation's stored records directly, before saving the new exchange. It never sends them to web search or relies on semantic recall to count or quote them. A transcript that exceeds the response size limit is labeled as partial.

The old `assistant_agent/conversation_db` and `assistant_agent/retrived_messages.txt` are not used or migrated. Older records lack conversation IDs, so importing them would mix histories. Those files were already tracked; `.gitignore` prevents new files from being added, but existing tracked files need to be removed from the index separately if you want them out of future commits.

## Quick checks

Run `python -m unittest -v evals.test_sources` to check the clock, search question, and retries without starting the model.

With Signal already running, run `python -m unittest -v evals.test_conversation` from the repository root in its Python environment. These live checks use the local model through the browser’s WebSocket endpoint. Each scenario gets a separate chat that is cleared afterward. They check switching between exact transcript requests, ordinary conversation and tasks that use earlier content. Testing both when a tool is needed and when it should stay idle follows [Anthropic’s guidance on agent evaluations](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

Try a greeting, a question that explicitly asks for web search, and an ambiguous follow-up. For search, check the source list and ask a question for which the snippets are insufficient. Open a fresh tab to verify conversation isolation, reload the original tab to restore its recent messages, then clear it and reload again.
