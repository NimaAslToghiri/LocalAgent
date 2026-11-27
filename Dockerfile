FROM vllm/vllm-openai:v0.21.0

WORKDIR /app
COPY requirements.txt .
RUN uv pip install --python /opt/venv/bin/python3 --no-cache -r requirements.txt
COPY assistant_agent/ assistant_agent/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SIGNAL_HOST=0.0.0.0 \
    MODEL_PATH=/model \
    SIGNAL_DATA_DIR=/data

EXPOSE 8000
# Replace the base image's OpenAI-compatible server with Signal.
ENTRYPOINT []
CMD ["/opt/venv/bin/python3", "-m", "assistant_agent.server"]
