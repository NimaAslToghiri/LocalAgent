"""Terminal entry point and shared app startup."""

import logging

from . import config
from .agent.graph_router import LangGraphAgent
from .agent.memory import ConversationMemory
from .llm.qwen import GenerationError, QwenAssistant
from .tools.internet_search_tool import SearchTool


logger = logging.getLogger(__name__)
CLI_CONVERSATION = "local-terminal"


def build_agent() -> LangGraphAgent:
    """Load the configured 4B engine and connect it to search and local memory."""
    config.configure_logging()
    if not (config.MODEL_PATH / "config.json").is_file():
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}. Run python download_model.py first.")
    logger.info("Loading model from %s (context: %d tokens)", config.MODEL_PATH, config.MAX_MODEL_LEN)
    llm = QwenAssistant(config.MODEL_PATH, config.MAX_MODEL_LEN, config.GPU_MEMORY_UTILIZATION)
    memory = ConversationMemory(config.DATA_DIR / "memory")
    return LangGraphAgent(llm, SearchTool(), memory)


def main() -> None:
    """Run a terminal conversation that can be resumed or cleared across launches."""
    agent = build_agent()
    try:
        while True:
            text = input("You: ").strip()
            if text in ("quit", "exit"):
                break
            if not text:
                continue
            try:
                if text == "/clear":
                    agent.clear(CLI_CONVERSATION)
                    print("Chat cleared.")
                    continue
                reply = agent.ask(text, CLI_CONVERSATION)
                print(f"\nAssistant: {reply['content']}\n")
                for source in reply["sources"]:
                    print(f"[{source['id']}] {source['title']}: {source['url']}")
            except (ValueError, GenerationError) as error:
                print(str(error))
            except Exception as error:
                logger.error("Turn failed: %s", type(error).__name__)
                print("That operation failed. Please try again.")
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        agent.search.close()


if __name__ == "__main__":
    main()
