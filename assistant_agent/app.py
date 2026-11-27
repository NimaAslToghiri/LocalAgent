from llm.qwen import QwenAssistant
from agent.graph_router import LangGraphAgent
from tools.internet_search_tool import SearchTool
import os
import logging
import datetime


os.environ["CUDA_PATH"] = "/usr/local/cuda-12.6"


def setup_logger(name: str) -> logging.Logger:
    """Configures a thread-safe, production-grade console and file logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # Console Handler (INFO level for clean user output)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File Handler (DEBUG level for deep trace troubleshooting)
        os.makedirs("logs", exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fh = logging.FileHandler(f"logs/{current_time}.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger


logger = setup_logger("AgentLogger")


def main():
    
    assistant = QwenAssistant(
        model_path="/home/nimaasltoghiri/Documents/self_projects/Agentic_Signal/models/Qwen3-4b-AWQ"
    )

    tools = [SearchTool()]

    agent = LangGraphAgent(
        llm=assistant,
        tools=tools
    )

    logger.info("Agent started")

    while True:

        user_input = input("You: ").strip()

        if user_input in ["exit", "quit"]:
            break

        if not user_input:
            continue

        response = agent.ask(user_input)

        print("\nAssistant:\n", response, "\n")


if __name__ == "__main__":
    main()