# agent/base_tool.py

from abc import ABC, abstractmethod

class BaseTool(ABC):
    """
    Abstract base class for all tools.
    Every tool must implement:
        - name: string identifier
        - can_handle(user_input: str): bool
        - run(user_input: str): str
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def can_handle(self, user_input: str) -> bool:
        pass

    @abstractmethod
    def run(self, user_input: str) -> str:
        pass