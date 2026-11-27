"""Data shared by the graph, conversation store, and user interfaces."""

from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired


class Source(TypedDict):
    """A link supplied by the search provider and cited in a reply."""

    id: int
    title: str
    url: str


class Evidence(Source):
    """A source and the snippet available to the graph."""

    snippet: str


class Citation(Source):
    """A cited source and the exact text matched in its snippet."""

    excerpts: list[str]


class Reply(TypedDict):
    """The answer and its validated source links, if any."""

    content: str
    sources: list[Citation]
    plain_text: NotRequired[bool]


class Turn(TypedDict):
    """A complete user/assistant exchange stored as one record."""

    user: str
    reply: Reply


class HistoryRequest(BaseModel):
    """Select exact records from the current conversation."""

    model_config = ConfigDict(extra="forbid", strict=True)
    author: Literal["user", "assistant", "both"]
    order: Literal["oldest", "newest"]
    limit: int | None = Field(ge=1)


class AgentState(TypedDict, total=False):
    """The inputs and partial updates produced during one graph run."""

    user_input: str
    conversation_id: str
    recent: list[Turn]
    route: str
    history_action: Literal["count", "read"]
    research_question: str
    query: str
    attempts: int
    deadline: float
    evidence: list[Evidence]
    search_status: str
    retry: bool
    reply: Reply
