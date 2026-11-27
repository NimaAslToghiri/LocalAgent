from typing import TypedDict

class AgentState(TypedDict):
    user_input: str
    refined_user_input: str
    route: str
    response: str
