from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from config import UserNeeds


class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_needs: UserNeeds

    