import os
import json
# import google.generativeai as genai
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model


os.environ["GOOGLE_API_KEY"] = "AIzaSyB-qsnPlpBVdVWqvXDUwIHbxYhhkklfezY"

# -------------------------
# User needs template
# -------------------------
USER_NEEDS_TEMPLATE = {
    "Price_Range_Min": {"value": None, "constraint_type": None, "description": "A integer representing the minimum price."},
    "Price_Range_Max": {"value": None, "constraint_type": None, "description": "A integer representing the maximum price."},
    "Location_MRT": {"value": None, "constraint_type": None, "description": "A string representing the nearest MRT station."},
    "Bedroom_Count": {"value": None, "constraint_type": None, "description": "A integer representing the number of bedrooms."},
}


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = init_chat_model("google_genai:gemini-2.0-flash")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break