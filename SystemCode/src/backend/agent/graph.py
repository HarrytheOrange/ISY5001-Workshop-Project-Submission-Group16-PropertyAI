# agent/graph.py
# !!!
# Not functioning see app.py

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from state import State

class UserNeedGraph:
    def __init__(self, initial_state=None, llm_name="google_genai:gemini-2.0-flash"):
        # self.state = initial_state or {"messages": [], "user_needs": {}}
        self.llm = init_chat_model(llm_name)
        self.tools = [self.update_user_need]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.graph_builder = StateGraph(State)
        self._build_graph()
        # self.graph = self.graph_builder.compile()


    @tool
    def update_user_need(field_name: str, value) -> dict:
        """Update the user_needs dictionary in the state."""
        global state
        if field_name not in state["user_needs"]:
            raise ValueError(f"Unknown user_needs field: {field_name}")
        state["user_needs"][field_name]["value"] = value
        return {"user_needs": state["user_needs"]}


    def chatbot(self, state: State):
        system_prompt = f"""
        You are a helpful assistant collecting user housing needs.
        Current known user_needs:
        {state["user_needs"]}
        Ask for missing ones. 
        If you learn a new value, call the update_user_need tool.
        """

        new_message = self.llm_with_tools.invoke(
            [{"role": "system", "content": system_prompt}] + state["messages"]
        )

        return {
            "messages": state["messages"] + [new_message],
            "user_needs": state["user_needs"]
        }


    def _build_graph(self):
        self.graph_builder.add_node("chatbot", self.chatbot)

        tool_node = ToolNode(self.tools)
        self.graph_builder.add_node("tools", tool_node)

        self.graph_builder.add_conditional_edges(
            "chatbot", tools_condition, {"tools": "tools", END: END}
        )
        self.graph_builder.add_edge("tools", "chatbot")
        self.graph_builder.add_edge(START, "chatbot")
