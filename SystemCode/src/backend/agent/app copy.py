from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.schema import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from state import State
from config import UserNeeds
from pathlib import Path
import os
from tools import HDBPriceTool, CondoPriceTool, OneMapRouteTool
from fastapi.staticfiles import StaticFiles
import requests
import json

# ================================================================
# 初始化 FastAPI
# ================================================================
app = FastAPI(title="User Need Chatbot API")


# 获取绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# 检查文件夹是否存在（自动创建）
os.makedirs(TEMP_DIR, exist_ok=True)

# 挂载静态文件（✅ 正确绝对路径）
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")


# ================================================================
# 允许跨域访问（前端可直接访问）
# ================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或指定前端来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# 定义输入输出模型
# ================================================================
class ChatRequest(BaseModel):
    user_input: str
    thread_id: str = "1"

class ChatResponse(BaseModel):
    ai_response: str
    user_needs: dict
    finished: bool


url = "https://www.onemap.gov.sg/api/auth/post/getToken"
payload = {
    "email": "e0792212@u.nus.edu",
    "password": "QingChen666!"
}
response = requests.post(url, json=payload)
OneMap_TOKEN = json.loads(response.text)["access_token"]

hdb_tool = HDBPriceTool(model_path="../models/hdb_model.json", feature_path="../models/hdb_features.json")
condo_tool = CondoPriceTool(model_path="../models/condo_model.json", feature_path="../models/condo_features.json", encoder_path="../models/target_encoder_condo.pkl")
map_route_tool = OneMapRouteTool(token=OneMap_TOKEN)

@tool
def warp_predict_condo_price(house_info: dict) -> float:
    """
        Predict the resale price of a condo unit.

        Args:
            house_info (dict): Must include:
                - project_name (str)
                - area_sqft (float)
                - type_of_sale (str)
                - tenure (int)
                - year_completed (float)
                - postal_district (int)
                - market_segment (str)
                - sale_year (int)
                - sale_month (int)
                - floor_level_num (float)

        Returns:
            float: predicted resale price (SGD)
        """
    return condo_tool.predict_condo_price(house_info)

@tool
def warp_predict_hdb_price(house_info: dict) -> float:
    """
    Predict the resale price of an HDB flat.

    Args:
        house_info (dict): Must include keys
            - month (e.g. "2021-11")
            - flat_type (e.g. "EXECUTIVE")
            - storey_range (e.g. "04 TO 06")
            - floor_area_sqm (float)
            - lease_commence_date (int)
            - remaining_lease (string, e.g. "74 years 11 months")
            - postal (int)

    Returns:
        float: predicted resale price in SGD.
    """
    # print("Predicting HDB price with info:\n\n\n")
    return hdb_tool.predict_hdb_price(house_info)

@tool
def warp_get_sg_route(inputs: dict) -> dict:
    """
    Generate a Singapore route map between two places using OneMap API.

    Input example:
    {
        "start": "Chinatown Point",
        "end": "Parc Clematis"
    }

    Returns JSON with map file paths and summary.
    """
    a = inputs.get("start")
    b = inputs.get("end")
    return map_route_tool.generate_route(a, b)


class UserNeedGraph:
    def __init__(self, llm_name="google_genai:gemini-2.0-flash"):
        # self.state = initial_state or {"messages": [], "user_needs": {}}
        self.llm = init_chat_model(llm_name)
        self.tools = [self.update_user_need, warp_predict_hdb_price, warp_predict_condo_price, warp_get_sg_route]
        # print("\n\n\nself.tools:", self.tools)
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
        Ask for missing ones. Each time, ask for only one missing field.
        Ask in the tone of a friendly and professional real estate agent.
        If you learn a new value, call the update_user_need tool.

        If the user asks about HDB resale price estimation, use the predict_hdb_price tool.
        The house_info is a dictionary like:
        {{
            "month": "2021-11",
            "flat_type": "EXECUTIVE",
            "storey_range": "04 TO 06",
            "floor_area_sqm": 140,
            "lease_commence_date": 1997,
            "remaining_lease": "74 years 11 months",
            "postal": 650627
        }}(If user not provide, you can make up a reasonable example)

        If the user asks about condo resale price estimation, use the predict_condo_price tool.
        The house_info is a dictionary like:
        {{
            "project_name": "NORMANTON PARK",
            "area_sqft": 721.19,
            "type_of_sale": "Resale",
            "tenure": 99,
            "year_completed": 2018,
            "postal_district": 19,
            "market_segment": "Outside Central Region",
            "sale_year": 2024,
            "sale_month": 12,
            "floor_level_num": 13
        }} (If user not provide, you can make up a reasonable example.)

        If the user asks about public transport route, use the get_sg_route tool.
        Input example:
        {{
        "start": "Chinatown Point",
        "end": "Parc Clematis"
        }}(Besides describing the route to the user, also output a [[Image]] sign to let the frontend know.)
        
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


user_needs: UserNeeds = {
    # --- 基本预算信息 ---
    "Price_Range_Min": {
        "value": None,
        "constraint_type": "integer",
        "description": "An integer representing the minimum price in SGD."
    },
    "Price_Range_Max": {
        "value": None,
        "constraint_type": "integer",
        "description": "An integer representing the maximum price in SGD."
    },

    # --- 房屋与位置 ---
    "Bedroom_Count": {
        "value": None,
        "constraint_type": "integer",
        "description": "An integer representing the desired number of bedrooms."
    },
    "Location_MRT": {
        "value": None,
        "constraint_type": "string",
        "description": "A string representing the preferred MRT station or nearby area."
    },

    # --- 出行与生活方式 ---
    "Has_Car": {
        "value": None,
        "constraint_type": "boolean",
        "description": "A boolean indicating whether the buyer owns a car."
    },
    "Work_Location": {
        "value": None,
        "constraint_type": "string",
        "description": "A string representing the buyer's workplace or main commuting destination."
    },

    # --- 优先级偏好 (1~5 分) ---
    "Priority_Affordability": {
        "value": None,
        "constraint_type": "integer",
        "description": "An integer (1–5) indicating the importance of affordability."
    },
    "Priority_MRT_Access": {
        "value": None,
        "constraint_type": "integer",
        "description": "An integer (1–5) indicating the importance of MRT accessibility."
    },
    "Priority_Bus_Access": {
        "value": None,
        "constraint_type": "integer",
        "description": "An integer (1–5) indicating the importance of bus accessibility."
    },
    "Priority_School_Proximity": {
        "value": None,
        "constraint_type": "integer",
        "description": "An integer (1–5) indicating the importance of proximity to schools."
    },
    "Priority_Park_Access": {
        "value": None,
        "constraint_type": "integer",
        "description": "An integer (1–5) indicating the importance of access to parks."
    },
    "Priority_Amenities": {
        "value": None,
        "constraint_type": "integer",
        "description": "An integer (1–5) indicating the importance of nearby amenities."
    },

    # --- 家庭类型与身份信息 ---
    "Household_Type": {
        "value": None,
        "constraint_type": "string",
        "description": "A string representing the type of household (e.g., 'Single', 'Couple', 'Family with kids')."
    },
    "Buyer_Name": {
        "value": None,
        "constraint_type": "string",
        "description": "A string representing the buyer's name."
    },
    "Buyer_ID": {
        "value": 123,
        "constraint_type": "string",
        "description": "A string representing the unique buyer ID."
    }
}




# ================================================================
# 初始化用户需求结构
# ================================================================
user_needs: UserNeeds = {
    "Price_Range_Min": {"value": None, "constraint_type": "integer", "description": "Min price in SGD."},
    "Price_Range_Max": {"value": None, "constraint_type": "integer", "description": "Max price in SGD."},
    "Bedroom_Count": {"value": None, "constraint_type": "integer", "description": "Desired number of bedrooms."},
    "Location_MRT": {"value": None, "constraint_type": "string", "description": "Preferred MRT or area."},
    "Has_Car": {"value": None, "constraint_type": "boolean", "description": "Whether the buyer owns a car."},
    "Work_Location": {"value": None, "constraint_type": "string", "description": "Workplace or main destination."},
    "Priority_Affordability": {"value": None, "constraint_type": "integer", "description": "Importance of affordability (1–5)."},
    "Priority_MRT_Access": {"value": None, "constraint_type": "integer", "description": "Importance of MRT access (1–5)."},
    "Priority_Bus_Access": {"value": None, "constraint_type": "integer", "description": "Importance of bus access (1–5)."},
    "Priority_School_Proximity": {"value": None, "constraint_type": "integer", "description": "Importance of nearby schools (1–5)."},
    "Priority_Park_Access": {"value": None, "constraint_type": "integer", "description": "Importance of park access (1–5)."},
    "Priority_Amenities": {"value": None, "constraint_type": "integer", "description": "Importance of amenities (1–5)."},
    "Household_Type": {"value": None, "constraint_type": "string", "description": "Household type (Single/Couple/Family)."},
    "Buyer_Name": {"value": None, "constraint_type": "string", "description": "Buyer's name."},
    "Buyer_ID": {"value": 123, "constraint_type": "string", "description": "Unique buyer ID."},
}

# ================================================================
# 初始化 Graph 系统
# ================================================================
memory = InMemorySaver()
graph_system = UserNeedGraph()
graph = graph_system.graph_builder.compile(checkpointer=memory)

# ================================================================
# FastAPI 接口定义
# ================================================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        state_snapshot = graph.get_state(config)
        state = state_snapshot.values if state_snapshot else {}

        # 初始化 state 结构
        if "messages" not in state:
            state["messages"] = [
                {"role": "system", "content": "You are a helpful assistant collecting user housing needs."},
                {"role": "user", "content": "Hello"},
            ]
        if "user_needs" not in state:
            state["user_needs"] = user_needs

        # 添加用户输入
        state["messages"].append({"role": "user", "content": request.user_input})

        # 执行 LLM Graph
        result = graph.invoke(state, config)
        graph.update_state(config, result)

        # 解析回复
        ai_message = result["messages"][-1]
        ai_response = ai_message.content if isinstance(ai_message, AIMessage) else str(ai_message)
        finished = all(v["value"] is not None for v in result["user_needs"].values())

        return ChatResponse(ai_response=ai_response, user_needs=result["user_needs"], finished=finished)

    except Exception as e:
        raise HTTPException(status_cod00e=500, detail=str(e))

@app.post("/reset/{thread_id}")
async def reset_thread(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    graph.update_state(config, {"messages": [], "user_needs": user_needs})
    return {"status": "ok", "message": f"Thread {thread_id} reset."}

# ================================================================
# 启动入口
# ================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
