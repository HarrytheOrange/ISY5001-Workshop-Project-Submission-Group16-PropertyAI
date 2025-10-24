from langchain.schema import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from config import UserNeeds
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from state import State
# from prompts import get_user_need_prompt
# from graph import UserNeedGraph
import pprint
from tools import HDBPriceTool, CondoPriceTool, OneMapRouteTool, HDBRecommendationTool, CondoRecommendationTool
import requests
import json


url = "https://www.onemap.gov.sg/api/auth/post/getToken"
payload = {
    "email": "e0792212@u.nus.edu",
    "password": "QingChen666!"
}
response = requests.post(url, json=payload)
OneMap_TOKEN = json.loads(response.text)["access_token"]

if response.status_code == 200:
    data = response.json()
    OneMap_TOKEN = data.get("access_token")
    # print("✅ OneMap Token:", OneMap_TOKEN)
else:
    print("❌ Failed to get token, status:", response.status_code)
    print("Response:", response.text)

hdb_tool = HDBPriceTool(model_path="../models/hdb_model.json", feature_path="../models/hdb_features.json")
condo_tool = CondoPriceTool(model_path="../models/condo_model.json", feature_path="../models/condo_features.json", encoder_path="../models/target_encoder_condo.pkl")
map_route_tool = OneMapRouteTool(token=OneMap_TOKEN)
hdb_recommendation_tool = HDBRecommendationTool(
    model_path=r"../models/ranker_lgbm.joblib",
    items_path=r"../../../data/item_matrix_merged.csv",
    pa_sim_path=r"../../../data/PA_centroid_similarity_0_1.csv"
)
condo_recommendation_tool = CondoRecommendationTool(
    model_path=r"../models/ranker_lgbm_condo.joblib",
    items_path=r"../../../data/item_matrix_merged_condo_reordered_20251019_180809.csv",
    pa_sim_path=r"../../../data/PA_centroid_similarity_0_1_condo.csv"
)

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

@tool
def warp_hdb_recommend(inputs: dict) -> dict:
    """
    Generate top HDB recommendations based on user preferences.

    Example Input:
    {
        "Budget_SGD": 1100000,
        "Preferred_Flat_Area_or_Room_Num": 5,
        "Preferred_Place": "JURONG EAST",
        "Priority_Distance_Proximity": 0.8,
        "Floor_Preference": 2,
        "NewHome_Preference": 0.5,
        "Priority_MRT_Access": 5,
        "Priority_Bus_Access": 5,
        "Priority_Amenities": 5,
        "Priority_School_Proximity": 5,
        "Priority_Park_Access": 5
    }

    Returns a JSON of top-N recommendations.
    """
    return hdb_recommendation_tool.recommend(inputs)

@tool
def warp_condo_recommend(inputs: dict) -> dict:
    """
    Generate top condo recommendations based on user preferences.

    Example Input:
    {
        "Budget_SGD": 4000000,
        "Preferred_Flat_Area_or_Room_Num": 100,
        "Preferred_Place": "JURONG EAST",
        "Priority_Distance_Proximity": 0.8,
        "Floor_Preference": 2,
        "NewHome_Preference": 0.5,
        "Priority_MRT_Access": 5,
        "Priority_Bus_Access": 5,
        "Priority_Amenities": 5,
        "Priority_School_Proximity": 5,
        "Priority_Park_Access": 5
    }

    Returns a JSON of top-N recommendations.
    """
    return condo_recommendation_tool.recommend(inputs)


# You are a helpful assistant collecting user housing needs.
# If the user isn't asking a specific question, talk with the user about the missing part of user's needs. But if the user is asking a specific question you can handle, prioritise the user's question.

class UserNeedGraph:
    def __init__(self, llm_name="google_genai:gemini-2.0-flash"):
        # self.state = initial_state or {"messages": [], "user_needs": {}}
        self.llm = init_chat_model(llm_name)
        self.tools = [self.update_user_need, warp_predict_hdb_price, warp_predict_condo_price, warp_get_sg_route, warp_hdb_recommend, warp_condo_recommend]
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
        You are a helpful assistant acting as a real estate agent.
        
        Current known user_needs:
        {state["user_needs"]}
        You can ask for missing ones.
        Each time, ask for only one missing field.
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
        }} (Besides describing the route to the user, also output a [[Image]] sign to let the frontend know.)

        If the user asks for **HDB** recommendations, use the hdb_recommend tool.
        Input example:
        {{
            "Budget_SGD": 1100000,
            "Preferred_Flat_Area_or_Room_Num": 5,
            "Preferred_Place": "JURONG EAST",
            "Priority_Distance_Proximity": 0.8,
            "Floor_Preference": 2,
            "NewHome_Preference": 0.5,
            "Priority_MRT_Access": 5,
            "Priority_Bus_Access": 5,
            "Priority_Amenities": 5,
            "Priority_School_Proximity": 5,
            "Priority_Park_Access": 5
        }} (If user not provide, you can make up a reasonable example.)

        If the user asks for **Condo** recommendations, use the condo_recommend tool.
        Input example:
        {{
            "Budget_SGD": 4000000,
            "Preferred_Flat_Area_or_Room_Num": 100,  #In square feets
            "Preferred_Place": "JURONG EAST",
            "Priority_Distance_Proximity": 0.8,
            "Floor_Preference": 2,
            "NewHome_Preference": 0.5,
            "Priority_MRT_Access": 5,
            "Priority_Bus_Access": 5,
            "Priority_Amenities": 5,
            "Priority_School_Proximity": 5,
            "Priority_Park_Access": 5
        }} (If user not provide, you can make up a reasonable example.)
        
        The user's task might involve multiple tools and analyzing the outputs to provide the best assistance.
        For example, the user might ask for property recommendations after providing their budget and location preferences.
        The user might also ask to compare several properties based on predicted prices and commute routes.
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



memory = InMemorySaver()
graph_system = UserNeedGraph()

graph = graph_system.graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

init_messages = [
    {"role": "system", "content": "You are a helpful assistant collecting user housing needs."},
    {"role": "user", "content": "Hello"}
]

state = {"messages": init_messages, "user_needs": user_needs}
graph.update_state(config, state)


while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        break
    
    state["messages"].append({"role": "user", "content": user_input})
    result = graph.invoke(state, config)
    # print(result)
    graph.update_state(config, result)


    ai_message = result["messages"][-1]
    if isinstance(ai_message, AIMessage):
        print(f"AI: {ai_message.content}")


    # print("-----")
    # print("    当前用户需求：")
    # for k, v in result["user_needs"].items():
    #     print(f"      - {k}: {v['value']}")
    # print("-----")

    if all(v["value"] is not None for v in result["user_needs"].values()):
        print("✅ 所有需求已收集完成！")
        break
