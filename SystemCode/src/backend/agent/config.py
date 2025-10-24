import os
from typing_extensions import TypedDict, Literal, Optional, Annotated

os.environ["GOOGLE_API_KEY"] = "AIzaSyB-qsnPlpBVdVWqvXDUwIHbxYhhkklfezY"

class Field(TypedDict):
    value: Optional[object]
    constraint_type: Optional[str]
    description: str

# Define the entire user needs template
class UserNeeds_basic(TypedDict):
    Price_Range_Min: Annotated[Field, "Integer: minimum price"]
    Price_Range_Max: Annotated[Field, "Integer: maximum price"]
    Location_MRT: Annotated[Field, "String: nearest MRT station"]
    Bedroom_Count: Annotated[Field, "Integer: number of bedrooms"]



class UserNeeds(TypedDict):
    # --- 基本预算信息 ---
    Price_Range_Min: Annotated[int, "Minimum price in SGD"]
    Price_Range_Max: Annotated[int, "Maximum price in SGD"]

    # --- 房屋与位置 ---
    Bedroom_Count: Annotated[int, "Number of bedrooms desired"]
    Location_MRT: Annotated[str, "Preferred MRT station or area near workplace"]

    # --- 出行与生活方式 ---
    Has_Car: Annotated[bool, "Whether the buyer owns a car"]
    Work_Location: Annotated[str, "Workplace or main commuting destination"]

    # --- 优先级偏好 (1~5 分) ---
    Priority_Affordability: Annotated[int, "Importance of affordability (1–5)"]
    Priority_MRT_Access: Annotated[int, "Importance of MRT accessibility (1–5)"]
    Priority_Bus_Access: Annotated[int, "Importance of bus accessibility (1–5)"]
    Priority_School_Proximity: Annotated[int, "Importance of proximity to schools (1–5)"]
    Priority_Park_Access: Annotated[int, "Importance of access to parks (1–5)"]
    Priority_Amenities: Annotated[int, "Importance of nearby amenities (1–5)"]

    # --- 家庭类型与身份信息 ---
    Household_Type: Annotated[str, "Type of household, e.g., 'Single', 'Couple', 'Family with kids'"]
    Buyer_Name: Annotated[str, "Name of the buyer"]
    Buyer_ID: Annotated[str, "Unique buyer ID"]
