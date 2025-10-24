import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
# import tensorflow as tf
import xgboost as xgb
from datetime import datetime
from langchain.tools import tool
import requests
import pytz
import requests
from openmap_utils import (
    get_latlng,
    is_in_sg,
    route_between,
    fetch_static_with_backoff,
    overlay_markers_on_png,
    collect_bbox_and_modes,
    build_route_summary,
    add_legend_right,
    simplify_onemap_route
)
# from hdb_recommendation_utils import main_infer
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from hdb_recommendation_utils import main_infer
from condo_recommendation_utils import main_infer_condo

class HDBPriceTool:
    def __init__(self, model_path="../models/hdb_model.json", feature_path="../models/hdb_features.json"):
        # 自动根据当前脚本目录计算相对路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.normpath(os.path.join(base_dir, model_path))
        feature_path = os.path.normpath(os.path.join(base_dir, feature_path))

        print("[DEBUG] Loading model from:", model_path)
        print("[DEBUG] Loading features from:", feature_path)

        with open(model_path, "rb") as f:
            model_bytes = f.read()

        booster = xgb.Booster()
        booster.load_model(bytearray(model_bytes))
        self.model = xgb.XGBRegressor()
        self.model._Booster = booster

        with open(feature_path, "r", encoding="utf-8") as f:
            self.X_columns = json.load(f)


    def predict_hdb_price(self, house_info: dict) -> float:
        """Predict HDB resale price based on house information.
        house_info example:
        {
            "month": "2021-11",
            "flat_type": "EXECUTIVE",
            "storey_range": "04 TO 06",
            "floor_area_sqm": 140,
            "lease_commence_date": 1997,
            "remaining_lease": "74 years 11 months",
            "postal": 650627
        }
        """
        print("calling predict_hdb_price tool...")
        df_input = pd.DataFrame([house_info])

        # 1️⃣ 处理楼层区间
        storey_order = [f"{i:02d} TO {i+2:02d}" for i in range(1, 100, 3)]
        storey_map = {v: i for i, v in enumerate(storey_order)}
        df_input["storey_range"] = df_input["storey_range"].map(storey_map).fillna(-1).astype(int)

        # 2️⃣ 处理日期
        df_input["month"] = pd.to_datetime(df_input["month"], format="%Y-%m", errors="coerce")
        df_input["year"] = df_input["month"].dt.year
        df_input["month_num"] = df_input["month"].dt.month
        df_input = df_input.drop(columns=["month"])

        # 3️⃣ 处理剩余租约
        def lease_to_months(x):
            if isinstance(x, str):
                years = int(re.search(r"(\d+)\s*year", x).group(1)) if re.search(r"(\d+)\s*year", x) else 0
                months = int(re.search(r"(\d+)\s*month", x).group(1)) if re.search(r"(\d+)\s*month", x) else 0
                return years * 12 + months
            return np.nan

        df_input["remaining_lease_months"] = df_input["remaining_lease"].apply(lease_to_months)
        df_input = df_input.drop(columns=["remaining_lease"])

        # 4️⃣ One-hot 编码
        df_input = pd.get_dummies(df_input, columns=["flat_type"], drop_first=True)

        # 5️⃣ 对齐特征
        df_input = df_input.reindex(columns=self.X_columns, fill_value=0)

        # 6️⃣ 预测价格
        y_pred = self.model.predict(df_input)[0]
        return float(y_pred)


class CondoPriceTool:
    """Condo resale price prediction tool."""

    def __init__(self, model_path: str, feature_path: str, encoder_path: str):
        """
        Initialize the CondoPriceTool with model, encoder, and feature config.

        Args:
            model_path (str): Path to the XGBoost model (.json or .ubj)
            feature_path (str): Path to the JSON file listing feature columns.
            encoder_path (str): Path to the target encoder pickle file.
        """
        # ----------------------------
        # ✅ 自动定位文件路径
        # ----------------------------
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.normpath(os.path.join(base_dir, model_path))
        feature_path = os.path.normpath(os.path.join(base_dir, feature_path))
        encoder_path = os.path.normpath(os.path.join(base_dir, encoder_path))

        print("[DEBUG] Loading model from:", model_path)
        print("[DEBUG] Loading features from:", feature_path)
        print("[DEBUG] Loading encoder from:", encoder_path)

        # ----------------------------
        # ✅ 安全加载模型（绕过 Windows UTF-8 问题）
        # ----------------------------
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        booster = xgb.Booster()
        booster.load_model(bytearray(model_bytes))  # 🔥 关键：直接读字节，不走路径
        self.model = xgb.XGBRegressor()
        self.model._Booster = booster

        # ----------------------------
        # ✅ 加载 TargetEncoder & 特征
        # ----------------------------
        with open(encoder_path, "rb") as f:
            self.te = pickle.load(f)

        with open(feature_path, "r", encoding="utf-8") as f:
            self.X_columns = json.load(f)

    def predict_condo_price(self, house_info: dict) -> float:
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
        current_year = datetime.now().year
        df_input = pd.DataFrame([house_info])

        # Feature engineering
        df_input["building_age"] = current_year - df_input["year_completed"]
        df_input["is_year_missing"] = df_input["year_completed"].isna().astype(int)

        # TargetEncoder for project_name
        df_input["project_name_enc"] = self.te.transform(df_input[["project_name"]])

        # One-hot encode categorical columns
        df_input = pd.get_dummies(df_input, columns=["type_of_sale", "market_segment"], drop_first=True)

        # Align feature order
        df_input = df_input.reindex(columns=self.X_columns, fill_value=0)

        # Predict log(psf)
        y_pred_log = self.model.predict(df_input)
        y_pred_psf = np.expm1(y_pred_log)[0]

        # Convert to total price
        price = y_pred_psf * df_input["area_sqft"].iloc[0]
        return float(price)


class OneMapRouteTool:
    """
    封装新加坡 OneMap 路线图生成工具。
    - 输入起点/终点名称
    - 自动生成路线图及摘要
    """

    def __init__(self, token: str = None):
        self.token = token or os.environ.get("ONEMAP_TOKEN", "")
        if not self.token:
            raise ValueError("Missing ONEMAP_TOKEN")

        self.default_a = os.environ.get("ONEMAP_START_NAME", "Chinatown Point")
        self.default_b = os.environ.get("ONEMAP_END_NAME", "Parc Clematis")

    def generate_route(self, a_name: str = None, b_name: str = None):
        DEFAULT_A = self.default_a
        DEFAULT_B = self.default_b

        a_name = a_name or DEFAULT_A
        b_name = b_name or DEFAULT_B
        INPUT_PLACES = {"A": a_name, "B": b_name}

        output = {"A": get_latlng(INPUT_PLACES["A"], self.token),
                  "B": get_latlng(INPUT_PLACES["B"], self.token)}

        # ---------- 校验 ----------
        errors = []
        for key, info in output.items():
            lat, lon = info.get("latitude"), info.get("longitude")
            if lat is None or lon is None:
                errors.append({"invalid_location": key, "name": info.get("name"), "reason": "geocode_not_found"})
            elif not is_in_sg(lat, lon):
                errors.append({
                    "invalid_location": key,
                    "name": info.get("name"),
                    "reason": "out_of_singapore_bounds",
                    "lat": lat,
                    "lon": lon
                })

        if errors:
            return {
                "status": "error",
                "stage": "geocoding",
                "message": "One or more locations failed validation.",
                "errors": errors,
                "input": output
            }

        # ---------- 路线 ----------
        route_data, now = route_between(output["A"]["latitude"], output["A"]["longitude"],
                                        output["B"]["latitude"], output["B"]["longitude"], self.token)

        if not route_data or "plan" not in route_data or not route_data["plan"].get("itineraries"):
            return {
                "status": "error",
                "stage": "routing",
                "message": "Routing failed or no itineraries returned.",
                "input": output,
                "raw": route_data
            }

        # ---------- 静态图 ----------
        raw_png = "../temp/static_map.png"
        res = fetch_static_with_backoff(route_data, self.token, raw_png, include_transfer_points=True)
        if not res["ok"]:
            return {
                "status": "error",
                "stage": "static_map",
                "message": "Static map generation failed after degrade attempts.",
                "error": res["error"],
                "input": output
            }

        # ---------- 标注 + 图例 ----------
        marked_png = "../temp/static_map_marked.png"
        overlay_markers_on_png(raw_png, marked_png, res["meta"], draw_start_end=True, draw_transfers=True)

        _, modes_present = collect_bbox_and_modes(route_data)
        summary = build_route_summary(route_data)
        final_png = "../temp/static_map_with_legend.png"
        add_legend_right(
            marked_png, final_png, modes_present,
            title="Legend", show_summary=True, summary=summary,
            legend_width=240
        )

        simplified_route_data = simplify_onemap_route(route_data)
        # print(json.dumps(simplified_route_data, indent=2))
        # ---------- 输出 ----------
        return {
            "status": "success",
            "message": "Route created and map generated.",
            "route_data": simplified_route_data,
            "saved_files": {
                "static_with_legend": final_png,
            },
        }


class HDBRecommendationTool:
    """
    A tool to generate top property recommendations based on user preferences.
    """

    def __init__(self, model_path: str, items_path: str, pa_sim_path: str):
        self.model_path = model_path
        self.items_path = items_path
        self.pa_sim_path = pa_sim_path
        self.feature_cols =['sim_budget', 'sim_budget_missing', 'sim_location', 'sim_location_missing', 'sim_mrt_access', 'sim_mrt_access_missing', 'sim_bus_access', 'sim_bus_access_missing', 'sim_amenities', 'sim_amenities_missing', 'sim_school', 'sim_school_missing', 'sim_area', 'sim_area_missing', 'sim_floor', 'sim_floor_missing', 'sim_newhome', 'sim_newhome_missing', 'sim_park_access', 'sim_park_access_missing']
    

    def recommend(self, user_input: dict, topn: int = 10, k_candidates: int = 300, use_location_rerank: bool = True) -> dict:
        """
        Recommend top property listings based on user input.
        """
        result = main_infer(
            model_path=self.model_path,
            feature_cols=self.feature_cols,
            items_path=self.items_path,
            pa_sim_path=self.pa_sim_path,
            item_place_col="Plan",
            topn=topn,
            k_candidates=k_candidates,
            use_location_rerank=use_location_rerank,
            user_input=user_input
        )
        print(result)
        return result



class CondoRecommendationTool:
    """
    A tool to generate top condo recommendations based on user preferences.
    """

    def __init__(self, model_path: str, items_path: str, pa_sim_path: str):
        self.model_path = model_path
        self.items_path = items_path
        self.pa_sim_path = pa_sim_path
        self.feature_cols = ['sim_budget', 'sim_budget_missing', 'sim_location', 'sim_location_missing', 'sim_mrt_access', 'sim_mrt_access_missing', 'sim_bus_access', 'sim_bus_access_missing', 'sim_amenities', 'sim_amenities_missing', 'sim_school', 'sim_school_missing', 'sim_area', 'sim_area_missing', 'sim_floor', 'sim_floor_missing', 'sim_newhome', 'sim_newhome_missing', 'sim_park_access', 'sim_park_access_missing']
    

    def recommend(self, user_input: dict, topn: int = 10, k_candidates: int = 300, use_location_rerank: bool = True) -> dict:
        """
        Recommend top property listings based on user input.
        """
        result = main_infer(
            model_path=self.model_path,
            feature_cols=self.feature_cols,
            items_path=self.items_path,
            pa_sim_path=self.pa_sim_path,
            item_place_col="Plan",
            topn=topn,
            k_candidates=k_candidates,
            use_location_rerank=use_location_rerank,
            user_input=user_input
        )
        print(result)
        return result
