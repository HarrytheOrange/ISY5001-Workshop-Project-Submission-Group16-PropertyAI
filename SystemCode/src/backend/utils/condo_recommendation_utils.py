from __future__ import annotations
import numpy as np
import pandas as pd
import json
from joblib import load as joblib_load
from typing import Optional, Dict, Union

# =========================================================
# 0) Utils
# =========================================================
def _norm_text(x) -> Optional[str]:
    if pd.isna(x):
        return None
    return str(x).strip().upper()

def gaussian_sim(user_val: float, item_val: float, sigma: float = 1.0) -> float:
    """Smooth 'closeness is better' similarity in [0,1]."""
    u, v = float(user_val), float(item_val)
    if sigma <= 0:
        return 1.0 if np.isclose(u, v) else 0.0
    d = u - v
    return float(np.exp(-(d * d) / (2.0 * sigma * sigma)))

def _clip_area(a: float, lo: float = 30.0, hi: float = 200.0) -> float:
    """Clamp area into the valid domain."""
    if pd.isna(a):
        return np.nan
    return float(np.clip(float(a), lo, hi))

def gaussian_rel(delta: float, sigma: float) -> float:
    """Gaussian on a relative delta; returns score in [0,1]."""
    if sigma <= 0:
        return 1.0 if np.isclose(delta, 0.0) else 0.0
    return float(np.exp(-(delta * delta) / (2.0 * sigma * sigma)))

def norm_priority(priority_raw: float) -> float:
    """Map raw priority in [1,5] → [0,1]."""
    if pd.isna(priority_raw):
        return 0.5
    return float(np.clip((float(priority_raw) - 1.0) / 4.0, 0.0, 1.0))

def sat_count(x: float, alpha: float = 1.0) -> float:
    """Monotonic saturation in [0,1]: s=1-exp(-alpha*x)."""
    if pd.isna(x) or x <= 0:
        return 0.0
    return float(1.0 - np.exp(-alpha * float(x)))

def blend_with_priority(item_score: float, priority_raw: float, neutral: float = 0.5, contrast: float = 1.0) -> float:
    """
    Make priority effect stronger:
      - map item_score from [0,1] to [-1,1] by s' = 2*s - 1
      - interpolate between neutral' (=0) and s' by p (0..1), then map back.
      - contrast>1 slightly amplifies the effect.
    """
    p = norm_priority(priority_raw)      # 0..1
    s = float(np.clip(item_score, 0.0, 1.0))
    s_ = (2*s - 1) * contrast            # [-contrast, +contrast]
    out_ = p * s_                        # 0→neutral(0), 1→s_
    out = (out_ / contrast + 1) / 2      # back to [0,1]
    # 混一点点原本 neutral，避免极端：
    return float(0.15 * neutral + 0.85 * out)

# =========================================================
# 1) Feature mappings — flat / floor / newhome (Gaussian)
# =========================================================
# ============================================
# 2) storey_range ↔ Floor_Preference (Gaussian)
#    HDB storey buckets expanded to 1..5 scale
# ============================================

_STOREY_TO_LEVEL = {
    "01 TO 03": 1.0,
    "04 TO 06": 1.2,
    "07 TO 09": 1.5,
    "10 TO 12": 1.7,
    "13 TO 15": 2.0,
    "16 TO 18": 2.4,
    "19 TO 21": 2.8,
    "22 TO 24": 3.0,
    "25 TO 27": 3.3,
    "28 TO 30": 3.7,
    "31 TO 33": 4.0,
    "34 TO 36": 4.0,
    "37 TO 39": 4.5,
    "40 TO 42": 4.5,
    "43 TO 45": 5.0,
    "46 TO 48": 5.0,
    "49 TO 51": 5.0,
    "52 TO 54": 5.0,
    "55 TO 57": 5.0
}

# --- 相对高斯：越接近越好（仅保留 rel_sigma 一个参数） ---
def score_area_closeness(user_area: float,
                         item_area: float,
                         rel_sigma: float = 0.12) -> float:
    """
    Return a similarity in [0,1] using a relative Gaussian on percentage diff:
        rel = (item - user) / user
        sim = exp( - rel^2 / (2 * rel_sigma^2) )
    仅一个超参数：rel_sigma（相对容差），默认 0.12 ≈ 12%。
    """
    # ——最小清洗：转成数值并裁剪到[30,200]，避免字符串/越界导致NaN——
    ua = pd.to_numeric(user_area, errors="coerce")
    ia = pd.to_numeric(item_area, errors="coerce")
    if not pd.isna(ua):
        ua = float(np.clip(ua, 30.0, 200.0))
    if not pd.isna(ia):
        ia = float(np.clip(ia, 30.0, 200.0))

    # === 临时调试：打印并“打断” ===
    # print(f"[DEBUG] area inputs -> user_area(ua)={ua}, item_area(ia)={ia}")
    # 方式A（推荐在notebook/脚本临时用）：打印后直接返回一个占位分数，观察输入是否为NaN
    # return 0.5  # ← 调试完成后删掉这一行

    # 方式B（强力中断）：打印后直接中止执行
    #raise SystemExit("DEBUG BREAK after printing ua/ia")

    # ——正式计算——
    if pd.isna(ua) or pd.isna(ia) or ua <= 0:
        return 0.5  # 缺值兜底（这就是你之前看到全是0.5的触发条件）

    rel = (ia - ua) / ua
    if rel_sigma <= 0:
        return 1.0 if np.isclose(rel, 0.0) else 0.0
    return float(np.exp(-(rel * rel) / (2.0 * rel_sigma * rel_sigma)))

def storey_to_level(storey_range: str, default_level: float = 2.0) -> float:
    """
    Map textual 'storey_range' to a continuous level in {1..5}.
    Unknown/missing values → default_level (mid-level = 2.5).
    """
    key = _norm_text(storey_range)
    if key in _STOREY_TO_LEVEL:
        return _STOREY_TO_LEVEL[key]
    # Try partial match (e.g. "35 TO 37" not in dict but similar)
    if isinstance(key, str):
        try:
            low = int(key.split(" TO ")[0])
            if low <= 6: return 1.0
            elif low <= 12: return 2.0
            elif low <= 21: return 3.0
            elif low <= 33: return 4.0
            else: return 5.0
        except Exception:
            return default_level
    return default_level

def score_floor(user_floor_pref: float,
                item_storey_range: str,
                sigma: float = 1.2) -> float:
    """
    Gaussian similarity between user's floor preference (1..5)
    and property's level (1..5).
    Slightly larger sigma to tolerate close floors.
    Returns a value in [0, 1].
    """
    item_level = storey_to_level(item_storey_range)
    return gaussian_sim(user_floor_pref, item_level, sigma=sigma)

def score_newhome(user_newhome_pref: float, sigma: float = 0.25) -> float:
    """
    NewHome_Preference: 1=new, 0.5=neutral, 0=resale.
    For HDB resale, item=0.0; neutral returns 1.0.
    """
    if pd.isna(user_newhome_pref):
        return 1.0
    val = float(user_newhome_pref)
    if np.isclose(val, 0.5):
        return 1.0
    return gaussian_sim(val, 0.0, sigma=sigma)

def compute_match_scores_gaussian(user_row: dict | pd.Series,
                                  item_row: dict | pd.Series,
                                  sigmas: dict | None = None) -> dict:
    """Return sim_area, sim_floor, sim_newhome in [0,1]（接口保持原样）."""
    s = sigmas or {"area": 0.12, "floor": 1.0, "newhome": 0.25}

    return {
        # 注意：你说用户侧字段现在叫 Preferred_Flat_Area_or_Room_Num floor_area_sqm，就把这里同步一下）
        "sim_area":    score_area_closeness(user_row["Preferred_Flat_Area_or_Room_Num"],
                                            item_row["floor_area_sqm"],
                                            s["area"]),
        "sim_floor":   score_floor(user_row["Floor_Preference"],
                                   item_row["storey_range"],
                                   s["floor"]),
        "sim_newhome": score_newhome(user_row["NewHome_Preference"],
                                     s["newhome"]),
    }

# =========================================================
# 2) Feature mappings — facility accessibility environment (higher is better)
# =========================================================
def score_park_access(priority_park_access: float, pk_500m_in: float) -> float:
    item_score = 1.0 if (not pd.isna(pk_500m_in) and pk_500m_in > 0) else 0.0
    return blend_with_priority(item_score, priority_park_access, neutral=0.5)

def score_bus_access(priority_bus_access: float, bus_200: float, bus_500: float,
                     alpha_200: float = 0.8, alpha_500: float = 0.4) -> float:
    s200 = sat_count(bus_200, alpha_200)
    s500 = sat_count(bus_500, alpha_500)
    item_score = float(np.clip((2.0/3.0)*s200 + (1.0/3.0)*s500, 0.0, 1.0))
    return blend_with_priority(item_score, priority_bus_access, neutral=0.5)

def score_mrt_access(priority_mrt_access: float, mrt_200: float, mrt_500: float,
                     alpha_200: float = 2.0, alpha_500: float = 1.2) -> float:
    s200 = sat_count(mrt_200, alpha_200)
    s500 = sat_count(mrt_500, alpha_500)
    item_score = float(np.clip((2.0/3.0)*s200 + (1.0/3.0)*s500, 0.0, 1.0))
    return blend_with_priority(item_score, priority_mrt_access, neutral=0.5)

def score_amenities(priority_amenities: float, hwkr_500m: float, mall_500m: float,
                    alpha_hwkr: float = 0.7, alpha_mall: float = 0.6) -> float:
    s_h = sat_count(hwkr_500m, alpha_hwkr)
    s_m = sat_count(mall_500m, alpha_mall)
    item_score = float(np.clip(0.25*s_h + 0.75*s_m, 0.0, 1.0))
    return blend_with_priority(item_score, priority_amenities, neutral=0.5)

def score_school_proximity(priority_school: float, gp_sch_1k: float, gp_sch_2k: float,
                           alpha_1k: float = 0.9, alpha_2k: float = 0.5) -> float:
    s1 = sat_count(gp_sch_1k, alpha_1k)
    s2 = sat_count(gp_sch_2k, alpha_2k)
    item_score = float(np.clip((2.0/3.0)*s1 + (1.0/3.0)*s2, 0.0, 1.0))
    return blend_with_priority(item_score, priority_school, neutral=0.5)

def compute_env_priority_scores(user_row: dict | pd.Series,
                                item_row: dict | pd.Series,
                                alphas: dict | None = None) -> dict:
    """
    Safe version: all user priorities fetched via .get(..., np.nan),
    so missing priorities are treated as neutral inside blend_with_priority().
    """
    alphas = alphas or {
        "bus_200": 0.8, "bus_500": 0.4,
        "mrt_200": 2.0, "mrt_500": 1.2,
        "hwkr": 0.7, "mall": 0.6,
        "sch_1k": 0.9, "sch_2k": 0.5
    }

    # ---- safely get user priorities (missing -> NaN -> neutral 0.5) ----
    p_park = user_row.get("Priority_Park_Access", np.nan)
    p_bus  = user_row.get("Priority_Bus_Access", np.nan)
    p_mrt  = user_row.get("Priority_MRT_Access", np.nan)
    p_am   = user_row.get("Priority_Amenities", np.nan)
    p_sch  = user_row.get("Priority_School_Proximity", np.nan)

    # ---- items (already .get with defaults) ----
    s_park = score_park_access(
        p_park,
        item_row.get("PK_500M_IN", 0)
    )

    s_bus = score_bus_access(
        p_bus,
        item_row.get("bus_200", 0),
        item_row.get("bus_500", 0),
        alpha_200=alphas["bus_200"],
        alpha_500=alphas["bus_500"]
    )

    s_mrt = score_mrt_access(
        p_mrt,
        item_row.get("mrt_200", 0),
        item_row.get("mrt_500", 0),
        alpha_200=alphas["mrt_200"],
        alpha_500=alphas["mrt_500"]
    )

    s_am = score_amenities(
        p_am,
        item_row.get("HWKR_500M", 0),
        item_row.get("MALL_500M", 0),
        alpha_hwkr=alphas["hwkr"],
        alpha_mall=alphas["mall"]
    )

    s_sch = score_school_proximity(
        p_sch,
        item_row.get("GP_SCH_1K", 0),
        item_row.get("GP_SCH_2K", 0),
        alpha_1k=alphas["sch_1k"],
        alpha_2k=alphas["sch_2k"]
    )

    return {
        "sim_park_access": s_park,
        "sim_bus_access":  s_bus,
        "sim_mrt_access":  s_mrt,
        "sim_amenities":   s_am,
        "sim_school":      s_sch,
    }

# =========================================================
# 3) Budget affinity (asymmetric Gaussian)
# =========================================================
def _asymmetric_gaussian(x: float, center: float, sigma_left: float, sigma_right: float) -> float:
    dx = float(x) - float(center)
    sigma = float(sigma_right if dx >= 0 else sigma_left)
    if sigma <= 0:
        return 1.0 if np.isclose(dx, 0.0) else 0.0
    return float(np.exp(-(dx * dx) / (2.0 * sigma * sigma)))

def budget_affinity_score(budget_sgd: float, resale_price: float, #always low a bit
                          target_under: float = 0.01, # bis a little
                          sigma_below: float = 0.08,
                          sigma_above: float = 0.12,
                          hard_clip: float = 0.60) -> float:
    """Smooth satisfaction between budget and price in [0,1]."""
    if pd.isna(budget_sgd) or budget_sgd <= 0 or pd.isna(resale_price) or resale_price <= 0:
        return 0.0
    d = (float(resale_price) - float(budget_sgd)) / float(budget_sgd)
    d = float(np.clip(d, -hard_clip, hard_clip))
    center = -abs(float(target_under))
    score = _asymmetric_gaussian(d, center, sigma_below, sigma_above)
    return float(np.clip(score, 0.0, 1.0))

# =========================================================
# 4) Location similarity with sensitivity
# =========================================================
def _norm_place(x: Optional[Union[str, float]]) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return str(x).strip().upper()

def load_pa_similarity_matrix(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
    else:
        df = pd.read_excel(path, index_col=0)
    df.index = df.index.to_series().map(_norm_place)
    df.columns = pd.Index([_norm_place(c) for c in df.columns])
    common = df.index.intersection(df.columns)
    df = df.loc[common, common].copy()
    df = df.clip(0.0, 1.0)
    df = 0.5 * (df + df.T)
    np.fill_diagonal(df.values, 1.0)
    return df

def _temperature_blend(base_sim: float, sensitivity: float,
                       neutral: float = 0.6, tau_low: float = 0.6, tau_high: float = 3.0) -> float:
    p = float(np.clip(0.0 if sensitivity is None else sensitivity, 0.0, 1.0))
    s = float(np.clip(0.0 if base_sim is None else base_sim, 0.0, 1.0))
    tau = tau_low + (tau_high - tau_low) * p
    s_temp = s ** tau
    return float(np.clip((1.0 - p) * neutral + p * s_temp, 0.0, 1.0))

def pa_location_similarity_with_sensitivity(
    user_place: Optional[str],
    item_place: Optional[str],
    sim_df: pd.DataFrame,
    distance_sensitivity: Optional[float],
    alias_map_user: Optional[Dict[str, str]] = None,
    alias_map_item: Optional[Dict[str, str]] = None,
    default_when_missing: float = 0.0,
    neutral: float = 0.6,
    tau_low: float = 0.6,
    tau_high: float = 3.0
) -> float:
    u = _norm_place(user_place)
    v = _norm_place(item_place)
    if alias_map_user and u is not None:
        u = _norm_place(alias_map_user.get(u, u))
    if alias_map_item and v is not None:
        v = _norm_place(alias_map_item.get(v, v))
    if u is None or v is None:
        base = float(default_when_missing)
    elif (u in sim_df.index) and (v in sim_df.columns):
        base = float(np.clip(sim_df.loc[u, v], 0.0, 1.0))
    else:
        base = float(default_when_missing)
    return _temperature_blend(base, distance_sensitivity, neutral, tau_low, tau_high)


def _norm_place(x: Optional[Union[str, float]]) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return str(x).strip().upper()

def load_pa_similarity_matrix(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
    else:
        df = pd.read_excel(path, index_col=0)
    df.index = df.index.to_series().map(_norm_place)
    df.columns = pd.Index([_norm_place(c) for c in df.columns])
    common = df.index.intersection(df.columns)
    df = df.loc[common, common].copy()
    df = df.clip(0.0, 1.0)
    df = 0.5 * (df + df.T)
    np.fill_diagonal(df.values, 1.0)
    return df

# --------------------------------------------------------
# 0) Small utilities
# --------------------------------------------------------
def _neutral_defaults() -> dict:
    """Neutral fallbacks when a feature is missing at inference."""
    return {"sim_budget": 0.8, "sim_location": 0.6, "default": 0.5}

def _pick(row, *candidates, default=""):
    """Pick the first present column from candidates."""
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return row[c]
    return default


# --------------------------------------------------------
# 1) Candidate retrieval (budget band ±20%)
# --------------------------------------------------------
def build_inference_candidates(items_df: pd.DataFrame,
                               budget: float,
                               k_candidates: int = 300,
                               seed: int = 2025) -> pd.DataFrame:
    """
    Retrieve k candidates within ±20% of budget; fallback to global sample.
    """
    band = items_df[(items_df["resale_price"] >= 0.80 * budget) &
                    (items_df["resale_price"] <= 1.20 * budget)]
    if len(band) < k_candidates:
        return items_df.sample(min(k_candidates, len(items_df)), random_state=seed)
    return band.sample(k_candidates, random_state=seed)


# --------------------------------------------------------
# 2) Feature computation (same sims as training; no masking)
# --------------------------------------------------------
def compute_features_for_user_items(user_row: pd.Series | dict,
                                    items_df: pd.DataFrame,
                                    sim_df_pa: pd.DataFrame,
                                    item_place_col: str = "Plan") -> pd.DataFrame:
    """
    Build the sim_* features on each (user, item) and keep rich item attributes
    for explanation. No masking at inference.
    """
    user = pd.Series(user_row) if not isinstance(user_row, pd.Series) else user_row
    rows = []
    neutr = _neutral_defaults()

    for _, it in items_df.iterrows():
        # --- sims used by model ---
        feats = {}
        feats["sim_budget"] = budget_affinity_score(user.get("Budget_SGD"), it["resale_price"])
        feats |= compute_match_scores_gaussian(user, it)     # sim_area, sim_floor, sim_newhome
        feats |= compute_env_priority_scores(user, it)       # sim_park_access, sim_bus_access, sim_mrt_access, sim_amenities, sim_school
        feats["sim_location"] = pa_location_similarity_with_sensitivity(
            user_place=user.get("Preferred_Place", None),
            item_place=it.get(item_place_col, None),
            sim_df=sim_df_pa,
            distance_sensitivity=user.get("Priority_Distance_Proximity", None),
            default_when_missing=0.0,
            neutral=neutr["sim_location"]
        )

        # --- carry item attributes for printing ---
        block  = _pick(it, "block")
        street = _pick(it, "street_name", "street_nam")
        town   = _pick(it, "town")
        pa     = _pick(it, item_place_col)
        storey = _pick(it, "storey_range")
        area   = _pick(it, "floor_area_sqm")
        lease  = _pick(it, "remaining_lease", "remaining_lease_mths", "remaining_lease_months")

        # nearby facility counts / flags
        def _int(x): 
            return int(x) if pd.notna(x) else 0
        rows.append({
            "item_id": it.get("item_id", it.get("id", _)),
            "block": block, "street": street, "town": town, "pa": pa,
            "storey_range": storey,
            "floor_area_sqm": area, "remaining_lease": lease,
            "resale_price": it.get("resale_price", np.nan),
            "mrt_200": _int(it.get("mrt_200", np.nan)),
            "mrt_500": _int(it.get("mrt_500", np.nan)),
            "bus_200": _int(it.get("bus_200", np.nan)),
            "bus_500": _int(it.get("bus_500", np.nan)),
            "MALL_500M": _int(it.get("MALL_500M", np.nan)),
            "HWKR_500M": _int(it.get("HWKR_500M", np.nan)),
            "HOSP_1K": _int(it.get("HOSP_1K", np.nan)),
            "GP_SCH_1K": _int(it.get("GP_SCH_1K", np.nan)),
            "GP_SCH_2K": _int(it.get("GP_SCH_2K", np.nan)),
            "PK_500M_IN": _int(it.get("PK_500M_IN", np.nan)),
            # model features (no masking at inference)
            **feats,
            "sim_budget_missing": 0,
            "sim_location_missing": 0,
            "sim_mrt_access_missing": 0,
            "sim_bus_access_missing": 0,
            "sim_amenities_missing": 0,
            "sim_school_missing": 0,
            "sim_area_missing": 0,
            "sim_floor_missing": 0,
            "sim_newhome_missing": 0,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------
# 3) Ensure feature alignment to model columns
# --------------------------------------------------------
def ensure_feature_alignment(df_feats: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Guarantee df contains every model feature; fill sims with neutrals and flags with 0.
    """
    neutr = _neutral_defaults()
    for c in feature_cols:
        if c not in df_feats.columns:
            if c.endswith("_missing"):
                df_feats[c] = 0
            elif c == "sim_budget":
                df_feats[c] = neutr["sim_budget"]
            elif c == "sim_location":
                df_feats[c] = neutr["sim_location"]
            else:
                df_feats[c] = neutr["default"]
    return df_feats[feature_cols]


# --------------------------------------------------------
# 4) Optional: location-sensitive re-ranking (inference-time)
# --------------------------------------------------------
def rerank_with_location_sensitivity(df_scored: pd.DataFrame,
                                     user_distance_sensitivity: float,
                                     loc_col: str = "sim_location",
                                     pred_col: str = "score",
                                     out_col: str = "score_final",
                                     hard_pref_place: str | None = None,
                                     boost_threshold: float = 0.75,
                                     boost_value: float = 0.05):
    """
    Blend model score with location similarity:
        score_final = (1 - alpha) * score + alpha * sim_location
    where alpha grows with user sensitivity. Optionally boost near-preferred places.
    """
    alpha = 0.3 + 0.5 * float(np.clip(user_distance_sensitivity or 0.0, 0.0, 1.0))
    df = df_scored.copy()
    df[out_col] = (1.0 - alpha) * df[pred_col] + alpha * df[loc_col]

    if hard_pref_place is not None:
        mask = df[loc_col] >= float(boost_threshold)
        df.loc[mask, out_col] += float(boost_value)
    return df.sort_values(out_col, ascending=False), alpha



# ---------- Utility helpers ----------
def _pick_metric(row: pd.Series, candidates: list[str]) -> float:
    """Try several possible column names; return the first valid float value."""
    for c in candidates:
        if c in row and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                pass
    return np.nan

def _should_block_top1(row: pd.Series,
                       sim_threshold: float = 0.30,
                       model_threshold: float = 1.0,
                       na_counts_as_fail: bool = True) -> tuple[bool, dict]:
    """Return (should_block, value_dict)."""
    budget = _pick_metric(row, ["sim_budget", "budget", "budget_sim"])
    loc    = _pick_metric(row, ["sim_location", "location", "loc"])
    area   = _pick_metric(row, ["sim_area", "area", "area_sim"])
    model  = _pick_metric(row, ["score", "Model", "model", "pred", "predict_score"])

    vals = {"budget": budget, "loc": loc, "area": area, "model": model}

    if na_counts_as_fail and (np.isnan(model) or any(np.isnan(v) for v in [budget, loc, area])):
        return True, vals

    bad_sim   = any(v < sim_threshold for v in [budget, loc, area] if not np.isnan(v))
    bad_model = (not np.isnan(model)) and (model < model_threshold)
    return (bad_sim or bad_model), vals

# --------------------------------------------------------
# 5) Recommend & pretty-print
# --------------------------------------------------------
# ---------- Helper functions ----------
def _pick_metric(row: pd.Series, candidates: list[str]) -> float:
    """Try multiple candidate column names and return the first valid float value."""
    for c in candidates:
        if c in row and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                pass
    return np.nan

def _should_block_top1(row: pd.Series,
                       sim_threshold: float = 0.30,
                       model_threshold: float = 1.0,
                       na_counts_as_fail: bool = True) -> tuple[bool, dict]:
    """Check whether Top-1 should be blocked based on similarity and model thresholds."""
    budget = _pick_metric(row, ["sim_budget", "budget", "budget_sim"])
    loc    = _pick_metric(row, ["sim_location", "location", "loc"])
    area   = _pick_metric(row, ["sim_area", "area", "area_sim"])
    model  = _pick_metric(row, ["score", "Model", "model", "pred", "predict_score"])
    vals = {"budget": budget, "loc": loc, "area": area, "model": model}

    # If NaNs exist, optionally count as failure
    if na_counts_as_fail and (np.isnan(model) or any(np.isnan(v) for v in [budget, loc, area])):
        return True, vals

    bad_sim   = any(v < sim_threshold for v in [budget, loc, area] if not np.isnan(v))
    bad_model = (not np.isnan(model)) and (model < model_threshold)
    return (bad_sim or bad_model), vals


# ---------- Main function ----------
def recommend_for_user(model,
                       feature_cols: list[str],
                       user_input: dict,
                       items_df: pd.DataFrame,
                       sim_df_pa: pd.DataFrame,
                       item_place_col: str = "Plan",
                       topn: int = 5,
                       k_candidates: int = 300,
                       seed: int = 2025,
                       use_location_rerank: bool = True,
                       sim_threshold=0.4,
                       model_threshold=1.0,
                       warn_threshold: float = 0.30) -> pd.DataFrame:
    """
    Full inference pipeline:
      1. Retrieve candidate properties
      2. Compute user-item features
      3. Predict ranking scores
      4. (Optional) location-sensitive re-ranking
      5. Produce a structured JSON-like payload for LLMs (and print summary)
    """
    # 1) Candidate retrieval
    budget = float(user_input["Budget_SGD"])
    cand = build_inference_candidates(items_df, budget, k_candidates=k_candidates, seed=seed)

    # 2) Feature computation
    feats_df = compute_features_for_user_items(user_input, cand, sim_df_pa, item_place_col=item_place_col)

    # 3) Model prediction
    X = ensure_feature_alignment(feats_df.copy(), feature_cols)
    feats_df["score"] = model.predict(X)

    # 4) Optional location re-ranking
    if use_location_rerank:
        user_p = float(user_input.get("Priority_Distance_Proximity", 0.0))
        pref_place = user_input.get("Preferred_Place", None)
        feats_df, alpha_used = rerank_with_location_sensitivity(
            feats_df, user_distance_sensitivity=user_p,
            loc_col="sim_location", pred_col="score", out_col="score_final",
            hard_pref_place=pref_place, boost_threshold=0.75, boost_value=0.05
        )
        rank_col = "score_final"
    else:
        alpha_used = None
        rank_col = "score"

    recs = feats_df.sort_values(rank_col, ascending=False).head(topn).copy()

    # ---------- Construct base structured payload ----------
    user_summary = {
        k: user_input.get(k, None) for k in [
            "Budget_SGD","Preferred_Flat_Area_or_Room_Num","Preferred_Place","Floor_Preference","NewHome_Preference",
            "Priority_MRT_Access","Priority_Bus_Access","Priority_Amenities",
            "Priority_School_Proximity","Priority_Park_Access","Priority_Distance_Proximity"
        ]
    }
    result_payload = {
        "status": None,
        "meta": {
            "topn": int(topn),
            "use_location_rerank": bool(use_location_rerank),
            "location_alpha": float(alpha_used) if alpha_used is not None else None,
            "thresholds": {
                "block_sim_threshold": float(sim_threshold),
                "block_model_threshold": float(model_threshold),
                "warn_threshold": float(warn_threshold),
            },
            "user": user_summary,
        },
        "fail_reasons": None,   # Explanation if blocked or no result
        "items": []        # Structured property results
    }

    # ---------- Step 1: No candidates ----------
    if recs.empty:
        result_payload["status"] = "no_result"
        result_payload["fail_reasons"] = {
            "type": "no_candidates_after_rerank",
            "message": "No suitable properties found under current preferences. Try increasing budget or relaxing filters."
        }
        print(json.dumps(result_payload, ensure_ascii=False, indent=2))
        return recs

    # ---------- Step 2: Top-1 fails quality threshold ----------
    need_block, vals = _should_block_top1(recs.iloc[0],
                                          sim_threshold=sim_threshold,
                                          model_threshold=model_threshold)
    if need_block:
        result_payload["status"] = "blocked_top1"
        result_payload["reasons"] = {
            "type": "top1_fails_quality_threshold",
            "triggered_values": {k: (None if np.isnan(v) else float(v)) for k, v in vals.items()},
            "required": {"sim_min": float(sim_threshold), "model_min": float(model_threshold)},
            "message": "Top-1 fails quality threshold; consider increasing budget, expanding preferred location, or loosening size/type preference."
        }
        print(json.dumps(result_payload, ensure_ascii=False, indent=2))
        return recs.head(0)

    # ---------- Step 3: Build structured items ----------
    def _safe_get(obj, name, default=None):
        try:
            v = getattr(obj, name)
            return v if v is not None else default
        except Exception:
            return default

    sim_fields = [
        "sim_budget","sim_location","sim_area",
        "sim_mrt_access","sim_bus_access","sim_amenities","sim_school","sim_floor","sim_newhome"
    ]

    # Small helper to pack nearby-facilities info
    nearby_pack = lambda r: {
        "mrt_200": int(_safe_get(r, "mrt_200", 0) or 0),
        "mrt_500": int(_safe_get(r, "mrt_500", 0) or 0),
        "bus_200": int(_safe_get(r, "bus_200", 0) or 0),
        "bus_500": int(_safe_get(r, "bus_500", 0) or 0),
        "hawker_500m": int(_safe_get(r, "HWKR_500M", 0) or 0),
        "mall_500m": int(_safe_get(r, "MALL_500M", 0) or 0),
        "school_1km": int(_safe_get(r, "GP_SCH_1K", 0) or 0),
        "school_2km": int(_safe_get(r, "GP_SCH_2K", 0) or 0),
        "park_500m_in": int(_safe_get(r, "PK_500M_IN", 0) or 0),
    }

    items_struct = []
    for i, r in enumerate(recs.itertuples(index=False), 1):
        addr = " ".join(str(x) for x in [_safe_get(r, "block", ""), _safe_get(r, "street", "")] if str(x))
        loc  = _safe_get(r, "pa", "") or _safe_get(r, "town", "")

        # Collect similarity scores
        sims = {}
        for s in sim_fields:
            if s in recs.columns:
                val = getattr(r, s, np.nan)
                sims[s.replace("sim_", "")] = None if pd.isna(val) else float(val)

        base_score  = _safe_get(r, "score", np.nan)
        final_score = _safe_get(r, rank_col, np.nan)

        # Identify features below warn threshold
        warn_keys = ["mrt_access", "bus_access", "amenities", "school", "floor"]
        low_flags = [k for k in warn_keys if (k in sims and sims[k] is not None and sims[k] < float(warn_threshold))]

        item_entry = {
            "rank": i,
            "attributes": {
                "address": addr,
                "location": loc,
                "flat_type": _safe_get(r, "flat_type", ""),
                "storey_range": _safe_get(r, "storey_range", ""),
                "floor_area_sqm": _safe_get(r, "floor_area_sqm", None),
                "remaining_lease": _safe_get(r, "remaining_lease", None),
                "resale_price": None if pd.isna(_safe_get(r, "resale_price", np.nan)) else float(_safe_get(r, "resale_price", np.nan)),
                "nearby": nearby_pack(r),
            },
            "scores": {
                "model": None if pd.isna(base_score) else float(base_score),
                "final": None if pd.isna(final_score) else float(final_score),
                "loc": None if pd.isna(_safe_get(r, "sim_location", np.nan)) else float(_safe_get(r, "sim_location", np.nan)),
                "sims": sims,
                "low_flags": low_flags,  # which features fall below warn_threshold
            }
        }
        items_struct.append(item_entry)

    result_payload["status"] = "ok"
    result_payload["items"] = items_struct


    lines = []
    # lines.append("\n=== Human-readable preview ===")
    lines.append(
        f"User: budget={user_summary.get('Budget_SGD')}, "
        f"place={user_summary.get('Preferred_Place')}, "
        f"flat_type={user_summary.get('Preferred_Room_Num')}, "
        f"floor_pref={user_summary.get('Floor_Preference')}, "
        f"newhome_pref={user_summary.get('NewHome_Preference')}"
    )
    if use_location_rerank:
        lines.append(f"Location alpha: {alpha_used:.2f}")
    lines.append(f"TOP-{topn} recommendations")

    for item in items_struct:
        a, s = item["attributes"], item["scores"]
        lines.append(f"\n#{item['rank']} | {a['address']} [{a['location']}]")
        lines.append(f"   Type/Floor: {a['flat_type']} / {a['storey_range']}")

        resale_price = a.get("resale_price")
        resale_price_str = f"${resale_price:,.0f}" if resale_price else "$0"

        lines.append(
            f"   Price: {resale_price_str} | "
            f"Model: {s.get('model', 0):.4f} | "
            f"Loc: {0 if s.get('loc') is None else s['loc']:.2f} | "
            f"Final: {s.get('final', 0):.4f}"
        )

        # Similarity evaluations
        eval_keys = ["mrt_access", "bus_access", "amenities", "school", "floor"]
        eval_pairs = []
        for k in eval_keys:
            val = s["sims"].get(k)
            if val is None:
                eval_pairs.append(f"{k}: NA")
            else:
                eval_pairs.append(f"{k}: {val:.2f}")
        lines.append("   match eval → " + " | ".join(eval_pairs))

        if s.get("low_flags"):
            lines.append(
                f"   ⚠ unmet features (sim<{warn_threshold:.2f}): "
                + ", ".join(s["low_flags"])
            )

    return "\n".join(lines)

    # ---------- Print structured payload and human-readable preview ----------
    # print(json.dumps(result_payload, ensure_ascii=False, indent=2))

    print("\n=== Human-readable preview ===")
    print(f"User: budget={user_summary['Budget_SGD']}, place={user_summary['Preferred_Place']}, "
          f"area={user_summary['Preferred_Flat_Area_or_Room_Num']}, floor_pref={user_summary['Floor_Preference']}, "
          f"newhome_pref={user_summary['NewHome_Preference']}")
    if use_location_rerank:
        print(f"location alpha: {alpha_used:.2f}")
    print(f"TOP-{topn} recommendations")

    for item in items_struct:
        a, s = item["attributes"], item["scores"]
        print(f"\n#{item['rank']} | {a['address']} [{a['location']}]")
        print(f"   Type/Floor: {a['flat_type']} / {a['storey_range']}")
        if a["floor_area_sqm"] or a["remaining_lease"]:
            print(f"   Area: {a['floor_area_sqm']} sqm   Lease: {a['remaining_lease']}")
        print(f"   Price: ${0 if a['resale_price'] is None else a['resale_price']:,.0f} | "
              f"Model: {s['model']:.4f} | Loc: {0 if s['loc'] is None else s['loc']:.2f} | Final: {s['final']:.4f}")

        eval_keys = ["mrt_access","bus_access","amenities","school","floor"]
        eval_pairs = []
        for k in eval_keys:
            val = s["sims"].get(k)
            if val is None:
                eval_pairs.append(f"{k}: NA")
            else:
                eval_pairs.append(f"{k}: {val:.2f}")

        print("   match eval → " + " | ".join(eval_pairs))
        if s["low_flags"]:
            print(f"   ⚠ unmet features (sim<{warn_threshold:.2f}): {', '.join(s['low_flags'])}")

    # Return DataFrame (you can change to 'return result_payload' if you want to return structured data directly)
    return recs



def main_infer_condo(
    # Data sources:
    items_path: str,
    pa_sim_path: str,
    # Fallback paths if artifacts is None:
    model_path: str | None = None,              # e.g. "ranker_lgbm.joblib"
    feature_cols: str | None = None,
    item_place_col: str = "PLAN",
    # Inference options:
    topn: int = 10,
    k_candidates: int = 300,
    seed: int = 2025,
    use_location_rerank: bool = True,
):
    """
    Run a full inference pass with a partially specified user profile.
    - If `artifacts` is provided (from training), it uses it directly.
    - Otherwise it loads model and feature_cols from disk.
    """
    # 1) Load model + feature_cols
    model = joblib_load(model_path)
            
    # 2) Load item matrix & PA similarity
    items_df = pd.read_csv(items_path)
    sim_df_pa = load_pa_similarity_matrix(pa_sim_path)

    # 3) Mock a partially specified user (some fields intentionally omitted)
    #    - Missing fields will be handled by neutral defaults inside feature builders.
    user_input = {
        "Budget_SGD": 4000000,                 # known & reliable
        "Preferred_Flat_Area_or_Room_Num": 100,              # wants 90 sqr
        "Preferred_Place": "JURONG EAST",      # explicit place
        "Priority_Distance_Proximity": 0.8,  # very distance-sensitive
        # The following are partially specified / masked on purpose:
        "Floor_Preference": 2,                # provided
        "NewHome_Preference": 0.5,            # neutral on new vs resale
        # priorities: leave some missing to test neutral handling
        "Priority_MRT_Access": 5,
        "Priority_Bus_Access": 5,
        "Priority_Amenities": 5,
        "Priority_School_Proximity": 5,
        "Priority_Park_Access": 5
    }

    # 4) Run recommendation (this calls: retrieval → features → predict → rerank → print)
    result = recommend_for_user(
        model=model,
        feature_cols=feature_cols,
        user_input=user_input,
        items_df=items_df,
        sim_df_pa=sim_df_pa,
        item_place_col=item_place_col,
        topn=topn,
        k_candidates=k_candidates,
        seed=seed,
        use_location_rerank=True,
    )
    return result

    imp_gain  = model.booster_.feature_importance(importance_type="gain")
    imp_split = model.booster_.feature_importance(importance_type="split")
    names     = model.booster_.feature_name()

    importance_df = pd.DataFrame({
    "feature": names,
    "gain": imp_gain,
    "split": imp_split
    }).sort_values("gain", ascending=False)


    return {"user_input": user_input, "recs": recs, "feature_cols": feature_cols, "model": model}
