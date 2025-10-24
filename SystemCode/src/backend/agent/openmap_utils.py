import os, sys, json, math, time, requests
from datetime import datetime
import pytz
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont  # pip install pillow

# =========================================================
# 地址查询与校验
# =========================================================
def get_latlng(place_name, token):
    url = "https://www.onemap.gov.sg/api/common/elastic/search"
    params = {"searchVal": place_name, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1}
    headers = {"Authorization": token}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"name": place_name, "latitude": None, "longitude": None, "error": str(e)}
    if data.get("found", 0) > 0:
        res = data["results"][0]
        return {"name": place_name, "latitude": res["LATITUDE"], "longitude": res["LONGITUDE"]}
    return {"name": place_name, "latitude": None, "longitude": None}

def is_in_sg(lat, lon):
    try:
        lat = float(lat); lon = float(lon)
        return 1.1 <= lat <= 1.5 and 103.6 <= lon <= 104.1
    except Exception:
        return False

# =========================================================
# 路线生成（OneMap Routing）
# =========================================================
def route_between(a_lat, a_lon, b_lat, b_lon, token):
    tz = pytz.timezone("Asia/Singapore")
    now = datetime.now(tz)
    date_str = now.strftime("%m-%d-%Y")
    time_str = now.strftime("%H:%M:%S")
    url = "https://www.onemap.gov.sg/api/public/routingsvc/route"
    params = {
        "start": f"{a_lat},{a_lon}",
        "end": f"{b_lat},{b_lon}",
        "routeType": "pt",
        "date": date_str,
        "time": time_str,
        "mode": "TRANSIT",
        "maxWalkDistance": 1000,
        "numItineraries": 1,
    }
    headers = {"Authorization": token}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json(), now

# =========================================================
# Polyline / 几何 / Zoom 工具
# =========================================================
def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    """Google Encoded Polyline → [(lon,lat), ...]"""
    coords = []
    index = lat = lon = 0
    while index < len(polyline_str):
        result = 0; shift = 0
        while True:
            b = ord(polyline_str[index]) - 63; index += 1
            result |= (b & 0x1f) << shift; shift += 5
            if b < 0x20: break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat
        result = 0; shift = 0
        while True:
            b = ord(polyline_str[index]) - 63; index += 1
            result |= (b & 0x1f) << shift; shift += 5
            if b < 0x20: break
        dlon = ~(result >> 1) if (result & 1) else (result >> 1)
        lon += dlon
        coords.append((lon / 1e5, lat / 1e5))
    return coords  # (lon,lat)

def even_subsample(latlon_list, k):
    n = len(latlon_list)
    if n <= k: return latlon_list
    idxs = [round(i*(n-1)/(k-1)) for i in range(k)]
    return [latlon_list[i] for i in idxs]

def _project(lat, lon, lat_c):
    return (lon * math.cos(math.radians(lat_c)), lat)  # 度坐标近似

def _perp_dist(p, a, b):
    (x, y), (x1, y1), (x2, y2) = p, a, b
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0: return math.hypot(x - x1, y - y1)
    t = max(0.0, min(1.0, ((x - x1)*dx + (y - y1)*dy)/(dx*dx + dy*dy)))
    xt, yt = x1 + t*dx, y1 + t*dy
    return math.hypot(x - xt, y - yt)

def rdp_simplify(latlon, eps_deg, lat_c):
    if len(latlon) <= 2: return latlon
    pts = [_project(lat, lon, lat_c) for (lat, lon) in latlon]
    keep = [0, len(pts)-1]; stack = [(0, len(pts)-1)]
    while stack:
        s, e = stack.pop()
        max_d, idx = 0.0, None
        for i in range(s+1, e):
            d = _perp_dist(pts[i], pts[s], pts[e])
            if d > max_d: max_d, idx = d, i
        if max_d > eps_deg and idx is not None:
            keep.append(idx); stack.append((s, idx)); stack.append((idx, e))
    keep = sorted(set(keep))
    return [latlon[i] for i in keep]

def auto_zoom_for_bbox(bbox, width_px, height_px, padding_px=24, min_zoom=10, max_zoom=18):
    min_lat, min_lon, max_lat, max_lon = bbox
    lat_c = (min_lat + max_lat) / 2.0
    meters_per_deg_lat = 111132.92
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat_c))
    width_m  = max(1e-6, (max_lon - min_lon) * meters_per_deg_lon)
    height_m = max(1e-6, (max_lat - min_lat) * meters_per_deg_lat)
    w_px = max(1, width_px - 2*padding_px)
    h_px = max(1, height_px - 2*padding_px)
    req_mpp = max(width_m / w_px, height_m / h_px)
    base = 156543.03392 * math.cos(math.radians(lat_c))
    z = math.floor(math.log2(base / req_mpp))
    return int(max(min_zoom, min(max_zoom, z)))

# =========================================================
# 样式（线与点）
# =========================================================
MODE_STYLE = {
    "WALK":   ((90,90,90),   4),
    "BUS":    ((200,0,0),    5),
    "SUBWAY": ((0,90,200),   5),
    "RAIL":   ((0,60,160),   5),
    "TRAM":   ((0,120,120),  4),
    "FERRY":  ((0,160,160),  4),
    "TRANSIT":((0,90,200),   5),
    "DRIVE":  ((20,20,20),   4),
    "CYCLE":  ((0,160,60),   4),
}
DEFAULT_COLOR = (177,0,0); DEFAULT_W = 4
MODE_LABEL_EN = {
    "WALK": "Walk", "BUS": "Bus", "SUBWAY": "Subway",
    "RAIL": "Rail", "TRAM": "Tram", "FERRY": "Ferry",
    "TRANSIT": "Transit", "DRIVE": "Drive", "CYCLE": "Cycle",
}
MODE_ORDER = ["WALK","BUS","SUBWAY","RAIL","TRAM","FERRY","DRIVE","CYCLE","TRANSIT"]

POINT_STYLE = {
    "START":   {"fill": (30,150,30),  "outline": (255,255,255), "r": 7},
    "END":     {"fill": (220,30,30),  "outline": (255,255,255), "r": 7},
    "TRANSFER":{"fill": (255,140,0),  "outline": (255,255,255), "r": 4},  # 更小
}

# =========================================================
# 静态图 URL 构造（稳健：抽稀+RDP+降级）
# =========================================================
def collect_bbox_and_modes(route_data):
    plan = route_data["plan"]; it = plan["itineraries"][0]
    s_lat, s_lon = float(plan["from"]["lat"]), float(plan["from"]["lon"])
    t_lat, t_lon = float(plan["to"]["lat"]), float(plan["to"]["lon"])
    all_lats, all_lons = [s_lat, t_lat], [s_lon, t_lon]
    modes_present = set()
    for leg in it.get("legs", []):
        modes_present.add((leg.get("mode") or "TRANSIT").upper())
        poly = leg.get("legGeometry", {}).get("points")
        if not poly: continue
        for (lon, lat) in decode_polyline(poly):
            all_lats.append(lat); all_lons.append(lon)
    bbox = (min(all_lats), min(all_lons), max(all_lats), max(all_lons))
    ordered_modes = [m for m in MODE_ORDER if m in modes_present]
    return bbox, ordered_modes

def collect_transfer_points(route_data):
    it = route_data["plan"]["itineraries"][0]
    transfers = []
    for i, leg in enumerate(it.get("legs", [])):
        if i == 0: continue
        st = leg.get("from")
        if st and "lat" in st and "lon" in st:
            transfers.append((float(st["lat"]), float(st["lon"])))
    return transfers

def build_lines_and_ptsmeta(route_data,
                            max_pts_per_leg,
                            rdp_tol_px,
                            width, height,
                            include_transfer_points,
                            transit_as_stops_only):
    plan = route_data["plan"]; it = plan["itineraries"][0]
    s_lat, s_lon = float(plan["from"]["lat"]), float(plan["from"]["lon"])
    t_lat, t_lon = float(plan["to"]["lat"]), float(plan["to"]["lon"])
    bbox, _ = collect_bbox_and_modes(route_data)
    lat_c = (bbox[0] + bbox[2]) / 2.0

    base = 156543.03392 * math.cos(math.radians(lat_c))
    approx_zoom = 13
    mpp = base / (2 ** approx_zoom)
    tol_m = max(0.1, rdp_tol_px * mpp)
    eps_deg = tol_m / 111132.92  # meter→deg

    line_pieces = []
    pts_meta = {
        "start": (s_lat, s_lon),
        "end": (t_lat, t_lon),
        "transfers": collect_transfer_points(route_data) if include_transfer_points else []
    }

    for leg in it.get("legs", []):
        mode = (leg.get("mode") or "TRANSIT").upper()
        color, w = MODE_STYLE.get(mode, (DEFAULT_COLOR, DEFAULT_W))

        if transit_as_stops_only and mode in {"BUS","SUBWAY","RAIL","TRAM"}:
            f, t = leg.get("from", {}), leg.get("to", {})
            if "lat" in f and "lon" in f and "lat" in t and "lon" in t:
                coords = [(float(f["lat"]), float(f["lon"])), (float(t["lat"]), float(t["lon"]))]
            else:
                coords = []
        else:
            poly = leg.get("legGeometry", {}).get("points")
            if not poly: coords = []
            else:
                coords = [(lat, lon) for (lon, lat) in decode_polyline(poly)]
                coords = even_subsample(coords, max_pts_per_leg)
                if len(coords) > 2 and rdp_tol_px > 0:
                    coords = rdp_simplify(coords, eps_deg, lat_c)

        if not coords: continue
        coord_str = "],[".join([f"{la:.6f},{lo:.6f}" for la, lo in coords])
        line_pieces.append(f"[[{coord_str}]]:{color[0]},{color[1]},{color[2]}:{w}")

    return line_pieces, pts_meta

def build_static_url(route_data, width, height, zoom,
                     max_pts_per_leg, rdp_tol_px,
                     include_transfer_points, transit_as_stops_only,
                     max_url_len):
    bbox, _ = collect_bbox_and_modes(route_data)
    min_lat, min_lon, max_lat, max_lon = bbox
    center_lat = (min_lat + max_lat)/2.0
    center_lon = (min_lon + max_lon)/2.0
    if zoom is None:
        zoom = auto_zoom_for_bbox(bbox, width, height, padding_px=24, min_zoom=10, max_zoom=18)

    line_pieces, pts_meta = build_lines_and_ptsmeta(
        route_data, max_pts_per_leg, rdp_tol_px, width, height,
        include_transfer_points, transit_as_stops_only
    )
    if not line_pieces:
        raise ValueError("No line pieces built")

    base = "https://www.onemap.gov.sg/api/staticmap/getStaticImage"
    # 不附带 points（我们本地绘制，URL 更短）
    url = (f"{base}?layerchosen=default"
           f"&latitude={center_lat:.6f}&longitude={center_lon:.6f}"
           f"&zoom={zoom}&width={width}&height={height}"
           f"&lines=" + "|".join(line_pieces))

    if len(url) > max_url_len:
        raise RuntimeError(f"URL_TOO_LONG:{len(url)}")
    meta = {"center_lat": center_lat, "center_lon": center_lon, "zoom": zoom,
            "width": width, "height": height, "pts_meta": pts_meta}
    return url, meta

def fetch_static_with_backoff(route_data, token, out_png, include_transfer_points=True):
    attempts = [
        # max_pts_per_leg, rdp_tol_px, width, height, transit_as_stops_only
        (40, 1.5, 512, 512, False),
        (25, 2.5, 512, 512, False),
        (20, 3.5, 480, 480, False),
        (15, 4.5, 400, 400, False),
        (12, 5.5, 360, 360, False),
        (10, 6.5, 320, 320, True),
    ]
    headers = {"Authorization": token, "User-Agent": "Mozilla/5.0 (OneMap staticmap robust)"}
    MAX_URL_LEN = 1600
    last_err = None
    for (pts, tol, w, h, stops_only) in attempts:
        try:
            url, meta = build_static_url(
                route_data, width=w, height=h, zoom=None,
                max_pts_per_leg=pts, rdp_tol_px=tol,
                include_transfer_points=include_transfer_points,
                transit_as_stops_only=stops_only,
                max_url_len=MAX_URL_LEN
            )
        except Exception as e:
            last_err = f"build_url: {e}"
            time.sleep(0.5); continue

        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code == 200 and r.headers.get("Content-Type","").startswith("image/"):
                with open(out_png, "wb") as f:
                    f.write(r.content)
                return {"ok": True, "url": url, "meta": meta,
                        "w": w, "h": h, "pts": pts, "tol_px": tol, "stops_only": stops_only}
            else:
                last_err = f"http {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = f"request: {e}"
        time.sleep(0.6)
    return {"ok": False, "error": last_err}

# =========================================================
# 叠加标注（Start 置顶、Transfer 更小）
# =========================================================
def _lonlat_to_global_px(lon, lat, z):
    n = 256 * (2 ** z)
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y

def _lonlat_to_image_px(lon, lat, center_lon, center_lat, z, width, height):
    cx, cy = _lonlat_to_global_px(center_lon, center_lat, z)
    x, y = _lonlat_to_global_px(lon, lat, z)
    ix = (x - cx) + width / 2.0
    iy = (y - cy) + height / 2.0
    return int(round(ix)), int(round(iy))

def overlay_markers_on_png(png_path, out_path, meta,
                           draw_start_end=True, draw_transfers=True):
    img = Image.open(png_path).convert("RGBA")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    z   = meta["zoom"]
    clat, clon = meta["center_lat"], meta["center_lon"]
    pts = meta["pts_meta"]

    # 先画换乘点（底层）
    if draw_transfers and pts.get("transfers"):
        style = POINT_STYLE["TRANSFER"]; r = style["r"]
        for (lat, lon) in pts["transfers"]:
            x, y = _lonlat_to_image_px(lon, lat, clon, clat, z, W, H)
            draw.ellipse([x-r-2, y-r-2, x+r+2, y+r+2], fill=style["outline"])
            draw.ellipse([x-r,   y-r,   x+r,   y+r  ], fill=style["fill"])

    # 再画 End
    if draw_start_end:
        lat, lon = pts["end"]; style = POINT_STYLE["END"]; r = style["r"]
        x, y = _lonlat_to_image_px(lon, lat, clon, clat, z, W, H)
        draw.ellipse([x-r-2, y-r-2, x+r+2, y+r+2], fill=style["outline"])
        draw.ellipse([x-r,   y-r,   x+r,   y+r  ], fill=style["fill"])

    # 最后画 Start（置顶）
    if draw_start_end:
        lat, lon = pts["start"]; style = POINT_STYLE["START"]; r = style["r"]
        x, y = _lonlat_to_image_px(lon, lat, clon, clat, z, W, H)
        draw.ellipse([x-r-2, y-r-2, x+r+2, y+r+2], fill=style["outline"])
        draw.ellipse([x-r,   y-r,   x+r,   y+r  ], fill=style["fill"])

    img.save(out_path)

# =========================================================
# Legend 内嵌绘图（matplotlib，直接 plot；可保存为预览图）
# =========================================================
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def _rgb255(c):  # (r,g,b) 0-255 -> 0-1
    return tuple(v/255.0 for v in c)

def plot_legend_inline(modes_present, summary=None, *, title="Legend", show=True, save_path=None):
    """
    直接在代码里用 matplotlib 画 Legend 面板（英文；每条一行）。
    - modes_present: ["WALK","BUS","SUBWAY", ...]
    - summary: 由 build_route_summary(route_data) 得到的 dict，允许为 None
    - show: 是否 plt.show()
    - save_path: 若提供则保存 PNG
    """
    # 行数估计：标题 + 模式若干 + 3个点示例 + 摘要最多4行 + 边距
    n_modes = len(modes_present)
    rows = 1 + n_modes + 3 + 4 + 1
    fig_h = max(3.5, rows * 0.35)
    fig, ax = plt.subplots(figsize=(4.2, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, rows); ax.axis("off")

    y = rows - 1  # 从上往下画
    # 标题
    ax.text(0.05, y, title, fontsize=14, weight="bold", va="center"); y -= 1.0

    # 线型图例
    for mode in modes_present:
        label = MODE_LABEL_EN.get(mode, mode.title())
        color, w = MODE_STYLE.get(mode, ((177,0,0), 4))
        ax.plot([0.05, 0.45], [y, y], lw=max(3, w), color=_rgb255(color))
        ax.text(0.50, y, label, fontsize=12, va="center")
        y -= 0.9

    # 点示例（Start / Transfer / End）
    for label, key in [("Start","START"), ("Transfer","TRANSFER"), ("End","END")]:
        sty = POINT_STYLE[key]
        # 外白圈
        ax.add_patch(Circle((0.15, y), radius=0.03, color=_rgb255(sty["outline"]), transform=ax.transAxes))
        # 内实心
        ax.add_patch(Circle((0.15, y), radius=0.024, color=_rgb255(sty["fill"]), transform=ax.transAxes))
        ax.text(0.22, y, label, fontsize=12, va="center", transform=ax.transAxes)
        y -= 0.9

    # 摘要（最多 4 行，每条一行）
    lines = []
    if summary and summary.get("available"):
        if summary.get("duration_min") is not None:
            lines.append(f"Duration: {summary['duration_min']} min")
        if summary.get("transfers") is not None:
            lines.append(f"Transfers: {summary['transfers']}")
        if summary.get("walkTime_min") is not None:
            t = f"Walking: {summary['walkTime_min']} min"
            if summary.get("walkDistance_m") is not None:
                t += f" ({int(summary['walkDistance_m'])} m)"
            lines.append(t)
        if summary.get("fare") is not None:
            lines.append(f"Fare: {summary['fare']}")

    for t in lines[:4]:
        ax.text(0.05, y, t, fontsize=10, va="center"); y -= 0.8

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

# =========================================================
# 右侧英文图例 + 摘要（绘到成图右侧）
# =========================================================
def _try_load_font(candidates, size):
    for p in candidates:
        try: return ImageFont.truetype(p, size)
        except Exception: continue
    return ImageFont.load_default()

def _wrap_text(draw, text, font, max_width):
    words = text.split(); lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_width:
            cur = test
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

def build_route_summary(route_json: dict) -> dict:
    plan = route_json.get("plan", {}); itins = plan.get("itineraries", [])
    if not itins: return {"available": False}
    it = itins[0]
    def sec2min(v):
        if v is None: return None
        return round(v/60) if v < 1e6 else round(v/60000)
    return {
        "available": True,
        "duration_min": sec2min(it.get("duration")),
        "walkTime_min": sec2min(it.get("walkTime")),
        "transitTime_min": sec2min(it.get("transitTime")),
        "waitingTime_min": sec2min(it.get("waitingTime")),
        "walkDistance_m": it.get("walkDistance"),
        "transfers": it.get("transfers"),
        "fare": it.get("fare"),
    }

def add_legend_right(png_path: str, out_path: str, modes_present: list,
                     *, title="Legend", show_summary=True, summary=None,
                     legend_width=240, padding=14, row_h=26,
                     bg=(255,255,255,255), panel_bg=(255,255,255,240), panel_border=(220,220,220,255)):
    img = Image.open(png_path).convert("RGBA")
    W, H = img.size

    fonts = ["NotoSansCJK-Regular.ttc","NotoSansSC-Regular.otf","Arial Unicode.ttf",
             "arialuni.ttf","SimHei.ttf","msyh.ttc","DejaVuSans.ttf","Arial.ttf"]
    font = _try_load_font(fonts, 14)
    font_bold = _try_load_font(fonts, 16)
    font_small = _try_load_font(fonts, 12)

    # 行数：已出现的线型 + Start/Transfer/End + 标题 + 摘要（最多 4 行）
    n_rows = len(modes_present) + 3
    extra = int(row_h * 4) if (show_summary and summary and summary.get("available")) else 0
    needed_h = padding + row_h + n_rows*row_h + extra + padding
    out_h = max(H, needed_h); out_w = W + legend_width

    canvas = Image.new("RGBA", (out_w, out_h), bg); canvas.paste(img, (0,0))
    draw = ImageDraw.Draw(canvas)

    # 面板
    x_panel = W
    draw.rectangle([x_panel, 0, out_w-1, out_h-1], fill=panel_bg, outline=panel_border)
    x0 = W + 12; y = padding

    # 标题
    draw.text((x0, y), title, fill=(0,0,0), font=font_bold); y += row_h

    # 线型
    for mode in modes_present:
        label = MODE_LABEL_EN.get(mode, mode.title())
        color, w = MODE_STYLE.get(mode, ((177,0,0), 4))
        x1, x2 = x0, x0 + 72; y_mid = y + row_h//2
        draw.line([(x1, y_mid), (x2, y_mid)], fill=color, width=max(3, w))
        draw.text((x2 + 10, y_mid - 8), label, fill=(0,0,0), font=font)
        y += row_h

    # 点图例
    for label, style_key in [("Start","START"), ("Transfer","TRANSFER"), ("End","END")]:
        style = POINT_STYLE[style_key]; r = style["r"]
        cx = x0 + 36; cy = y + row_h//2
        draw.ellipse([cx-r-2, cy-r-2, cx+r+2, cy+r+2], fill=style["outline"])
        draw.ellipse([cx-r,   cy-r,   cx+r,   cy+r  ], fill=style["fill"])
        draw.text((x0 + 72, cy - 8), label, fill=(0,0,0), font=font)
        y += row_h

    # 摘要（英文；每条单行，自动换行避免溢出）
    if show_summary and summary and summary.get("available"):
        y += 6; draw.line([(W+10, y), (out_w-10, y)], fill=(210,210,210,255), width=1); y += 10
        lines = []
        if summary.get("duration_min") is not None:
            lines.append(f"Duration: {summary['duration_min']} min")
        if summary.get("transfers") is not None:
            lines.append(f"Transfers: {summary['transfers']}")
        if summary.get("walkTime_min") is not None:
            t = f"Walking: {summary['walkTime_min']} min"
            if summary.get("walkDistance_m") is not None:
                t += f" ({int(summary['walkDistance_m'])} m)"
            lines.append(t)
        if summary.get("fare") is not None:
            lines.append(f"Fare: {summary['fare']}")
        max_text_w = legend_width - 24
        for t in lines[:4]:
            for sub in _wrap_text(draw, t, font_small, max_text_w):
                draw.text((x0, y), sub, fill=(0,0,0), font=font_small)
                y += row_h - 6

    canvas.save(out_path)


def simplify_onemap_route(api_response: dict):
    """
    Simplify OneMap / OpenTripPlanner API route response for LLM interpretation.
    """
    plan = api_response.get("plan", {})
    itineraries = plan.get("itineraries", [])
    if not itineraries:
        return {"summary": {}, "legs": []}

    # 只取第一个路线方案（通常 itineraries[0]）
    route = itineraries[0]

    # ---- summary ----
    summary = {
        "total_duration_min": round(route.get("duration", 0) / 60, 1),
        "total_walk_distance_m": round(route.get("walkDistance", 0), 1),
        "transfers": route.get("transfers", 0),
        "fare": float(route.get("fare", 0)) if route.get("fare") else None
    }

    # ---- legs ----
    legs_data = []
    for leg in route.get("legs", []):
        leg_info = {
            "mode": leg.get("mode"),
            "from": leg.get("from", {}).get("name"),
            "to": leg.get("to", {}).get("name"),
            "distance_m": round(leg.get("distance", 0), 1),
            "duration_s": leg.get("duration", 0)
        }

        # 补充交通信息
        if leg.get("transitLeg"):
            leg_info["route"] = leg.get("route")
            leg_info["agency"] = leg.get("agencyName")
            # 中途站名（只取名字）
            stops = leg.get("intermediateStops", [])
            if stops:
                leg_info["intermediate_stops"] = [s["name"] for s in stops if "name" in s]

        legs_data.append(leg_info)

    return {
        "summary": summary,
        "legs": legs_data
    }

