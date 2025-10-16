# =========================================================
# Farm Profit Mapping Tool V4 - WORKING LOGIC + COMPACT LAYOUT
# =========================================================
import os
import io
import zipfile
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
import streamlit as st
import folium
from streamlit_folium import st_folium
from branca.element import MacroElement, Template
from matplotlib import colors as mpl_colors
from shapely.geometry import Point

# ===========================
# ALPHASHAPE AUTO-INSTALLER (Bulletproof Cloud Version)
# ===========================
import importlib
import subprocess
import sys

def ensure_alphashape():
    """
    Ensures alphashape is available and functional.
    Safe for Streamlit Cloud / Cursor environments.
    Never crashes the app, always falls back gracefully.
    """
    try:
        alphashape_module = importlib.import_module("alphashape")
        return alphashape_module
    except Exception as e:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "alphashape==1.3.1", "--quiet", "--disable-pip-version-check"],
                check=True,
                timeout=60
            )
            alphashape_module = importlib.import_module("alphashape")
            st.success("✅ alphashape installed and imported successfully")
            return alphashape_module
        except Exception as e2:
            return None

alphashape = ensure_alphashape()
ALPHA_OK = alphashape is not None

# Alphashape status (silent unless needed)

# Clear caches
st.cache_data.clear()
st.cache_resource.clear()

# ===========================
# COORDINATE NORMALIZATION & BOUNDARY HELPERS
# ===========================
from shapely.geometry import MultiPoint

def normalize_coordinates(df):
    """Standardize coordinate columns and return df, lon_col, lat_col"""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    lat_col, lon_col = None, None
    for c in df.columns:
        if c.startswith("lat"):
            lat_col = c
        elif c.startswith("lon") or c.startswith("long"):
            lon_col = c
    if not lat_col and "latitude" in df.columns: 
        lat_col = "latitude"
    if not lon_col and "longitude" in df.columns: 
        lon_col = "longitude"
    return df, lon_col, lat_col

def build_field_boundary(df, lon_col, lat_col, alpha_val=0.0025):
    """Build a concave (alpha-shape) or convex boundary from GPS points."""
    if df is None or df.empty or lon_col not in df.columns or lat_col not in df.columns:
        return None
    
    # Filter valid coordinates
    valid_df = df[[lon_col, lat_col]].dropna()
    if len(valid_df) < 3:
        return None
    
    pts = list(zip(valid_df[lon_col], valid_df[lat_col]))
    
    if ALPHA_OK and alphashape is not None:
        try:
            hull = alphashape.alphashape(pts, alpha_val)
            return hull
        except Exception as e:
            st.warning(f"⚠️ Alpha-shape failed ({str(e)[:50]}) - using convex hull fallback")
    
    # Convex hull fallback
    return MultiPoint(pts).convex_hull

# ===========================
# COMPACT THEME + LAYOUT
# ===========================
def apply_compact_theme():
    st.set_page_config(page_title="Farm ROI Tool V4", layout="wide")
    st.title("Farm Profit Mapping Tool V4")

# ===========================
# HELPERS
# ===========================
import re

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def pick_col(df: pd.DataFrame, preferred: list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    norm_map = {_norm(c): c for c in df.columns}
    # exact / preferred order first
    for key in preferred:
        if key in norm_map:
            return norm_map[key]
    # then fuzzy contains "yield" / "yld"
    for k, orig in norm_map.items():
        if "yield" in k or re.search(r"\byld\b", k):
            return orig
    return None

YIELD_PREFS = [
    "yield", "yld_vol_dr", "yld_mass_dr", "yield_dry", "dry_yield",
    "yld_vol_wt", "yld_mass_wt", "wet_yield", "crop_flw_m"
]

LAT_PREFS = [
    "latitude", "lat", "point_y", "y", "ycoord", "y_coord", "northing", "north"
]

LON_PREFS = [
    "longitude", "lon", "long", "point_x", "x", "xcoord", "x_coord", "easting", "east"
]

def auto_height(df: pd.DataFrame, row_h: int = 36, header: int = 44, pad: int = 16) -> int:
    n = max(1, len(df))
    return int(header + n * row_h + pad)

def df_px_height(nrows: int, row_h: int = 28, header: int = 34, pad: int = 2) -> int:
    return int(header + max(1, nrows) * row_h + pad)

# ---- Data editor height helper (matches Streamlit's real row metrics) ----
def editor_height(num_rows: int) -> int:
    # Visual metrics measured from the current theme:
    HEADER = 42   # header row incl. border
    ROW    = 34   # body row height
    PAD    = 8    # tighter bottom padding (was 12)
    if num_rows < 1:
        num_rows = 1
    return int(HEADER + num_rows * ROW + PAD)

def find_col(df: pd.DataFrame, names) -> Optional[str]:
    if df is None or df.empty:
        return None
    norm = {c.lower().replace(" ", "_"): c for c in df.columns}
    for n in names:
        key = n.lower().replace(" ", "_")
        if key in norm:
            return norm[key]
    return None

def df_to_gdf(df: pd.DataFrame, lat_col: str = None, lon_col: str = None, crs: str = "EPSG:4326") -> Optional[gpd.GeoDataFrame]:
    """Convert DataFrame to GeoDataFrame with robust lat/lon detection."""
    if df is None or df.empty:
        return None

    if lat_col is None or lon_col is None:
        # Common candidates for lat/lon columns
        candidates = {"lat": ["lat", "latitude", "y"], "lon": ["lon", "longitude", "long", "x"]}
        found_lat, found_lon = None, None
        for c in df.columns:
            cl = c.lower()
            if any(k == cl or cl.startswith(k) for k in candidates["lat"]):
                found_lat = c
            if any(k == cl or cl.startswith(k) for k in candidates["lon"]):
                found_lon = c
        lat_col = found_lat
        lon_col = found_lon

    if lat_col is None or lon_col is None:
        st.warning("Latitude/Longitude columns not found — returning None.")
        return None

    try:
        gdf = gpd.GeoDataFrame(df.copy(), geometry=[Point(xy) for xy in zip(df[lon_col].astype(float), df[lat_col].astype(float))], crs=crs)
        return gdf
    except Exception as e:
        st.error(f"Failed to convert to GeoDataFrame: {e}")
        return None

def safe_read_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """Safely read CSV file with encoding fallback."""
    try:
        if uploaded_file is None:
            return None
        return pd.read_csv(uploaded_file)
    except Exception:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin1")
        except Exception:
            st.error("Failed to parse CSV. Ensure it's a valid comma-separated file.")
            return None

# =========================================================
# HARDENED load_vector_file (recursive + best shapefile select)
# =========================================================
def load_vector_file(uploaded_file):
    """
    Read GeoJSON/JSON/ZIP(SHP)/SHP into EPSG:4326 GeoDataFrame.
    Recursively searches ZIPs for nested .shp files and picks the best candidate.
    """
    try:
        name = uploaded_file.name.lower()

        if name.endswith((".geojson", ".json")):
            gdf = gpd.read_file(uploaded_file)

        elif name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zpath = os.path.join(tmpdir, "in.zip")
                with open(zpath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(tmpdir)

                shp_candidates = []
                for root, _, files in os.walk(tmpdir):
                    for fn in files:
                        if fn.lower().endswith(".shp"):
                            shp_candidates.append(os.path.join(root, fn))
                if not shp_candidates:
                    return None

                chosen = None
                best_score = -1.0
                for shp in shp_candidates:
                    try:
                        base = os.path.splitext(shp)[0]
                        if not (os.path.exists(base + ".dbf") or os.path.exists(base + ".DBF")):
                            continue
                        g = gpd.read_file(shp)
                        if g is None or g.empty:
                            continue
                        cols_lower = [c.lower() for c in g.columns]
                        has_yield = any(k in cols_lower for k in [
                            "yield","dry_yield","wet_yield","yld_mass_dr","yld_vol_dr",
                            "yld_mass_wt","yld_vol_wt","crop_flw_m","yield_dry","yield_wet"
                        ])
                        is_point = g.geom_type.astype(str).str.contains("Point",case=False).any()
                        score = (2 if has_yield else 0) + (1 if is_point else 0) + min(len(g),2000)/2000.0
                        if score > best_score:
                            best_score = score
                            chosen = g
                    except Exception:
                        continue
                gdf = chosen

        elif name.endswith(".shp"):
            with tempfile.TemporaryDirectory() as tmpdir:
                shp_path = os.path.join(tmpdir, os.path.basename(uploaded_file.name))
                with open(shp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                gdf = gpd.read_file(shp_path)
        else:
            return None

        if gdf is None or gdf.empty:
            return None

        try:
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)  # best-effort default if missing
            elif getattr(gdf.crs, "is_projected", False):
                gdf = gdf.to_crs(epsg=4326)
        except Exception:
            pass

        # geometry hardening - SKIP for Point geometries (buffer corrupts them)
        try:
            # Only apply buffer to Polygon/MultiPolygon geometries
            if not gdf.geometry.geom_type.isin(["Point", "MultiPoint"]).all():
                gdf["geometry"] = gdf.geometry.buffer(0)
        except Exception:
            pass
        
        # Only explode MultiPolygon/MultiPoint - NOT regular Points
        try:
            if gdf.geometry.geom_type.isin(["MultiPolygon", "MultiPoint", "MultiLineString"]).any():
                gdf = gdf.explode(index_parts=False, ignore_index=True)
        except TypeError:
            if gdf.geometry.geom_type.isin(["MultiPolygon", "MultiPoint", "MultiLineString"]).any():
                gdf = gdf.explode().reset_index(drop=True)

        return gdf

    except Exception as e:
        st.warning(f"Vector read failure for {uploaded_file.name}: {e}")
        return None

# =========================================================
# DEBUG REPORT for any loaded GeoDataFrame
# =========================================================
def debug_report_gdf(gdf: gpd.GeoDataFrame, label: str):
    """Sidebar/expander debug: CRS, geometry types, columns, sample rows."""
    if gdf is None or getattr(gdf, "empty", True):
        return
    try:
        with st.expander(f"Debug · {label}", expanded=False):
            st.write({
                "rows": len(gdf),
                "crs": str(gdf.crs),
                "geom_types": gdf.geom_type.value_counts().to_dict()
                if hasattr(gdf, "geom_type") else "N/A",
                "total_bounds": getattr(gdf, "total_bounds", None),
            })
            st.write("Columns:", list(gdf.columns))
            st.write(gdf.head(5))
    except Exception:
        pass

# =========================================================
# process_prescription and bootstrap (unchanged)
# =========================================================
def process_prescription(file, prescrip_type="fertilizer"):
    """
    Returns: (grouped_table, original_gdf_or_None)
    grouped_table columns: product, Acres, CostTotal, CostPerAcre
    """
    if file is None:
        return pd.DataFrame(columns=["product", "Acres", "CostTotal", "CostPerAcre"]), None
    try:
        name = file.name.lower()
        gdf_orig = None
        if name.endswith((".geojson", ".json", ".zip", ".shp")):
            gdf = load_vector_file(file)
            if gdf is None or gdf.empty:
                st.error(f"Could not read {prescrip_type} map.")
                return pd.DataFrame(columns=["product", "Acres", "CostTotal", "CostPerAcre"]), None
            gdf.columns = [c.strip().lower().replace(" ", "_") for c in gdf.columns]
            gdf_orig = gdf.copy()
            try:
                reps = gdf.geometry.representative_point()
                gdf["Longitude"] = reps.x
                gdf["Latitude"]  = reps.y
            except Exception:
                gdf["Longitude"], gdf["Latitude"] = np.nan, np.nan
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        else:
            df = pd.read_csv(file)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        if "product" not in df.columns:
            for c in ["variety", "hybrid", "type", "name", "material", "blend"]:
                if c in df.columns:
                    df.rename(columns={c: "product"}, inplace=True)
                    break
            else:
                df["product"] = prescrip_type.capitalize()
        if "acres" not in df.columns:
            df["acres"] = 0.0
        if "costtotal" not in df.columns:
            if {"price_per_unit", "units"}.issubset(df.columns):
                df["costtotal"] = pd.to_numeric(df["price_per_unit"], errors="coerce").fillna(0) * \
                                  pd.to_numeric(df["units"], errors="coerce").fillna(0)
            elif {"rate", "price"}.issubset(df.columns):
                df["costtotal"] = pd.to_numeric(df["rate"], errors="coerce").fillna(0) * \
                                  pd.to_numeric(df["price"], errors="coerce").fillna(0)
            else:
                df["costtotal"] = 0.0

        if df.empty:
            return pd.DataFrame(columns=["product", "Acres", "CostTotal", "CostPerAcre"]), gdf_orig

        grouped = df.groupby("product", as_index=False).agg(
            Acres=("acres", "sum"), CostTotal=("costtotal", "sum")
        )
        grouped["CostPerAcre"] = grouped.apply(
            lambda r: r["CostTotal"] / r["Acres"] if r["Acres"] > 0 else 0, axis=1
        )
        return grouped, gdf_orig
    except Exception as e:
        st.warning(f"Failed to read {file.name}: {e}")
        return pd.DataFrame(columns=["product", "Acres", "CostTotal", "CostPerAcre"]), None

def _bootstrap_defaults():
    """Ensure all keys exist so nothing crashes."""
    for k in ["chem", "ins", "insect", "fert", "seed", "rent", "mach", "labor", "col", "fuel", "int", "truck"]:
        st.session_state.setdefault(k, 0.0)
    st.session_state.setdefault("corn_yield", 200.0)
    st.session_state.setdefault("corn_price", 5.0)
    st.session_state.setdefault("bean_yield", 60.0)
    st.session_state.setdefault("bean_price", 12.0)
    st.session_state.setdefault("sell_price", st.session_state["corn_price"])
    st.session_state.setdefault("target_yield", 200.0)
    st.session_state.setdefault("fixed_products", pd.DataFrame(
        {"Type": ["Seed", "Fertilizer"], "Product": ["", ""], "Rate": [0.0, 0.0],
         "CostPerUnit": [0.0, 0.0], "$/ac": [0.0, 0.0]}
    ))
    st.session_state.setdefault("yield_df", pd.DataFrame())
    st.session_state.setdefault("fert_layers_store", {})
    st.session_state.setdefault("seed_layers_store", {})
    st.session_state.setdefault("fert_gdfs", {})
    st.session_state.setdefault("seed_gdf", None)
    st.session_state.setdefault("zones_gdf", None)
    st.session_state.setdefault("expenses_dict", {})
    st.session_state.setdefault("base_expenses_per_acre", 0.0)

# =========================================================
# UI: Uploaders row + summaries (FULLY HARDENED VERSION)
# =========================================================
def render_uploaders():
    st.subheader("Upload Maps")
    u1, u2, u3, u4 = st.columns(4)

    # ------------------------- ZONES -------------------------
    with u1:
        st.caption("Zone Map · GeoJSON/JSON/ZIP(SHP)")
        zone_file = st.file_uploader("Zone", type=["geojson", "json", "zip"],
                                     key="up_zone", accept_multiple_files=False)
        if zone_file:
            # Track zone file name for map refresh (only if changed)
            if zone_file.name != st.session_state.get("zone_file_name"):
                st.session_state["zone_file_name"] = zone_file.name
                st.session_state["map_refresh_trigger"] = st.session_state.get("map_refresh_trigger", 0) + 1
            
            zones_gdf = load_vector_file(zone_file)
            if zones_gdf is not None and not zones_gdf.empty:
                # Look for zone ID column first
                zone_col = next((c for c in ["Zone_ID", "zone_id", "ZONE_ID", "Zone", "zone", "ZONE", "Name", "name"]
                                 if c in zones_gdf.columns), None)
                if not zone_col:
                    zones_gdf["Zone"] = range(1, len(zones_gdf) + 1)
                    zone_col = "Zone"
                
                # Group by zone ID to combine features with same zone
                if zone_col in zones_gdf.columns:
                    zones_gdf = zones_gdf.dissolve(by=zone_col, aggfunc='first')
                    zones_gdf["Zone"] = zones_gdf.index
                else:
                    zones_gdf["Zone"] = zones_gdf[zone_col]

                g2 = zones_gdf.copy()
                if g2.crs is None:
                    g2.set_crs(epsg=4326, inplace=True)
                if g2.crs.is_geographic:
                    g2 = g2.to_crs(epsg=5070)
                zones_gdf["Calculated Acres"] = (g2.geometry.area * 0.000247105).astype(float)
                zones_gdf["Override Acres"] = zones_gdf["Calculated Acres"].astype(float)

                disp = zones_gdf[["Zone", "Calculated Acres", "Override Acres"]].copy()
                
                edited = st.data_editor(
                    disp,
                    num_rows="fixed", hide_index=True, 
                    height=editor_height(len(disp)),
                    use_container_width=True,
                    column_config={
                        "Zone": st.column_config.TextColumn(disabled=True),
                        "Calculated Acres": st.column_config.NumberColumn(format="%.2f", disabled=True),
                        "Override Acres": st.column_config.NumberColumn(format="%.2f"),
                    },
                    key="zones_editor"
                )
                
                st.markdown("""
<style>
/* Hide the vertical scrollbar track for this specific editor only */
div[data-testid="stDataFrame"][data-baseweb][aria-describedby*="zones_editor"] 
  [data-testid="stDataFrameScrollableContainer"]::-webkit-scrollbar { display:none; }
div[data-testid="stDataFrame"][data-baseweb][aria-describedby*="zones_editor"] 
  [data-testid="stDataFrameScrollableContainer"] { scrollbar-width: none; overflow: hidden; }
</style>
""", unsafe_allow_html=True)
                edited["Override Acres"] = pd.to_numeric(edited["Override Acres"], errors="coerce") \
                    .fillna(edited["Calculated Acres"])
                zones_gdf["Override Acres"] = edited["Override Acres"].astype(float).values
                st.caption(
                    f"Zones: {len(zones_gdf)} | Calc: {zones_gdf['Calculated Acres'].sum():,.2f} ac | "
                    f"Override: {zones_gdf['Override Acres'].sum():,.2f} ac"
                )
                st.session_state["zones_gdf"] = zones_gdf
            else:
                st.error("Could not read zone file.")
        else:
            st.caption("No zone file uploaded.")

    # ------------------------- YIELD -------------------------
    with u2:
        st.caption("Yield Map(s) · SHP/GeoJSON/ZIP(SHP)/CSV")
        yield_files = st.file_uploader(
            "Yield", type=["zip", "shp", "geojson", "json", "csv"],
            key="up_yield", accept_multiple_files=True
        )

        st.session_state["yield_df"] = pd.DataFrame()

        if yield_files:
            # Track yield file names for map refresh (only if changed)
            joined = "_".join([f.name for f in yield_files])
            if joined != st.session_state.get("yield_file_name"):
                st.session_state["yield_file_name"] = joined
                st.session_state["map_refresh_trigger"] = st.session_state.get("map_refresh_trigger", 0) + 1
            
            frames, messages = [], []

            YIELD_PREFS = [
                "yld_vol_dr", "yld_mass_dr", "dry_yield", "dry_yld", "yield",
                "harvestyield", "crop_yield", "yld_bu_ac", "prod_yield", "yld_bu_per_ac",
                "crop_flw_m", "yld_mass_w", "yld_vol_we"
            ]

            for yf in yield_files:
                try:
                    name = yf.name.lower()
                    df, gdf = None, None

                    # --- CSV ---
                    if name.endswith(".csv"):
                        df = safe_read_csv(yf)
                        if df is not None and not df.empty:
                            df.columns = [c.strip() for c in df.columns]
                            
                            # Check if geometry column exists (for CSV with POINT data)
                            if "geometry" in df.columns:
                                try:
                                    # Convert POINT strings to actual geometry
                                    from shapely import wkt
                                    df["geometry"] = df["geometry"].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
                                    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
                                    
                                    # Extract coordinates from geometry
                                    reps = gdf.geometry.representative_point()
                                    gdf["Longitude"] = reps.x
                                    gdf["Latitude"] = reps.y
                                    
                                    # Compact pill for coordinates extraction
                                    st.markdown(f"""
                                    <div style="display:inline-block; background:#1f3b53; color:#cfe7ff;
                                                padding:4px 8px; border-radius:6px; font-size:12.5px; margin:2px 0;">
                                        ✅ Extracted coordinates from geometry column in <b>{yf.name}</b>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.session_state["_yield_gdf_raw"] = gdf
                                    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
                                    
                                except Exception as e:
                                    st.warning(f"Could not parse geometry column: {e}")
                                    # Fallback to regular lat/lon detection
                                    gdf = df_to_gdf(df)
                                    if gdf is not None:
                                        st.session_state["_yield_gdf_raw"] = gdf
                                        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
                                    else:
                                        messages.append(f"{yf.name}: no valid lat/lon columns found — skipped.")
                                        continue
                            else:
                                # No geometry column, try regular lat/lon detection
                                gdf = df_to_gdf(df)
                                if gdf is not None:
                                    st.session_state["_yield_gdf_raw"] = gdf
                                    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
                                else:
                                    messages.append(f"{yf.name}: no valid lat/lon columns found — skipped.")
                                    continue
                        else:
                            messages.append(f"{yf.name}: could not read CSV — skipped.")
                            continue

                    # --- SHP/GEOJSON/ZIP(SHP) ---
                    else:
                        # ============================================================
                        # TRUE POINT GEOMETRY OVERRIDE — FINAL FIX
                        # ============================================================
                        gdf = load_vector_file(yf)
                        if gdf is None or gdf.empty:
                            messages.append(f"{yf.name}: could not load geometry — skipped.")
                            continue
                        
                        # Ensure WGS84 CRS
                        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                            try:
                                gdf = gdf.to_crs(epsg=4326)
                            except Exception as e:
                                st.error(f"CRS conversion failed for {yf.name}: {e}")

                        # ======================================================
                        # ABSOLUTE PRIORITY: Extract coordinates from geometry
                        # ======================================================
                        has_valid_coords = False
                        
                        # Check if we have Point geometries (even if mixed with other types)
                        if gdf.geometry.geom_type.isin(["Point"]).any():
                            # Filter to only Point geometries if mixed
                            if not gdf.geometry.geom_type.isin(["Point"]).all():
                                gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
                            
                            # Check for empty geometries
                            if gdf.geometry.is_empty.any():
                                gdf = gdf[~gdf.geometry.is_empty].copy()
                            
                            if len(gdf) > 0:
                                try:
                                    # Extract coordinates directly from Point geometry
                                    gdf["Longitude"] = gdf.geometry.x
                                    gdf["Latitude"] = gdf.geometry.y
                                    
                                    # Ensure CRS is set
                                    if gdf.crs is None:
                                        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
                                    
                                    has_valid_coords = True
                                    
                                except Exception as e:
                                    st.error(f"Failed to extract coordinates from Point geometry: {e}")
                        
                        # PRIORITY 2: Check for X/Y columns in attribute table
                        if not has_valid_coords and "X" in gdf.columns and "Y" in gdf.columns:
                            try:
                                gdf["Longitude"] = pd.to_numeric(gdf["X"], errors="coerce")
                                gdf["Latitude"] = pd.to_numeric(gdf["Y"], errors="coerce")
                                gdf = gdf.dropna(subset=["Latitude", "Longitude"])
                                
                                # Validate geographic ranges
                                if (gdf["Latitude"].min() >= -90 and gdf["Latitude"].max() <= 90 and
                                    gdf["Longitude"].min() >= -180 and gdf["Longitude"].max() <= 180):
                                    # Create geometry from coordinates
                                    gdf = gpd.GeoDataFrame(
                                        gdf, 
                                        geometry=gpd.points_from_xy(gdf["Longitude"], gdf["Latitude"]), 
                                        crs="EPSG:4326"
                                    )
                                    has_valid_coords = True
                                else:
                                    st.error(f"X/Y columns contain invalid coordinate ranges")
                            except Exception as e:
                                st.error(f"Failed to extract from X/Y columns: {e}")
                        
                        # PRIORITY 3: Check for other coordinate column names
                        if not has_valid_coords:
                            lat_col, lon_col = None, None
                            for c in gdf.columns:
                                c_lower = c.lower()
                                if lat_col is None and c_lower in ["y", "lat", "latitude", "northing"]:
                                    lat_col = c
                                if lon_col is None and c_lower in ["x", "lon", "long", "longitude", "easting"]:
                                    lon_col = c
                            
                            if lat_col and lon_col:
                                try:
                                    gdf["Longitude"] = pd.to_numeric(gdf[lon_col], errors="coerce")
                                    gdf["Latitude"] = pd.to_numeric(gdf[lat_col], errors="coerce")
                                    gdf = gdf.dropna(subset=["Latitude", "Longitude"])
                                    
                                    if (gdf["Latitude"].min() >= -90 and gdf["Latitude"].max() <= 90 and
                                        gdf["Longitude"].min() >= -180 and gdf["Longitude"].max() <= 180):
                                        gdf = gpd.GeoDataFrame(
                                            gdf, 
                                            geometry=gpd.points_from_xy(gdf["Longitude"], gdf["Latitude"]), 
                                            crs="EPSG:4326"
                                        )
                                        has_valid_coords = True
                                except Exception as e:
                                    st.error(f"Failed to extract from {lon_col}, {lat_col}: {e}")
                        
                        # Final check: Do we have valid coordinates?
                        if not has_valid_coords:
                            st.error(f"No valid coordinates found in {yf.name} - reconstruction may be inaccurate")

                        # All coordinate extraction complete above - now handle fallback reconstruction
                        try:
                            if not has_valid_coords:
                                    # Try multiple repair methods
                                    try:
                                        # Method 1: Buffer repair
                                        gdf.geometry = gdf.geometry.buffer(0)
                                        if not gdf.geometry.is_empty.all():
                                            st.info("✅ Repaired geometries using buffer method")
                                        else:
                                            # Method 2: Try to reconstruct from bounds if available
                                            if hasattr(gdf, 'bounds') and not gdf.bounds.isnull().all().all():
                                                st.info("Attempting to reconstruct geometries from bounds...")
                                                # This is a fallback - create point geometries from bounds center
                                                bounds = gdf.bounds
                                                centers_lon = (bounds['minx'] + bounds['maxx']) / 2
                                                centers_lat = (bounds['miny'] + bounds['maxy']) / 2
                                                from shapely.geometry import Point
                                                gdf.geometry = [Point(lon, lat) for lon, lat in zip(centers_lon, centers_lat)]
                                                st.info("✅ Created point geometries from bounds")
                                            else:
                                                # Method 3: Continue with previous logic
                                                st.info("Checking for other coordinate columns in attribute data...")
                                            coord_cols = []
                                            for col in gdf.columns:
                                                col_lower = col.lower()
                                                # More specific coordinate detection - exclude yield columns
                                                if (any(coord in col_lower for coord in ['latitude', 'longitude', 'lat', 'lon', 'x_coord', 'y_coord', 'easting', 'northing']) 
                                                    and not any(yield_word in col_lower for yield_word in ['yld', 'yield', 'mass', 'vol', 'flw'])):
                                                    coord_cols.append(col)
                                            
                                            if len(coord_cols) >= 2:
                                                st.info(f"Found coordinate columns: {coord_cols}")
                                                try:
                                                    # Use the first two coordinate columns found
                                                    lat_col = coord_cols[0] if 'lat' in coord_cols[0].lower() else coord_cols[1]
                                                    lon_col = coord_cols[1] if 'lon' in coord_cols[1].lower() else coord_cols[0]
                                                    
                                                    # Create point geometries from coordinate columns
                                                    from shapely.geometry import Point
                                                    gdf.geometry = [Point(lon, lat) for lon, lat in zip(gdf[lon_col], gdf[lat_col])]
                                                    st.info(f"✅ Created geometries from coordinate columns: {lat_col}, {lon_col}")
                                                except Exception as coord_error:
                                                    st.error(f"Failed to create geometries from coordinates: {coord_error}")
                                                    messages.append(f"{yf.name}: empty geometries — skipped.")
                                                    continue
                                            else:
                                                # Method 4: Try to convert GPS distance/angle to coordinates
                                                # ONLY if we don't already have valid coords from attribute table
                                                if has_valid_coords:
                                                    st.info("✅ Skipping coordinate reconstruction (true coordinates already loaded).")
                                                elif 'Distance_f' in gdf.columns and 'Track_deg_' in gdf.columns:
                                                    st.info("Found GPS distance/angle columns. Attempting coordinate conversion...")
                                                    try:
                                                        # Use field center from any available layer, with smart fallback
                                                        ref_lat = 38.8075  # Default fallback
                                                        ref_lon = -87.5390  # Default fallback
                                                        ref_source = "default"
                                                        
                                                        # Try zones first
                                                        zones_gdf = st.session_state.get("zones_gdf")
                                                        if zones_gdf is not None and not zones_gdf.empty:
                                                            zone_bounds = zones_gdf.total_bounds
                                                            ref_lat = (zone_bounds[1] + zone_bounds[3]) / 2
                                                            ref_lon = (zone_bounds[0] + zone_bounds[2]) / 2
                                                            ref_source = "zones"
                                                        else:
                                                            # Try other layers for reference
                                                            seed_gdf = st.session_state.get("seed_gdf")
                                                            if seed_gdf is not None and not seed_gdf.empty:
                                                                seed_bounds = seed_gdf.total_bounds
                                                                ref_lat = (seed_bounds[1] + seed_bounds[3]) / 2
                                                                ref_lon = (seed_bounds[0] + seed_bounds[2]) / 2
                                                                ref_source = "seed"
                                                            else:
                                                                # Try fertilizer layers
                                                                fert_gdfs = st.session_state.get("fert_gdfs", {})
                                                                for _k, fg in fert_gdfs.items():
                                                                    if fg is not None and not fg.empty:
                                                                        fert_bounds = fg.total_bounds
                                                                        ref_lat = (fert_bounds[1] + fert_bounds[3]) / 2
                                                                        ref_lon = (fert_bounds[0] + fert_bounds[2]) / 2
                                                                        ref_source = "fertilizer"
                                                                        break
                                                        
                                                        # Convert GPS distance/angle to lat/lon with proper field coverage
                                                        from shapely.geometry import Point
                                                        
                                                        # CRITICAL: Calibrate reconstruction to fill actual field extent
                                                        # Get field dimensions from zones_gdf for proper scaling
                                                        if zones_gdf is not None and not zones_gdf.empty:
                                                            xmin, ymin, xmax, ymax = zones_gdf.total_bounds
                                                            ref_lon = (xmin + xmax) / 2
                                                            ref_lat = (ymin + ymax) / 2
                                                            field_width_degrees = xmax - xmin
                                                            field_height_degrees = ymax - ymin
                                                            st.info(f"Using field extent: width={field_width_degrees:.6f}°, height={field_height_degrees:.6f}°")
                                                        else:
                                                            # Fallback to typical field size (~1 mile = 0.014 degrees)
                                                            field_width_degrees = 0.014
                                                            field_height_degrees = 0.014
                                                            st.warning("No zones available - using default field dimensions")
                                                        
                                                        # Extract distance and angle data
                                                        distances = pd.to_numeric(gdf["Distance_f"], errors="coerce").fillna(0).values
                                                        angles = gdf["Track_deg_"].astype(float).values
                                                        
                                                        # Normalize distance to 0-1 range across full dataset
                                                        dist_min, dist_max = distances.min(), distances.max()
                                                        if dist_max > dist_min:
                                                            norm_distances = (distances - dist_min) / (dist_max - dist_min)
                                                        else:
                                                            norm_distances = np.ones_like(distances) * 0.5
                                                        
                                                        # Check if we have pass number for better reconstruction
                                                        if 'Pass_Num' in gdf.columns:
                                                            pass_nums = pd.to_numeric(gdf["Pass_Num"], errors="coerce").fillna(0).values
                                                            max_pass = pass_nums.max()
                                                            if max_pass > 0:
                                                                # Use pass number for cross-track position
                                                                norm_pass = pass_nums / max_pass
                                                                # Reconstruct using distance along track and pass across track
                                                                dx = (norm_distances - 0.5) * field_width_degrees
                                                                dy = (norm_pass - 0.5) * field_height_degrees
                                                                st.info("✅ Using Pass_Num for improved spatial reconstruction")
                                                            else:
                                                                # Use angle-based reconstruction
                                                                angles_rad = np.radians(angles)
                                                                dx = (norm_distances - 0.5) * field_width_degrees * np.sin(angles_rad)
                                                                dy = (norm_distances - 0.5) * field_height_degrees * np.cos(angles_rad)
                                                        else:
                                                            # Use angle-based reconstruction with full field spread
                                                            angles_rad = np.radians(angles)
                                                            # Scale to fill field extent
                                                            dx = (norm_distances - 0.5) * field_width_degrees * np.sin(angles_rad)
                                                            dy = (norm_distances - 0.5) * field_height_degrees * np.cos(angles_rad)
                                                        
                                                        # Apply spatial calibration
                                                        lons = ref_lon + dx
                                                        lats = ref_lat + dy
                                                        
                                                        # Ensure reconstructed bounds match field extent (final calibration)
                                                        if zones_gdf is not None and not zones_gdf.empty:
                                                            # Compute current extent
                                                            lon_curr_min, lon_curr_max = lons.min(), lons.max()
                                                            lat_curr_min, lat_curr_max = lats.min(), lats.max()
                                                            
                                                            # Scale to match field extent precisely
                                                            lon_scale = field_width_degrees / (lon_curr_max - lon_curr_min + 1e-9)
                                                            lat_scale = field_height_degrees / (lat_curr_max - lat_curr_min + 1e-9)
                                                            
                                                            # Apply scaling around center
                                                            lons = ref_lon + (lons - ref_lon) * lon_scale * 0.95  # 95% to avoid edge overlap
                                                            lats = ref_lat + (lats - ref_lat) * lat_scale * 0.95
                                                        
                                                        gdf.geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
                                                        st.info(f"✅ Reconstructed yield coordinates calibrated to field extent: {len(gdf)} points")
                                                        
                                                    except Exception as gps_error:
                                                        st.error(f"GPS conversion failed: {gps_error}")
                                                        st.warning("Found GPS distance/angle columns but cannot convert to coordinates.")
                                                        st.error("❌ GPS distance/angle data requires conversion to lat/lon coordinates.")
                                                        st.info("Please provide a shapefile with actual latitude/longitude coordinates.")
                                                        messages.append(f"{yf.name}: GPS distance/angle data cannot be converted — skipped.")
                                                        continue
                                                else:
                                                    st.error(f"Cannot repair empty geometries in {yf.name}")
                                                    st.info(f"Available columns: {list(gdf.columns)}")
                                                    messages.append(f"{yf.name}: empty geometries — skipped.")
                                                    continue
                                    except Exception as repair_error:
                                        st.error(f"Geometry repair failed: {repair_error}")
                                        messages.append(f"{yf.name}: geometry repair failed — skipped.")
                                        continue
                            
                            # Extract coordinates using representative points ONLY if we don't have valid ones
                            if not has_valid_coords:
                                reps = gdf.geometry.representative_point()
                                gdf["Longitude"] = reps.x
                                gdf["Latitude"] = reps.y
                                st.info(f"✅ Extracted coordinates from geometry in {yf.name}")
                            
                        except Exception as e:
                            st.warning(f"Coordinate extraction failed for {yf.name}: {e}")
                            messages.append(f"{yf.name}: coordinate extraction failed — skipped.")
                            continue

                        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
                        # Save the gdf with coordinates to session state
                        st.session_state["_yield_gdf_raw"] = gdf

                    # --- Detect yield column (universal brand support) ---
                    # Support Ag Leader, John Deere, Case IH, and other formats
                    preferred_yield_cols = [
                        "Yld_Vol_Dr", "Dry_Yield", "Yield_Volume_Dry", "Yield_buac",
                        "DryYield_buac", "Yld_Mass_Dr", "Yield", "Yld_Vol_We", "Crop_Flw_M",
                        "Yld_Mass_D", "dry_yield", "DRY_YIELD", "YIELD", "Yield_Dry",
                        "Dry_Yield_Volume", "yld_vol_dr", "yld_mass_dr"
                    ]
                    
                    # First try exact match
                    yield_col = next((c for c in preferred_yield_cols if c in df.columns), None)
                    
                    # If no exact match, try case-insensitive
                    if not yield_col:
                        col_map = {c.lower(): c for c in df.columns}
                        for pref in preferred_yield_cols:
                            if pref.lower() in col_map:
                                yield_col = col_map[pref.lower()]
                                break
                    
                    # If still no match, try partial matching
                    if not yield_col:
                        for c in df.columns:
                            c_lower = c.lower()
                            if ("yld" in c_lower and ("dry" in c_lower or "vol" in c_lower or "mass" in c_lower)) or \
                               ("yield" in c_lower and "dry" in c_lower) or \
                               "crop_flw" in c_lower:
                                yield_col = c
                                break
                    
                    if not yield_col:
                        messages.append(f"{yf.name}: no recognized yield column — skipped.")
                        st.warning(f"⚠️ No valid yield column found in {yf.name}")
                        st.warning(f"Available columns: {list(df.columns)}")
                        st.warning(f"Looking for patterns: Yld_Vol_Dr, Dry_Yield, Yield_buac, etc.")
                        continue
                    
                    # Compact pill for chosen yield column
                    st.markdown(f"""
                    <div style="display:inline-block; background:#1f3b53; color:#cfe7ff;
                                padding:4px 8px; border-radius:6px; font-size:12.5px; margin:2px 0;">
                        ✅ Using yield column: <b>{yield_col}</b>
                    </div>
                    """, unsafe_allow_html=True)

                    # --- Fill coords if missing ---
                    if "Latitude" not in df.columns or "Longitude" not in df.columns:
                        if gdf is not None and not gdf.empty:
                            reps = gdf.geometry.representative_point()
                            df["Longitude"] = reps.x
                            df["Latitude"] = reps.y

                    # --- Clean numeric + filter ---
                    df["Yield"] = pd.to_numeric(df[yield_col], errors="coerce").fillna(0)
                    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
                    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

                    if df["Latitude"].notna().any() and df["Longitude"].notna().any():
                        df = df[
                            (df["Latitude"].between(-90, 90))
                            & (df["Longitude"].between(-180, 180))
                        ]

                    if len(df) > 10:
                        p5, p95 = np.nanpercentile(df["Yield"], [5, 95])
                        df = df[df["Yield"].between(p5, p95)]

                    if df.empty:
                        messages.append(f"{yf.name}: geometry extracted but no valid points.")
                        continue

                    frames.append(df[["Yield", "Latitude", "Longitude"]])
                    messages.append(f"{yf.name}: using '{yield_col}' with {len(df):,} valid yield points.")

                except Exception as e:
                    messages.append(f"{yf.name}: {e}")

            # =========================================================
            # ✅ Preserve geometry + ensure CRS = WGS84
            # =========================================================
            if frames:
                combo = pd.concat(frames, ignore_index=True)
                gdf_full = st.session_state.get("_yield_gdf_raw")

                if gdf_full is not None and not getattr(gdf_full, "empty", True):
                    # Use the gdf that already has coordinates (from synthetic generation)
                    st.session_state["yield_df"] = gdf_full.copy()
                else:
                    st.session_state["yield_df"] = combo.copy()

                # Compact pill for "loaded successfully"
                st.markdown(
                    f"""
                    <div style="
                        display:inline-block;
                        background:#1f4d33;
                        color:#d6ffdf;
                        padding:4px 8px;
                        border-radius:6px;
                        font-size:12.5px;
                        margin:2px 0;
                    ">
                        ✅ Yield loaded: <b>{len(frames)} file(s)</b>, {len(combo):,} points
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        display:inline-block;
                        background:#4d1f1f;
                        color:#ffd6d6;
                        padding:4px 8px;
                        border-radius:6px;
                        font-size:12.5px;
                        margin:2px 0;
                    ">
                        ❌ No valid yield data found
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.caption("No yield files uploaded.")
        
        # === Compact Sell Price Input (integrated into Yield section) ===
        # Tighten spacing above Sell Price label
        st.markdown("<div style='margin-top:-10px;'></div>", unsafe_allow_html=True)
        
        # --- Compact Sell Price Control ---
        st.markdown("""
<div style='display:flex; flex-direction:column; align-items:flex-start; margin-top:-4px;'>
    <label style='font-weight:600; font-size:14px; margin-bottom:2px;'>
        Crop Sell Price ($/bu)
    </label>
</div>
""", unsafe_allow_html=True)
        
        # Input directly follows label, no gap
        col_price, _ = st.columns([0.45, 0.55])
        with col_price:
            sell_price_val = st.number_input(
                label="Crop Sell Price ($/bu)",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key="sell_price_v4",          # new key to avoid sticky cache
                label_visibility="collapsed",
                format="%.2f"
            )
        
        st.session_state["sell_price"] = float(sell_price_val)
        
        # Subtle visual anchor below input to balance vertical rhythm
        st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

    # ------------------------- FERTILIZER -------------------------
    with u3:
        st.caption("Fertilizer RX · CSV/GeoJSON/JSON/ZIP(SHP)")
        fert_files = st.file_uploader("Fert", type=["csv", "geojson", "json", "zip"],
                                      key="up_fert", accept_multiple_files=True)
        st.session_state["fert_layers_store"] = {}
        st.session_state["fert_gdfs"] = {}

        if fert_files:
            # Track fert file names for map refresh (only if changed)
            joined = "_".join([f.name for f in fert_files])
            if joined != st.session_state.get("fert_file_names"):
                st.session_state["fert_file_names"] = joined
                st.session_state["map_refresh_trigger"] = st.session_state.get("map_refresh_trigger", 0) + 1
            
            summary = []
            for f in fert_files:
                try:
                    grouped, gdf_orig = process_prescription(f, "fertilizer")
                    if gdf_orig is not None and not gdf_orig.empty:
                        reps = gdf_orig.geometry.representative_point()
                        gdf_orig["Longitude"], gdf_orig["Latitude"] = reps.x, reps.y
                    if not grouped.empty:
                        key = os.path.splitext(f.name)[0].lower().replace(" ", "_")
                        st.session_state["fert_layers_store"][key] = grouped
                        st.session_state["fert_gdfs"][key] = gdf_orig
                        summary.append({"File": f.name, "Products": len(grouped)})
                except Exception as e:
                    st.warning(f"Fertilizer {f.name}: {e}")

            if summary:
                fert_df = pd.DataFrame(summary)
                
                st.dataframe(fert_df, 
                             hide_index=True,
                             height=editor_height(len(fert_df)),
                             use_container_width=True)
                
                st.markdown("""
<style>
/* Hide the vertical scrollbar track for this specific editor only */
div[data-testid="stDataFrame"]:has([aria-describedby*="fert"]) 
  [data-testid="stDataFrameScrollableContainer"]::-webkit-scrollbar { display:none; }
div[data-testid="stDataFrame"]:has([aria-describedby*="fert"]) 
  [data-testid="stDataFrameScrollableContainer"] { scrollbar-width: none; overflow: hidden; }
</style>
""", unsafe_allow_html=True)
            else:
                st.error("No valid fertilizer RX maps detected.")
        else:
            st.caption("No fertilizer files uploaded.")

    # ------------------------- SEED -------------------------
    with u4:
        st.caption("Seed RX · CSV/GeoJSON/JSON/ZIP(SHP)")
        seed_files = st.file_uploader("Seed", type=["csv", "geojson", "json", "zip"],
                                      key="up_seed", accept_multiple_files=True)
        st.session_state["seed_layers_store"] = {}
        st.session_state["seed_gdf"] = None

        if seed_files:
            # Track seed file names for map refresh (only if changed)
            joined = "_".join([f.name for f in seed_files])
            if joined != st.session_state.get("seed_file_name"):
                st.session_state["seed_file_name"] = joined
                st.session_state["map_refresh_trigger"] = st.session_state.get("map_refresh_trigger", 0) + 1
            
            summary = []
            last_gdf = None
            for f in seed_files:
                try:
                    grouped, gdf_orig = process_prescription(f, "seed")
                    if gdf_orig is not None and not gdf_orig.empty:
                        reps = gdf_orig.geometry.representative_point()
                        gdf_orig["Longitude"], gdf_orig["Latitude"] = reps.x, reps.y
                        last_gdf = gdf_orig
                    if not grouped.empty:
                        key = os.path.splitext(f.name)[0].lower().replace(" ", "_")
                        st.session_state["seed_layers_store"][key] = grouped
                        summary.append({"File": f.name, "Products": len(grouped)})
                except Exception as e:
                    st.warning(f"Seed {f.name}: {e}")

            if last_gdf is not None and not last_gdf.empty:
                st.session_state["seed_gdf"] = last_gdf

            if summary:
                seed_df = pd.DataFrame(summary)
                
                st.dataframe(seed_df,
                             hide_index=True,
                             height=editor_height(len(seed_df)),
                             use_container_width=True)
                
                st.markdown("""
<style>
/* Hide the vertical scrollbar track for this specific editor only */
div[data-testid="stDataFrame"]:has([aria-describedby*="seed"]) 
  [data-testid="stDataFrameScrollableContainer"]::-webkit-scrollbar { display:none; }
div[data-testid="stDataFrame"]:has([aria-describedby*="seed"]) 
  [data-testid="stDataFrameScrollableContainer"] { scrollbar-width: none; overflow: hidden; }
</style>
""", unsafe_allow_html=True)
            else:
                st.error("No valid seed RX maps detected.")
        else:
            st.caption("No seed files uploaded.")

# ==============================================================
# Scoped fix: remove Streamlit region padding under upload tables
# ==============================================================
st.markdown("""
<style>
/* Remove bottom padding + shadow for all top-row data editors */
div[data-testid="stDataFrame"][data-baseweb][aria-describedby*="zones_editor"],
div[data-testid="stDataFrame"][data-baseweb][aria-describedby*="fert_summary_editor"],
div[data-testid="stDataFrame"][data-baseweb][aria-describedby*="seed_summary_editor"] {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
    box-shadow: none !important;
}
div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stDataFrame"]) {
    margin-bottom: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# UI: Fixed inputs + Variable/Flat/CornSoy strip
# ===========================
def _mini_num(label: str, key: str, default: float = 0.0, step: float = 0.1):
    st.caption(label)
    current_value = st.session_state.get(key, default)
    return st.number_input(key, min_value=0.0, value=float(current_value),
                           step=step, label_visibility="collapsed")

def render_fixed_inputs_and_strip():
    st.markdown("### Fixed Inputs ($/ac)")

    r = st.columns(12, gap="small")
    with r[0]:  st.session_state["chem"]  = _mini_num("Chem ($/ac)",        "chem",  0.0, 1.0)
    with r[1]:  st.session_state["ins"]   = _mini_num("Insur ($/ac)",       "ins",   0.0, 1.0)
    with r[2]:  st.session_state["insect"]= _mini_num("Insect/Fung ($/ac)", "insect",0.0, 1.0)
    with r[3]:  st.session_state["fert"]  = _mini_num("Fert Flat ($/ac)",   "fert",  0.0, 1.0)
    with r[4]:  st.session_state["seed"]  = _mini_num("Seed Flat ($/ac)",   "seed",  0.0, 1.0)
    with r[5]:  st.session_state["rent"]  = _mini_num("Cash Rent ($/ac)",   "rent",  0.0, 1.0)
    with r[6]:  st.session_state["mach"]  = _mini_num("Mach ($/ac)",        "mach",  0.0, 1.0)
    with r[7]:  st.session_state["labor"] = _mini_num("Labor ($/ac)",       "labor", 0.0, 1.0)
    with r[8]:  st.session_state["col"]   = _mini_num("Living ($/ac)",      "col",   0.0, 1.0)
    with r[9]:  st.session_state["fuel"]  = _mini_num("Fuel ($/ac)",        "fuel",  0.0, 1.0)
    with r[10]: st.session_state["int"]   = _mini_num("Interest ($/ac)",    "int",   0.0, 1.0)
    with r[11]: st.session_state["truck"] = _mini_num("Truck Fuel ($/ac)",  "truck", 0.0, 1.0)

    expenses = {
        "Chemicals": st.session_state["chem"],
        "Insurance": st.session_state["ins"],
        "Insecticide/Fungicide": st.session_state["insect"],
        "Fertilizer (Flat)": st.session_state["fert"],
        "Seed (Flat)": st.session_state["seed"],
        "Cash Rent": st.session_state["rent"],
        "Machinery": st.session_state["mach"],
        "Labor": st.session_state["labor"],
        "Cost of Living": st.session_state["col"],
        "Extra Fuel": st.session_state["fuel"],
        "Extra Interest": st.session_state["int"],
        "Truck Fuel": st.session_state["truck"],
    }
    st.session_state["expenses_dict"] = expenses
    st.session_state["base_expenses_per_acre"] = sum(expenses.values())

# =========================================================
# SECTION: Variable Rate, Flat Rate (2 Columns, no scroll)
# =========================================================
def render_input_sections():
    # --- Scoped CSS: keep things tidy & full-width without inner scroll ---
    st.markdown("""
    <style>
      [data-testid="stDataEditorGrid"],
      [data-testid="stDataFrameContainer"],
      [data-testid="stDataFrameScrollableContainer"],
      [data-testid="stDataFrame"] {
          overflow: visible !important;
          height: auto !important;
          max-height: none !important;
          width: 100% !important;
      }
      [data-testid="stDataEditorGrid"] table,
      [data-testid="stDataFrame"] table {
          width: 100% !important;
          min-width: 100% !important;
          table-layout: fixed !important;
      }
    </style>
    """, unsafe_allow_html=True)

    def _profit_color(v):
        if isinstance(v, (int, float)):
            if v > 0:
                return "color:limegreen;font-weight:bold;"
            if v < 0:
                return "color:#ff4d4d;font-weight:bold;"
        return ""

    # ===============================
    # PREP: Gather all detected inputs
    # ===============================
    fert_store = st.session_state.get("fert_layers_store", {})
    seed_store = st.session_state.get("seed_layers_store", {})

    fert_products, seed_products = [], []
    for _, df in fert_store.items():
        if isinstance(df, pd.DataFrame) and "product" in df.columns:
            fert_products.extend(list(df["product"].dropna().unique()))
    for _, df in seed_store.items():
        if isinstance(df, pd.DataFrame) and "product" in df.columns:
            seed_products.extend(list(df["product"].dropna().unique()))

    fert_products = sorted(set(fert_products))
    seed_products = sorted(set(seed_products))

    # Auto-calculate Units Applied from uploaded RX maps (area-weighted)
    fert_units_map = {}
    for key, gdf in st.session_state.get("fert_gdfs", {}).items():
        if gdf is not None and not gdf.empty:
            try:
                # Robust fertilizer normalization (ft² / m² auto-detect)
                gdf_copy = gdf.copy()
                
                try:
                    if gdf_copy.crs is None:
                        gdf_copy = gdf_copy.set_crs("EPSG:4326")
                    if gdf_copy.crs.is_geographic:
                        gdf_copy = gdf_copy.to_crs(epsg=5070)
                except Exception:
                    pass
                
                # Determine true area factor
                gdf_copy["raw_area"] = gdf_copy.geometry.area
                mean_area = gdf_copy["raw_area"].mean()
                
                # Detect if polygons are likely in ft² or m²
                if mean_area > 43560:      # ~1 acre in ft²
                    area_factor = 2.2957e-5   # ft² → acres
                else:
                    area_factor = 0.000247105 # m² → acres
                
                gdf_copy["area_acres"] = gdf_copy["raw_area"] * area_factor
                
                # Calculate total units applied
                rate_col = next((c for c in gdf_copy.columns if "rate" in c.lower() or "tgt" in c.lower()), None)
                if rate_col:
                    gdf_copy["rate_numeric"] = pd.to_numeric(gdf_copy[rate_col], errors="coerce").fillna(0)
                    total_units = (gdf_copy["rate_numeric"] * gdf_copy["area_acres"]).sum()
                    fert_units_map[key] = round(total_units, 2)
                else:
                    fert_units_map[key] = 0.0
                        
            except Exception:
                fert_units_map[key] = 0.0
    
    seed_units_total = 0.0
    seed_gdf = st.session_state.get("seed_gdf")
    if seed_gdf is not None and not seed_gdf.empty:
        try:
            # Robust seed normalization (ft² / m² auto-detect)
            gdf_copy = seed_gdf.copy()
            
            try:
                if gdf_copy.crs is None:
                    gdf_copy = gdf_copy.set_crs("EPSG:4326")
                if gdf_copy.crs.is_geographic:
                    gdf_copy = gdf_copy.to_crs(epsg=5070)
            except Exception:
                pass
            
            # Determine true area factor
            gdf_copy["raw_area"] = gdf_copy.geometry.area
            mean_area = gdf_copy["raw_area"].mean()
            
            # Detect if polygons are likely in ft² or m²
            if mean_area > 43560:      # ~1 acre in ft²
                area_factor = 2.2957e-5   # ft² → acres
            else:
                area_factor = 0.000247105 # m² → acres
            
            gdf_copy["area_acres"] = gdf_copy["raw_area"] * area_factor
            
            # Calculate total units applied
            rate_col = next((c for c in gdf_copy.columns if "rate" in c.lower() or "tgt" in c.lower()), None)
            if rate_col:
                gdf_copy["rate_numeric"] = pd.to_numeric(gdf_copy[rate_col], errors="coerce").fillna(0)
                total_seeds = (gdf_copy["rate_numeric"] * gdf_copy["area_acres"]).sum()
                
                # Convert total seeds → seed units (bags)
                seeds_per_unit = st.session_state.get("seeds_per_unit", 80000)  # default = corn
                total_units = round(total_seeds / seeds_per_unit, 2)
                seed_units_total = total_units
            else:
                seed_units_total = 0.0
                    
        except Exception:
            seed_units_total = 0.0
    
    # Build editor rows from detected products with auto-filled units
    # Link fertilizer products to RX maps by matching names (fuzzy matching)
    import re
    
    all_variable_inputs = []
    for p in fert_products:
        units = 0.0
        # Remove all non-alphanumeric characters for matching
        p_clean = re.sub(r'[^a-zA-Z0-9]', '', p).lower()
        
        for key, total in fert_units_map.items():
            key_clean = re.sub(r'[^a-zA-Z0-9]', '', key).lower()
            
            # Flexible matching with special cases for common names
            if (
                p_clean in key_clean
                or key_clean in p_clean
                or (("uan" in p_clean or "n" in p_clean) and "n" in key_clean and len(key_clean) <= 20)
                or (("map" in p_clean or "phosphorus" in p_clean) and "map" in key_clean)
                or (("pot" in p_clean or "potash" in p_clean or "k" in p_clean) and "pot" in key_clean)
            ):
                units = total
                break
        
        all_variable_inputs.append({"Type": "Fertilizer", "Product": p, "Units Applied": round(units, 2), "Price per Unit ($)": 0.0})
    
    for p in seed_products:
        all_variable_inputs.append({"Type": "Seed", "Product": p, "Units Applied": seed_units_total, "Price per Unit ($)": 0.0})
    
    if not all_variable_inputs:
        all_variable_inputs = [{
            "Type": "Fertilizer", "Product": "", "Units Applied": 0.0, "Price per Unit ($)": 0.0
        }]

    cols = st.columns(2, gap="small")

    # -------------------------------------------------
    # 1) VARIABLE RATE INPUTS
    # -------------------------------------------------
    with cols[0]:
        st.markdown("### Variable Rate Inputs")
        with st.expander("Open Variable Rate Inputs", expanded=False):
            st.caption("Enter price per unit and total units applied from RX maps or manually.")
            rx_df = pd.DataFrame(all_variable_inputs)

            # Fixed rows = no checkbox column
            edited = st.data_editor(
                rx_df,
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                key="var_inputs_editor_final",
                column_config={
                    "Type": st.column_config.TextColumn(disabled=True),
                    "Product": st.column_config.TextColumn(),
                    "Units Applied": st.column_config.NumberColumn(format="%.4f"),
                    "Price per Unit ($)": st.column_config.NumberColumn(format="%.2f"),
                },
                height=auto_height(rx_df)
            ).fillna(0.0)

            # Calculate total cost
            edited["Total Cost ($)"] = edited["Units Applied"] * edited["Price per Unit ($)"]

            base_acres = float(st.session_state.get("base_acres", 1.0))
            st.session_state["variable_rate_inputs"] = edited
            st.session_state["variable_rate_cost_per_acre"] = (
                float(edited["Total Cost ($)"].sum()) / max(base_acres, 1.0)
            )
            
            # Add Seeds per Unit control
            st.markdown("---")
            st.markdown("**Seed Unit Settings**")
            st.caption("Adjust the number of seeds per unit for conversion (default 80,000 for corn, 140,000 for soybeans).")
            new_val = st.number_input(
                "Seeds per Unit",
                min_value=10000,
                max_value=200000,
                value=st.session_state.get("seeds_per_unit", 80000),
                step=1000,
                label_visibility="collapsed",
                key="seeds_per_unit_input"
            )
            st.session_state["seeds_per_unit"] = new_val

    # -------------------------------------------------
    # 2) FLAT RATE INPUTS
    # -------------------------------------------------
    with cols[1]:
        st.markdown("### Flat Rate Inputs")
        with st.expander("Open Flat Rate Inputs", expanded=False):
            st.caption("Uniform cost per acre for the entire field.")
            flat_products = sorted(set(fert_products + seed_products))
            flat_df = pd.DataFrame([{
                "Product": p, "Rate (units/ac)": 0.0, "Price per Unit ($)": 0.0
            } for p in flat_products]) if flat_products else pd.DataFrame(
                [{"Product": "", "Rate (units/ac)": 0.0, "Price per Unit ($)": 0.0}]
            )

            edited_flat = st.data_editor(
                flat_df,
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                key="flat_inputs_editor_final",
                column_config={
                    "Product": st.column_config.TextColumn(),
                    "Rate (units/ac)": st.column_config.NumberColumn(format="%.4f"),
                    "Price per Unit ($)": st.column_config.NumberColumn(format="%.2f"),
                },
                height=auto_height(flat_df)
            ).fillna(0.0)

            edited_flat["Cost per Acre ($/ac)"] = (
                edited_flat["Rate (units/ac)"] * edited_flat["Price per Unit ($)"]
            )

            # Persist in state for Section 9 math
            out_flat = edited_flat.copy()
            out_flat["$/ac"] = out_flat["Cost per Acre ($/ac)"]
            st.session_state["flat_products"] = edited_flat
            st.session_state["fixed_products"] = out_flat
            st.session_state["flat_rate_cost_per_acre"] = float(
                edited_flat["Cost per Acre ($/ac)"].sum()
            )

# ===========================
# Map helpers / overlays
# ===========================
def make_base_map():
    """Absolute fallback-proof map — never fails, even with no internet."""
    try:
        # Start with built-in CartoDB layer (reliable)
        m = folium.Map(
            location=[39.5, -98.35],
            zoom_start=5,
            tiles="CartoDB positron",
            attr="CartoDB",
            prefer_canvas=True,
            control_scale=False,
            zoom_control=False,
        )

        # Try to add Esri imagery (optional, best quality)
        try:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery",
                overlay=False,
                control=False,
                max_zoom=19,
                no_wrap=True
            ).add_to(m)
        except Exception:
            pass  # skip Esri if network fails

        # Add boundary labels if possible
        try:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri Boundaries",
                overlay=True,
                control=False,
                opacity=0.9,
                max_zoom=19,
                no_wrap=True
            ).add_to(m)
        except Exception:
            pass

        # Hide corner text, scale bar, and controls for compactness
        template = Template("""
        {% macro script(this, kwargs) %}
        var map = {{this._parent.get_name()}};
        map.scrollWheelZoom.disable();
        map.on('click', () => map.scrollWheelZoom.enable());
        map.on('mouseout', () => map.scrollWheelZoom.disable());
        setTimeout(() => {
          document.querySelectorAll('.leaflet-control-attribution, .leaflet-control-scale')
            .forEach(el => el.style.display='none');
        }, 500);
        {% endmacro %}
        """)
        macro = MacroElement()
        macro._template = template
        m.get_root().add_child(macro)

        return m

    except Exception as e:
        # Absolute emergency fallback (blank leaflet canvas)
        st.error(f"Map fallback triggered: {e}")
        return folium.Map(location=[39.5, -98.35], zoom_start=5, tiles="CartoDB positron", attr="CartoDB")

def clean_legend_title(name: str) -> str:
    """
    Remove file references, underscores, and target rate text from legend titles.
    """
    name = name.replace("_", " ").replace(".shp", "").replace(".geojson", "")
    name = name.replace("foss russ", "").replace("map poly", "")
    name = name.replace("Target Rate", "").strip(" —")
    return name.strip()

def add_gradient_legend(m, name, vmin, vmax, cmap, priority=99):
    """
    Priority-based legend renderer: legends stack in priority order
    Profit (priority 1) > Yield (priority 2) > Others (priority 99)
    """
    if vmin is None or vmax is None:
        print(f"DEBUG: add_gradient_legend() skipped for '{name}' (vmin={vmin}, vmax={vmax})")
        return

    # Skip hover layers from legend display
    if "(Hover)" in name:
        return

    # Initialize legend priority tracking
    if "_legend_priorities" not in st.session_state:
        st.session_state["_legend_priorities"] = []
    
    # Set priority based on layer name
    if "profit" in name.lower():
        priority = 1
    elif "yield" in name.lower():
        priority = 2
    elif "zone" in name.lower():
        priority = 3
    else:
        priority = 99
    
    # Clean the legend title
    display_name = clean_legend_title(name)
    
    # Track this legend's priority
    st.session_state["_legend_priorities"].append({"name": name, "priority": priority, "vmin": vmin, "vmax": vmax, "cmap": cmap})
    
    # Sort legends by priority and calculate position (filter out hover layers)
    sorted_legends = sorted(st.session_state["_legend_priorities"], key=lambda x: x["priority"])
    visible_legends = [leg for leg in sorted_legends if "(Hover)" not in leg["name"]]
    seq = next(i for i, legend in enumerate(visible_legends) if legend["name"] == name)
    
    # Calculate top offset based on priority order with proper spacing
    offset = 20 + seq * 110  # increased spacing between legends
    
    print(f"DEBUG: add_gradient_legend() called for '{display_name}' at priority {priority}, sequence {seq}, offset will be {offset}px")

    # Build gradient stops
    stops = [f"{mpl_colors.rgb2hex(cmap(i/100.0)[:3])} {i}%"
             for i in range(0, 101, 10)]
    gradient_css = ", ".join(stops)

    # Legend content without positioning (will be positioned by add_legend_html)
    legend_content = f"""
    <div style="
        font-family:sans-serif; font-size:12px; color:#b0b3b8;
        text-shadow:0 0 3px rgba(0,0,0,0.5); font-weight:500;
        background: rgba(255,255,255,0.0); padding:6px 10px; border-radius:6px;
        box-shadow:none; border:none; width:220px;">
      <div style="font-weight:600; margin-bottom:4px;">{display_name}</div>
      <div style="height:14px; border-radius:2px; margin-bottom:4px;
                  background:linear-gradient(90deg, {gradient_css});"></div>
      <div style="display:flex; justify-content:space-between;">
        <span>{vmin:.1f}</span><span>{vmax:.1f}</span>
      </div>
    </div>
    """
    add_legend_html(m, legend_content)

def add_hover_points(m, layer_name, grid_df, value_col):
    """Add invisible hover points for profit/yield layers"""
    if grid_df is None or grid_df.empty:
        return
    
    # Check if required columns exist
    if "Latitude" not in grid_df.columns or "Longitude" not in grid_df.columns:
        return
    if value_col not in grid_df.columns:
        return

    hover_features = []
    sampled_df = grid_df.sample(min(len(grid_df), 300))

    for _, row in sampled_df.iterrows():
        try:
            hover_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["Longitude"]), float(row["Latitude"])]
                },
                "properties": {
                    f"{layer_name}": float(row[value_col])
                }
            })
        except Exception:
            continue

    if hover_features:
        hover_geojson = folium.GeoJson(
            {"type": "FeatureCollection", "features": hover_features},
            name=f"{layer_name} (Hover)",
            show=False,
            style_function=lambda x: {"opacity": 0, "fillOpacity": 0},
        )
        hover_geojson.add_to(m)

def detect_rate_type(gdf):
    try:
        rate_col = None
        for c in gdf.columns:
            if "tgt" in c.lower() or "rate" in c.lower():
                rate_col = c
                break
        if rate_col and gdf[rate_col].nunique(dropna=True) == 1:
            return "Fixed Rate"
    except Exception:
        pass
    return "Variable Rate"

def infer_unit(gdf, rate_col, product_col):
    for cand in ["unit", "units", "uom", "rate_uom", "rateunit", "rate_unit"]:
        if cand in gdf.columns:
            vals = gdf[cand].dropna().astype(str).str.strip()
            if not vals.empty and vals.iloc[0] != "":
                return vals.iloc[0]
    rc = str(rate_col or "").lower()
    if any(k in rc for k in ["gpa", "gal", "uan"]): return "gal/ac"
    if any(k in rc for k in ["lb", "lbs", "dry", "nh3", "ammonia"]): return "lb/ac"
    if "kg" in rc: return "kg/ha"
    if any(k in rc for k in ["seed", "pop", "plant", "ksds", "kseed", "kseeds"]):
        try:
            med = pd.to_numeric(gdf[rate_col], errors="coerce").median()
            if 10 <= float(med) <= 90:
                return "k seeds/ac"
        except Exception:
            pass
        return "seeds/ac"
    prod_val = ""
    if product_col and product_col in gdf.columns:
        try:
            prod_val = str(gdf[product_col].dropna().astype(str).iloc[0]).lower()
        except Exception:
            prod_val = ""
    if "uan" in prod_val or "10-34-0" in prod_val: return "gal/ac"
    return None

def add_prescription_overlay(m, gdf, name, cmap, index):
    if gdf is None or gdf.empty:
        return
    gdf = gdf.copy()

    product_col, rate_col = None, None
    for c in gdf.columns:
        cl = str(c).lower()
        if product_col is None and "product" in cl:
            product_col = c
        if rate_col is None and ("tgt" in cl or "rate" in cl):
            rate_col = c

    gdf["RateType"] = detect_rate_type(gdf)

    if rate_col and pd.to_numeric(gdf[rate_col], errors="coerce").notna().any():
        vals = pd.to_numeric(gdf[rate_col], errors="coerce").dropna()
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmin == vmax:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    unit = infer_unit(gdf, rate_col, product_col)
    rate_alias = f"Target Rate ({unit})" if unit else "Target Rate"
    legend_name = f"{name} — {rate_alias}"

    def style_fn(feat):
        val = feat["properties"].get(rate_col) if rate_col else None
        if val is None or pd.isna(val):
            fill = "#808080"
        else:
            try:
                norm = (float(val) - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                norm = max(0.0, min(1.0, norm))
                fill = mpl_colors.rgb2hex(cmap(norm)[:3])
            except Exception:
                fill = "#808080"
        return {"stroke": False, "opacity": 0, "weight": 0,
                "fillColor": fill, "fillOpacity": 0.55}

    fields, aliases = [], []
    if product_col: fields.append(product_col); aliases.append("Product")
    if rate_col:    fields.append(rate_col);    aliases.append(rate_alias)
    fields.append("RateType"); aliases.append("Type")

    # --- Defensive layer naming ---
    if not name or str(name).strip().lower() in ["none", "null", "nan"]:
        name = "Unnamed Layer"

    folium.GeoJson(
        gdf, name=name, style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases),
        show=False  # Prescription layers OFF by default
    ).add_to(m)

    add_gradient_legend(m, legend_name, vmin, vmax, cmap, index)

def compute_bounds_for_heatmaps():
    """Compute flexible bounds that work with any combination of layers."""
    try:
        bnds = []
        layer_info = []
        
        # Collect bounds from all available layers
        zones_gdf = st.session_state.get("zones_gdf")
        if zones_gdf is not None and not getattr(zones_gdf, "empty", True):
            tb = zones_gdf.total_bounds
            if tb is not None and len(tb) == 4 and not any(pd.isna(tb)):
                bnds.append([[tb[1], tb[0]], [tb[3], tb[2]]])
                layer_info.append("zones")
        
        # Check seed layer
        seed_gdf = st.session_state.get("seed_gdf")
        if seed_gdf is not None and not getattr(seed_gdf, "empty", True):
            tb = seed_gdf.total_bounds
            if tb is not None and len(tb) == 4 and not any(pd.isna(tb)):
                bnds.append([[tb[1], tb[0]], [tb[3], tb[2]]])
                layer_info.append("seed")
                
        # Check fertilizer layers
        for _k, fg in st.session_state.get("fert_gdfs", {}).items():
            if fg is not None and not fg.empty:
                tb = fg.total_bounds
                if tb is not None and len(tb) == 4 and not any(pd.isna(tb)):
                    bnds.append([[tb[1], tb[0]], [tb[3], tb[2]]])
                    layer_info.append("fertilizer")
                    
        # Check yield layer
        ydf = st.session_state.get("yield_df")
        if ydf is not None and not ydf.empty:
            latc = find_col(ydf, ["latitude"]) or "Latitude"
            lonc = find_col(ydf, ["longitude"]) or "Longitude"
            if latc in ydf.columns and lonc in ydf.columns:
                bnds.append([[ydf[latc].min(), ydf[lonc].min()],
                             [ydf[latc].max(), ydf[lonc].max()]])
                layer_info.append("yield")
        
        if bnds:
            # Strategy: Use the largest bounding box as primary, but consider all layers
            south = min(b[0][0] for b in bnds)
            west = min(b[0][1] for b in bnds)
            north = max(b[1][0] for b in bnds)
            east = max(b[1][1] for b in bnds)
            
            # If zones are present, they get priority for field boundary definition
            if "zones" in layer_info and len(layer_info) > 1:
                # Use zones as primary but ensure other layers are visible
                zone_tb = zones_gdf.total_bounds
                if zone_tb is not None and len(zone_tb) == 4:
                    zone_south = zone_tb[1]
                    zone_west = zone_tb[0] 
                    zone_north = zone_tb[3]
                    zone_east = zone_tb[2]
                    
                    # Use zone bounds but expand to accommodate other layers
                    lat_range = zone_north - zone_south
                    lon_range = zone_east - zone_west
                    south = zone_south - (lat_range * 0.05)
                    west = zone_west - (lon_range * 0.05)
                    north = zone_north + (lat_range * 0.05)
                    east = zone_east + (lon_range * 0.05)
            else:
                # No zones or zones only - use combined bounds from all available layers
                # Add small buffer to combined bounds
                lat_range = north - south
                lon_range = east - west
                south -= (lat_range * 0.02)
                west -= (lon_range * 0.02)
                north += (lat_range * 0.02)
                east += (lon_range * 0.02)
            
            return south, west, north, east
    except Exception as e:
        st.warning(f"Bounds calculation error: {e}")
        
    return 25.0, -125.0, 49.0, -66.0  # fallback USA

# --- Legend Rendering with Auto-Compress ---
def add_legend_html(m, html_content, base_offset=20, spacing=115):
    """
    Adds legend HTML with auto-compress to prevent off-screen overflow.
    """
    if "legend_counter" not in st.session_state:
        st.session_state["legend_counter"] = 0
    else:
        st.session_state["legend_counter"] += 1

    # Max visible legends before compressing spacing
    max_legends = 6
    spacing = 110
    compress_spacing = 85  # tighter stack when >6 legends
    legend_count = st.session_state["legend_counter"]

    if legend_count >= max_legends:
        offset_px = 20 + (legend_count * compress_spacing)
    else:
        offset_px = 20 + (legend_count * spacing)

    legend_html = f"""
    <div class="legend-control" style="
        position:absolute;
        top:{offset_px}px;
        left:20px;
        z-index:9999;
        background:rgba(30,30,30,0.25);
        color:white;
        padding:6px 10px;
        border-radius:6px;
        font-size:13px;
        line-height:1.4;
        pointer-events:none;
        width:200px;
    ">
        {html_content}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

def add_heatmap_overlay(m, df, values, name, cmap, show_default, bounds, z_index=1000):
    try:
        # Bail if nothing to draw
        if df is None or df.empty:
            return None, None

        # CRITICAL: Validate that coordinate columns contain true geographic data
        def is_valid_lat(series):
            """Check if series contains valid latitude values."""
            try:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric) == 0:
                    return False
                return numeric.min() >= -90 and numeric.max() <= 90
            except:
                return False
        
        def is_valid_lon(series):
            """Check if series contains valid longitude values."""
            try:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric) == 0:
                    return False
                return numeric.min() >= -180 and numeric.max() <= 180
            except:
                return False
        
        # Find and validate coord columns
        latc = find_col(df, ["latitude"]) or "Latitude"
        lonc = find_col(df, ["longitude"]) or "Longitude"
        
        # Check if columns exist
        if latc not in df.columns or lonc not in df.columns:
            return None, None
        
        # Validate that columns contain real geographic coordinates
        if not is_valid_lat(df[latc]) or not is_valid_lon(df[lonc]):
            return None, None
        
        # Sanitize and keep only good rows
        df = df.copy()
        df[latc] = pd.to_numeric(df[latc], errors="coerce")
        df[lonc] = pd.to_numeric(df[lonc], errors="coerce")
        df[values.name] = pd.to_numeric(df[values.name], errors="coerce")
        df.dropna(subset=[latc, lonc, values.name], inplace=True)
        df = df[np.isfinite(df[latc]) & np.isfinite(df[lonc]) & np.isfinite(df[values.name])]

        # Still nothing? skip
        if df.empty or len(df) < 3:
            return None, None

        # Use provided bounds
        south, west, north, east = bounds

        vmin, vmax = float(df[values.name].min()), float(df[values.name].max())
        if vmin == vmax:
            vmax = vmin + 1.0

        pts_lon = df[lonc].astype(float).values
        pts_lat = df[latc].astype(float).values
        vals_ok = df[values.name].astype(float).values

        # Yield grid bounds = actual yield data extent (source of truth)
        xmin, xmax = df[lonc].min(), df[lonc].max()
        ymin, ymax = df[latc].min(), df[latc].max()
        
        # Apply small buffer (2%) to avoid edge artifacts
        lon_range = xmax - xmin
        lat_range = ymax - ymin
        buffer_pct = 0.02
        xmin -= lon_range * buffer_pct
        xmax += lon_range * buffer_pct
        ymin -= lat_range * buffer_pct
        ymax += lat_range * buffer_pct
        
        # Zones are ONLY used as an optional clipping mask, not for grid generation
        zones_gdf = st.session_state.get("zones_gdf")
        field_polygon = None
        if zones_gdf is not None and not getattr(zones_gdf, "empty", True):
            field_polygon = zones_gdf.geometry.unary_union
        
        # High resolution grid for swath-level detail
        n = 350  # was 500; visual quality holds, load time improves
        lon_lin = np.linspace(xmin, xmax, n)
        lat_lin = np.linspace(ymin, ymax, n)
        lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)

        # Use NEAREST interpolation to preserve blocky yield texture
        grid = griddata((pts_lon, pts_lat), vals_ok, (lon_grid, lat_grid), method="nearest")

        # Harvest extent mask - alpha shape (concave hull)
        try:
            from shapely.geometry import Point as ShapePoint
            from shapely.vectorized import contains
            
            yield_points_geom = gpd.GeoSeries(
                [ShapePoint(x, y) for x, y in zip(pts_lon, pts_lat)],
                crs="EPSG:4326"
            )
            
            if ALPHA_OK and alphashape is not None:
                try:
                    alpha_value = 0.0025
                    harvest_hull = alphashape.alphashape(yield_points_geom, alpha_value)
                except Exception:
                    harvest_hull = yield_points_geom.unary_union.convex_hull
            else:
                harvest_hull = yield_points_geom.unary_union.convex_hull
            
            harvest_mask = contains(harvest_hull, lon_grid, lat_grid)
            grid = np.where(harvest_mask, grid, np.nan)
            
        except Exception:
            pass

        # Clip grid to field polygon if available
        if field_polygon is not None:
            try:
                from shapely.vectorized import contains
                mask = contains(field_polygon, lon_grid, lat_grid)
                grid = np.where(mask, grid, np.nan)
            except Exception:
                pass

        # Create professional yield colormap (red to green gradient)
        yield_cmap = mpl_colors.LinearSegmentedColormap.from_list(
            "yieldmap", ["#a50026", "#d73027", "#fdae61", "#ffffbf", "#a6d96a", "#1a9850"]
        )
        
        # Use percentile-based normalization to remove outlier distortion
        vmin = np.nanpercentile(vals_ok, 5)
        vmax = np.nanpercentile(vals_ok, 95)
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Apply colormap with percentile normalization
        rgba = yield_cmap(norm(grid))
        rgba = np.flipud(rgba)
        rgba = (rgba * 255).astype(np.uint8)

        # Add yield map overlay
        overlay_bounds = [[ymin, xmin], [ymax, xmax]]
        
        # Create overlay with z-index for layer ordering
        overlay = folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=overlay_bounds,
            opacity=1.0,
            name=name,
            overlay=True,
            show=show_default
        )
        
        # Add overlay to map
        overlay.add_to(m)
        
        # Add z-index control using proper Template handling
        overlay_script = """
        {% macro script(this, kwargs) %}
        {{this._parent.get_name()}}.on('add', function() {
            if (this._container) {
                this._container.style.zIndex = """ + str(z_index) + """;
            }
        });
        {% endmacro %}
        """
        script_element = MacroElement()
        script_element._template = Template(overlay_script)
        m.get_root().add_child(script_element)
        
        # Create invisible GeoJSON hover layer for profit/yield overlays
        if "profit" in name.lower() or "yield" in name.lower():
            # Build hover points from sampled data (max 300 points for performance)
            hover_points = []
            sample_size = min(len(df), 300)
            sampled_df = df.sample(n=sample_size) if len(df) > sample_size else df
            
            for _, row in sampled_df.iterrows():
                try:
                    lat_val = row[latc]
                    lon_val = row[lonc]
                    data_val = row[values.name]
                    
                    if pd.notna(lat_val) and pd.notna(lon_val) and pd.notna(data_val):
                        hover_points.append({
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [float(lon_val), float(lat_val)]},
                            "properties": {name: float(data_val)}
                        })
                except Exception:
                    continue
            
            if hover_points:
                hover_geojson = folium.GeoJson(
                    {"type": "FeatureCollection", "features": hover_points},
                    name=f"{name} (Hover)",
                    show=False,
                    style_function=lambda x: {"opacity": 0, "fillOpacity": 0},
                    highlight_function=None,
                    tooltip=None
                )
                m.add_child(hover_geojson)
                hover_geojson.layer_name = None  # Prevent showing in layer control

        return vmin, vmax

    except Exception as e:
        st.warning(f"Yield map overlay fallback triggered: {e}")
        return None, None

# ===========================
# MAIN APP — HARDENED + STACKED LEGENDS
# ===========================
apply_compact_theme()
_bootstrap_defaults()

# --- Reset legend counter each rerun ---
st.session_state["legend_counter"] = 0

# ==============================================================
# 🔒 FINAL SCROLLBAR + HEIGHT NORMALIZER
# ==============================================================
st.markdown("""
<style>
/* Hide all scrollbars */
[data-testid="stDataFrameScrollableContainer"] {
  overflow: hidden !important;
  scrollbar-width: none !important;
  height: auto !important;
}
[data-testid="stDataFrameScrollableContainer"]::-webkit-scrollbar {
  display: none !important;
}
/* Normalize text sizing and remove phantom borders */
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
  font-size: 13px !important;
  line-height: 1.1 !important;
  border: none !important;
}
</style>

<script>
window.addEventListener("load", () => {
  // Wait for all dataframes to mount
  setTimeout(() => {
    const frames = document.querySelectorAll('[data-testid="stDataFrameScrollableContainer"]');
    frames.forEach(f => {
      // Measure internal table height and apply directly
      const table = f.querySelector("table");
      if (table) {
        const h = table.offsetHeight + 4; // +4 for bottom clip
        f.style.height = h + "px";
        f.style.overflow = "hidden";
        f.style.scrollbarWidth = "none";
      }
    });
  }, 400);  // slight delay ensures Streamlit finished virtualizing
});
</script>
""", unsafe_allow_html=True)

# --- Prevent outer scrollbar and fix map container height ---
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] {
    overflow: hidden !important;
}

[data-testid="stVerticalBlock"] {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow-y: hidden !important;
}

iframe[title="st_folium"] {
    height: 88vh !important;    /* dynamically fits viewport without clipping */
    max-height: 88vh !important;
}
</style>
""", unsafe_allow_html=True)

# Canonical default for sell price
if "sell_price" not in st.session_state:
    st.session_state["sell_price"] = 0.0

# One-time migration: retire the old widget key that held 5.00
# (Don't clear the whole session; just drop this sticky key.)
for old_key in ("sell_price_input", "sell_price_old"):
    if old_key in st.session_state:
        del st.session_state[old_key]

render_uploaders()
render_fixed_inputs_and_strip()
# Three collapsible input dropdowns ABOVE the map
render_input_sections()
st.markdown("---")

# ---------- build base map ----------
m = make_base_map()

# ---------- STACKED LEGEND SYSTEM ----------
def init_legend_rails(m):
    """Injects fixed legend containers (top-left rail used)."""
    rails_css = """
    <style>
      .legend-rail { 
        position:absolute; 
        z-index:9999; 
        font-family:sans-serif; 
        display:flex; 
        flex-direction:column; 
        gap:10px;
      }
      #legend-tl { top: 14px; left: 10px; width: 220px; }
      .legend-card {
        color: #b0b3b8 !important; 
        text-shadow: 0 0 3px rgba(0,0,0,0.5) !important;
        font-weight: 500 !important;
        background: rgba(255,255,255,0.0);
        padding: 6px 10px; 
        border-radius: 6px;
        box-shadow: none;
        user-select: none;
      }
      .legend-title { font-weight: 600; margin-bottom: 4px; }
      .legend-bar { height: 14px; border-radius: 2px; margin-bottom: 4px; }
      .legend-minmax { display:flex; justify-content:space-between; font-size:12px; }
    </style>
    <div id="legend-tl" class="legend-rail"></div>
    """
    m.get_root().html.add_child(folium.Element(rails_css))
    st.session_state.setdefault("_legend_counts", {"tl": 0})

def add_gradient_legend_pos(m, name, vmin, vmax, cmap, corner="tl"):
    """Adds a gradient legend to the chosen rail (top-left default)."""
    if vmin is None or vmax is None:
        return
    stops = [f"{mpl_colors.rgb2hex(cmap(i/100.0)[:3])} {i}%" for i in range(0, 101, 10)]
    gradient_css = ", ".join(stops)
    idx = st.session_state.get("_legend_counts", {}).get(corner, 0)
    card_html = f"""
    <div class="legend-card" id="legend-{corner}-{idx}">
      <div class="legend-title">{name}</div>
      <div class="legend-bar" style="background:linear-gradient(90deg, {gradient_css});"></div>
      <div class="legend-minmax"><span>{vmin:.1f}</span><span>{vmax:.1f}</span></div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(f"""
      <script>
        (function() {{
          var rail = document.getElementById("legend-{corner}");
          if (rail) {{
            rail.insertAdjacentHTML("beforeend", `{card_html}`);
          }}
        }})();
      </script>
    """))
    st.session_state["_legend_counts"][corner] = idx + 1

# Reset legend priorities to ensure proper ordering on each render
st.session_state["_legend_priorities"] = []
print(f"DEBUG: Legend priorities reset for proper ordering")

# Initialize the legend rail
init_legend_rails(m)

# ---------- WORKING ZONE LOGIC WITH AUTO-ZOOM ----------
zones_gdf = st.session_state.get("zones_gdf")
yield_gdf = st.session_state.get("_yield_gdf_raw")

# Calculate map center and zoom based on available data
map_center = [39.8283, -98.5795]  # US center fallback
map_zoom = 5

if yield_gdf is not None and not getattr(yield_gdf, "empty", True):
    try:
        # Use yield data bounds for map center and zoom
        bounds = yield_gdf.total_bounds
        map_center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        map_zoom = 15
    except Exception:
        pass

elif zones_gdf is not None and not getattr(zones_gdf, "empty", True):
    try:
        # Fallback to zones if no yield data
        bounds = zones_gdf.total_bounds
        map_center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        map_zoom = 15
    except Exception:
        pass

# Update the map with proper center and zoom
m.location = map_center
m.zoom_start = map_zoom

if zones_gdf is not None and not getattr(zones_gdf, "empty", True):
    try:
        zones_gdf = zones_gdf.copy()
        try:
            zones_gdf["geometry"] = zones_gdf.geometry.buffer(0)
        except Exception:
            pass
        
        # Only explode if there are multi-part geometries and we want to split them
        # Comment out explosion to keep original zone count
        # try:
        #     zones_gdf = zones_gdf.explode(index_parts=False, ignore_index=True)
        # except TypeError:
        #     zones_gdf = zones_gdf.explode().reset_index(drop=True)

        palette = ["#FF0000", "#FF8C00", "#FFFF00", "#32CD32", "#006400",
                   "#1E90FF", "#8A2BE2", "#FFC0CB", "#A52A2A", "#00CED1"]
        unique_vals = list(dict.fromkeys(sorted(list(zones_gdf["Zone"].astype(str).unique()))))
        color_map = {z: palette[i % len(palette)] for i, z in enumerate(unique_vals)}

        # Add zone fill
        folium.GeoJson(
            zones_gdf,
            name="Zones (Fill)",
            style_function=lambda feature: {
                "fillColor": color_map.get(str(feature["properties"].get("Zone", "")), "#808080"),
                "color": "#202020",
                "weight": 1,
                "fillOpacity": 0.25,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["Zone", "Calculated Acres", "Override Acres"] if c in zones_gdf.columns]
            ),
            show=True  # Zones ON by default
        ).add_to(m)

        # Add zone boundaries with thick black lines
        folium.GeoJson(
            zones_gdf,
            name="Zone Outlines (Top)",
            style_function=lambda feature: {
                "fillOpacity": 0,
                "color": "#000000",
                "weight": 3,
                "opacity": 1.0,
            },
            tooltip=None,
            show=True  # Zone outlines always ON
        ).add_to(m)

        legend_items = "".join(
            f"<div style='display:flex;align-items:center;margin:2px 0;'>"
            f"<div style='background:{color_map[z]};width:14px;height:14px;margin-right:6px;'></div>{z}</div>"
            for z in unique_vals
        )
        legend_html = f"""
        <div id="zone-legend" style="position:absolute; bottom:20px; right:20px; z-index:9999;
                     font-family:sans-serif; font-size:13px; color:white;
                     background-color: rgba(255,255,255,0.0); padding:6px 10px; border-radius:5px; width:160px;">
          <div style="font-weight:600; margin-bottom:4px; cursor:pointer;"
               onclick="var x = document.getElementById('zone-legend-items');
                        if (x.style.display === 'none') {{ x.style.display = 'block'; }}
                        else {{ x.style.display = 'none'; }}">
            Zone Colors ▼
          </div>
          <div id="zone-legend-items" style="display:block;">{legend_items}</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
    except Exception as e:
        st.warning(f"Skipping zones overlay: {e}")

# ---------- Prescription overlays ----------
# Distinct color palettes for fertilizer RX layers
palette_options = [plt.cm.Blues, plt.cm.Purples, plt.cm.Oranges, plt.cm.Greys]

legend_ix = 0
seed_gdf = st.session_state.get("seed_gdf")
if seed_gdf is not None and not getattr(seed_gdf, "empty", True):
    print(f"DEBUG: Adding Seed RX overlay at legend_ix={legend_ix}")
    add_prescription_overlay(m, seed_gdf, "Seed RX", plt.cm.Greens, legend_ix)
    legend_ix += 1

for idx, (k, fgdf) in enumerate(st.session_state.get("fert_gdfs", {}).items()):
    if fgdf is not None and not fgdf.empty:
        # Use distinct colormap for each fertilizer layer
        cmap = palette_options[idx % len(palette_options)]
        print(f"DEBUG: Adding Fertilizer RX '{k}' at legend_ix={legend_ix}")
        add_prescription_overlay(m, fgdf, f"Fertilizer RX: {k}", cmap, legend_ix)
        legend_ix += 1

print(f"DEBUG: Total prescription overlays added: {legend_ix}")

# ---------- Heatmaps (yield / profits) — FOOLPROOF ----------
bounds = compute_bounds_for_heatmaps()
ydf = st.session_state.get("yield_df")

# =========================================================
# DEFENSIVE CONVERSION — SINGLE BRANCH STRUCTURE
# =========================================================
df_for_maps = pd.DataFrame()

if isinstance(ydf, (pd.DataFrame, gpd.GeoDataFrame)) and not ydf.empty:
    try:
        # Convert to regular DataFrame first - PRESERVE COORDINATES
        if isinstance(ydf, gpd.GeoDataFrame):
            # Keep all columns including Latitude/Longitude that were just created
            df_for_maps = pd.DataFrame(ydf.drop(columns="geometry", errors="ignore"))
        else:
            df_for_maps = pd.DataFrame(ydf.copy())

        # Use existing coordinate columns if available
        if "Latitude" in df_for_maps.columns and "Longitude" in df_for_maps.columns:
            # Coordinates already exist, use them
            pass
        else:
            # Extract coordinates from geometry if needed
            if isinstance(ydf, gpd.GeoDataFrame) and "geometry" in ydf.columns:
                try:
                    reps = ydf.geometry.representative_point()
                    df_for_maps["Longitude"] = reps.x
                    df_for_maps["Latitude"] = reps.y
                except Exception as e:
                    st.warning(f"Coordinate extraction failed: {e}")
                    df_for_maps["Longitude"], df_for_maps["Latitude"] = np.nan, np.nan

        # Normalize coord column names if provided in CSV/JSON
        lower_cols = {c.lower(): c for c in df_for_maps.columns}
        if "longitude" in lower_cols and "Longitude" not in df_for_maps.columns:
            df_for_maps.rename(columns={lower_cols["longitude"]: "Longitude"}, inplace=True)
        if "latitude" in lower_cols and "Latitude" not in df_for_maps.columns:
            df_for_maps.rename(columns={lower_cols["latitude"]: "Latitude"}, inplace=True)

        # Detect/normalize yield column
        yield_candidates = [
            "yield", "dry_yield", "wet_yield", "yld_mass_dr", "yld_vol_dr",
            "yld_mass_wt", "yld_vol_wt", "crop_flw_m", "yld_bu_ac", "prod_yield", "harvestyield"
        ]
        ycol = next((c for c in df_for_maps.columns if c.lower() in yield_candidates or "yld" in c.lower()), None)
        if ycol and ycol != "Yield":
            df_for_maps.rename(columns={ycol: "Yield"}, inplace=True)

        # Ensure we have the required columns
        if "Latitude" not in df_for_maps.columns:
            df_for_maps["Latitude"] = np.nan
        if "Longitude" not in df_for_maps.columns:
            df_for_maps["Longitude"] = np.nan
        if "Yield" not in df_for_maps.columns:
            df_for_maps["Yield"] = 0.0

        # Convert to numeric
        df_for_maps["Latitude"] = pd.to_numeric(df_for_maps["Latitude"], errors="coerce")
        df_for_maps["Longitude"] = pd.to_numeric(df_for_maps["Longitude"], errors="coerce")
        df_for_maps["Yield"] = pd.to_numeric(df_for_maps["Yield"], errors="coerce").fillna(0)

    except Exception as e:
        st.warning(f"Data conversion failed: {e}")
        df_for_maps = pd.DataFrame(columns=["Latitude", "Longitude", "Yield"])
else:
    # Fallback empty DF if ydf missing
    df_for_maps = pd.DataFrame(columns=["Latitude", "Longitude", "Yield"])

# =========================================================
# SELECT ONLY ROWS WITH VALID COORDS FOR MAPPING (NO FULL WIPE)
# =========================================================
try:
    if not df_for_maps.empty and "Latitude" in df_for_maps.columns and "Longitude" in df_for_maps.columns:
        # Ensure columns are Series (not DataFrame)
        lat_series = df_for_maps["Latitude"]
        lon_series = df_for_maps["Longitude"]
        
        # Check if they are Series
        if hasattr(lat_series, 'between') and hasattr(lon_series, 'between'):
            valid_mask = (
                lat_series.between(-90, 90)
                & lon_series.between(-180, 180)
                & lat_series.notna()
                & lon_series.notna()
            )
            df_valid = df_for_maps.loc[valid_mask].copy()
        else:
            # Fallback: filter manually
            df_valid = df_for_maps[
                (df_for_maps["Latitude"] >= -90) & (df_for_maps["Latitude"] <= 90) &
                (df_for_maps["Longitude"] >= -180) & (df_for_maps["Longitude"] <= 180) &
                df_for_maps["Latitude"].notna() & df_for_maps["Longitude"].notna()
            ].copy()
        
        if df_valid.empty:
            df_valid = df_for_maps.copy()
    else:
        df_valid = df_for_maps.copy()
except Exception:
    df_valid = df_for_maps.copy()

if not df_valid.empty:
    try:
        # Clip extreme yield outliers (5–95%)
        if df_valid["Yield"].dropna().size > 0:
            low, high = np.nanpercentile(df_valid["Yield"], [5, 95])
            if np.isfinite(low) and np.isfinite(high) and low < high:
                df_valid["Yield"] = df_valid["Yield"].clip(lower=low, upper=high)

        # Normalize metric conversions
        if df_valid["Yield"].max() > 400:
            df_valid["Yield"] = df_valid["Yield"] / 15.93

        # Use flexible bounds calculation that works with any layer combination
        bounds = compute_bounds_for_heatmaps()
        
        # If no bounds calculated, fall back to yield data bounds
        if bounds == (25.0, -125.0, 49.0, -66.0) and "Latitude" in df_valid.columns and "Longitude" in df_valid.columns:
            south = float(df_valid["Latitude"].min())
            west = float(df_valid["Longitude"].min())
            north = float(df_valid["Latitude"].max())
            east = float(df_valid["Longitude"].max())
            bounds = (south, west, north, east)
    except Exception:
        pass

    # =========================================================
    # DYNAMIC PROFIT METRICS - RECALCULATES ON EVERY INPUT CHANGE
    # =========================================================
    try:
        # Calculate base expenses from fixed inputs
        expenses = st.session_state.get("expenses_dict", {})
        base_expenses_per_acre = float(sum(expenses.values()) if expenses else 0.0)

        # Calculate variable rate costs from uploaded prescription maps
        fert_var = seed_var = 0.0
        
        # Get variable rate inputs from the editor
        var_inputs = st.session_state.get("variable_rate_inputs", pd.DataFrame())
        if not var_inputs.empty:
            total_var_cost = var_inputs["Units Applied"] * var_inputs["Price per Unit ($)"]
            total_var_cost = pd.to_numeric(total_var_cost, errors="coerce").fillna(0).sum()
            
            # Distribute cost across fertilizer vs seed based on product types
            fert_products = var_inputs[var_inputs["Type"] == "Fertilizer"]
            seed_products = var_inputs[var_inputs["Type"] == "Seed"]
            
            if not fert_products.empty:
                fert_var = pd.to_numeric(fert_products["Units Applied"] * fert_products["Price per Unit ($)"], errors="coerce").fillna(0).sum()
            if not seed_products.empty:
                seed_var = pd.to_numeric(seed_products["Units Applied"] * seed_products["Price per Unit ($)"], errors="coerce").fillna(0).sum()

        # Calculate fixed rate costs
        fx = st.session_state.get("fixed_products", pd.DataFrame())
        fixed_costs = 0.0
        if not fx.empty and "$/ac" in fx.columns:
            fixed_costs = pd.to_numeric(fx["$/ac"], errors="coerce").fillna(0).sum()

        # Get current sell price (always reactive to input changes)
        active_sell_price = float(st.session_state.get("sell_price", 0.0))
        
        # Recalculate profit metrics with current inputs
        if "Yield" in df_for_maps.columns:
            df_for_maps["Revenue_per_acre"] = df_for_maps["Yield"] * active_sell_price
            df_for_maps["NetProfit_Variable"] = df_for_maps["Revenue_per_acre"] - (
                base_expenses_per_acre + fert_var + seed_var
            )
            df_for_maps["NetProfit_Fixed"] = df_for_maps["Revenue_per_acre"] - (
                base_expenses_per_acre + fixed_costs
            )
        else:
            df_for_maps["Revenue_per_acre"] = 0.0
            df_for_maps["NetProfit_Variable"] = 0.0
            df_for_maps["NetProfit_Fixed"] = 0.0
            
        # Store calculated values for hover popup access
        st.session_state["_profit_calculated"] = {
            "base_expenses": base_expenses_per_acre,
            "fert_var": fert_var,
            "seed_var": seed_var,
            "fixed_costs": fixed_costs,
            "sell_price": active_sell_price
        }
    except Exception as e:
        st.warning(f"Profit calculation fallback triggered: {e}")
        for c in ["Revenue_per_acre", "NetProfit_Variable", "NetProfit_Fixed"]:
            df_for_maps[c] = 0.0

    # =========================================================
    # RENDER HEATMAPS IN PRIORITY ORDER: PROFIT > YIELD > OTHERS
    # =========================================================
    def safe_overlay(colname, title, cmap, show_default, z_index=1000):
        if colname not in df_for_maps.columns or df_for_maps.empty:
            return None, None
        try:
            # Create overlay with specified z-index for layer ordering
            vmin, vmax = add_heatmap_overlay(
                m, df_for_maps, df_for_maps[colname], title, cmap, show_default, bounds, z_index
            )
            return vmin, vmax
        except Exception:
            return None, None

    # 1. PROFIT LAYER (TOP PRIORITY) - Always render first for top z-index
    profit_vmin = profit_vmax = None
    if st.session_state.get("sell_price", 0) > 0:
        profit_vmin, profit_vmax = safe_overlay("NetProfit_Variable", "Variable Rate Profit ($/ac)", plt.cm.RdYlGn, True, z_index=3000)
        print(f"DEBUG: Variable Profit overlay returned vmin={profit_vmin}, vmax={profit_vmax}")
        if profit_vmin is not None:
            add_gradient_legend(m, "Variable Rate Profit ($/ac)", profit_vmin, profit_vmax, plt.cm.RdYlGn)
            print(f"DEBUG: Added Variable Profit legend")

        # Fixed Profit overlay (if data exists)
        fx = st.session_state.get("fixed_products", pd.DataFrame())
        has_fixed_costs = (
            not fx.empty and
            pd.to_numeric(fx.get("$/ac", []), errors="coerce").fillna(0).sum() > 0
        )

        if has_fixed_costs:
            fmin, fmax = safe_overlay("NetProfit_Fixed", "Fixed Rate Profit ($/ac)", plt.cm.RdYlGn, False, z_index=2999)
            print(f"DEBUG: Fixed Profit overlay returned fmin={fmin}, fmax={fmax}")
            if fmin is not None:
                add_gradient_legend(m, "Fixed Rate Profit ($/ac)", fmin, fmax, plt.cm.RdYlGn)
                print(f"DEBUG: Added Fixed Profit legend")
    else:
        # ⚠️ Show small disclaimer when sell price not entered
        st.markdown(
            """
            <div style="color:#b0b3b8; font-size:14px; margin-bottom:8px;
                        background:rgba(255,255,255,0.05); padding:6px 10px;
                        border-radius:6px; width:fit-content;">
            ⚠️ Enter a valid sell price above to generate profit maps.
            </div>
            """,
            unsafe_allow_html=True
        )

    # 2. YIELD LAYER (SECOND PRIORITY)
    ymin, ymax = safe_overlay("Yield", "Yield (bu/ac)", plt.cm.RdYlGn, True, z_index=2000)
    print(f"DEBUG: Yield overlay returned ymin={ymin}, ymax={ymax}")
    if ymin is not None:
        add_gradient_legend(m, "Yield (bu/ac)", ymin, ymax, plt.cm.RdYlGn)
        print(f"DEBUG: Added Yield legend")

# Final fit using all active layers
try:
    if not df_for_maps.empty and "Latitude" in df_for_maps.columns and "Longitude" in df_for_maps.columns:
        lat_vals = df_for_maps["Latitude"].dropna()
        lon_vals = df_for_maps["Longitude"].dropna()
        if len(lat_vals) > 0 and len(lon_vals) > 0:
            south, north = lat_vals.min(), lat_vals.max()
            west, east = lon_vals.min(), lon_vals.max()
        else:
            south, west, north, east = compute_bounds_for_heatmaps()
    else:
        south, west, north, east = compute_bounds_for_heatmaps()
    
    m.fit_bounds([[south, west], [north, east]])
except Exception:
    pass

# Add layer control to make layers toggleable
folium.LayerControl(collapsed=False).add_to(m)

# --- LAYER VISIBILITY: Profit ON by default, others OFF ---
# (Note: Folium layers are already added with show=True/False in safe_overlay calls)
# This just sets the visual hierarchy via z-index if needed

# === FINAL Legend CSS Override (absolute transparency + enforced stacking) ===
legend_css = """
<style>
/* Completely transparent legends inside the Folium iframe */
.leaflet-control br,
.legend, .leaflet-control-layers, .branca-colormap, .legend-control {
    background: none !important;
    background-color: rgba(255,255,255,0.0) !important;
    box-shadow: none !important;
    border: none !important;
    color: #b0b3b8 !important;
    text-shadow: 0 0 3px rgba(0,0,0,0.5) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    backdrop-filter: none !important;
    opacity: 1.0 !important;
}

/* Separate and pin legends vertically */
.leaflet-top.leaflet-left .legend,
.leaflet-top.leaflet-left .branca-colormap {
    margin-top: 20px !important;
    margin-left: 20px !important;
    position: relative !important;
    display: block !important;
    float: none !important;
    clear: both !important;
}

/* Add fixed vertical gap between every legend block */
.leaflet-top.leaflet-left .legend + .legend,
.leaflet-top.leaflet-left .legend + .branca-colormap,
.leaflet-top.leaflet-left .branca-colormap + .legend,
.leaflet-top.leaflet-left .branca-colormap + .branca-colormap {
    margin-top: 12px !important;
}

/* Ensure these rules override Folium/Leaflet defaults */
.leaflet-control, .branca-colormap {
    z-index: 9999 !important;
}
</style>
"""
m.get_root().header.add_child(folium.Element(legend_css))

# =========================================================
# LAYER STACK + LEGEND ORDER CONTROL
# =========================================================

# --- AUTO-TOGGLE DEFAULTS ---
defaults = {
    "profit_layer": True,
    "yield_layer": True,
    "zones_layer": True,
    "zones_outline_layer": True,
    "fert_layer": False,
    "seed_layer": False
}
for k, v in defaults.items():
    st.session_state[k] = v

# Enforce visual stack order
def _front(key):
    lyr = st.session_state.get(key)
    if lyr:
        try: lyr.add_to(m)
        except: pass

for key in ["profit_layer", "yield_layer", "zones_layer", "zones_outline_layer"]:
    _front(key)


# --- INJECT JS: HOVER INSPECTOR (FINAL VERSION) ---
hover_js = """
<script>
(function(){
  window.addEventListener("load", () => {
    setTimeout(() => {
      if (!window.map) return;
      
      window.map.on('mousemove', function(e) {
        const hoverLayers = [];
        window.map.eachLayer(function(layer) {
          if (layer.options && layer.options.name && layer.options.name.includes('(Hover)')) {
            hoverLayers.push(layer);
          }
        });

        const popup = L.popup({offset: L.point(5, -10), autoClose: true, closeButton: false});
        let content = '';

        hoverLayers.forEach(layer => {
          layer.eachLayer(point => {
            if (point.getLatLng && point.getLatLng().distanceTo(e.latlng) < 25) {  // 25m hover radius
              const props = point.feature.properties;
              for (const key in props) {
                const value = parseFloat(props[key]).toFixed(1);
                content += `<b>${key}:</b> ${value}<br>`;
              }
            }
          });
        });

        if (content) {
          popup.setLatLng(e.latlng).setContent(content).openOn(window.map);
          setTimeout(() => window.map.closePopup(popup), 400);
        }
      });
    }, 600);
  });
})();
</script>
"""
m.get_root().html.add_child(folium.Element(hover_js))

# Popup CSS
popup_css = """
<style>
.multi-layer-popup{
  background:rgba(25,25,25,0.88);
  color:#fff;
  padding:6px 10px;
  border-radius:6px;
  font-size:13px;
  line-height:1.3;
  box-shadow:0 2px 6px rgba(0,0,0,0.4);
  backdrop-filter:blur(2px);
  pointer-events:none;
}
</style>
"""
m.get_root().header.add_child(folium.Element(popup_css))

# Dynamic map key for refresh on file upload (controlled)
map_key = f"main_map_{st.session_state.get('map_refresh_trigger', 0)}"

# === DIAGNOSTIC: Pre-render legend check ===
print(f"DEBUG: Final legend counter state before render: {st.session_state.get('_legend_counts')}")
html_preview = m.get_root().render()
legend_snips = [line for line in html_preview.splitlines() if "legend" in line.lower()]
print(f"=== LEGEND HTML PREVIEW START (found {len(legend_snips)} lines) ===")
for l in legend_snips[:20]:
    print(l)
print("=== LEGEND HTML PREVIEW END ===")

# Force correct visual stacking
layer_order = [
    "Variable Rate Profit ($/ac)",
    "Yield (bu/ac)",
    "Zones (Fill)",
    "Zone Outlines (Top)"
]
for name in layer_order:
    for layer in m._children.values():
        if getattr(layer, "name", None) == name:
            layer.add_to(m)

# --- Auto-fit map to full field extent ---
try:
    bounds = None
    # Priority: yield extent → zone extent → fertilizer extent
    if "ydf" in locals() and not ydf.empty:
        bounds = [[ydf["Latitude"].min(), ydf["Longitude"].min()],
                  [ydf["Latitude"].max(), ydf["Longitude"].max()]]
    elif "zones_gdf" in st.session_state and not st.session_state["zones_gdf"].empty:
        z = st.session_state["zones_gdf"].total_bounds
        bounds = [[z[1], z[0]], [z[3], z[2]]]
    elif "fert_gdf" in st.session_state and not st.session_state["fert_gdf"].empty:
        f = st.session_state["fert_gdf"].total_bounds
        bounds = [[f[1], f[0]], [f[3], f[2]]]

    if bounds:
        m.fit_bounds(bounds, padding=(20, 20))
        print("Fitting bounds:", bounds)
except Exception as e:
    print(f"Auto-fit skipped: {e}")

# --- Stable Map Rendering (Prevents Flicker) ---
if "map_drawn" not in st.session_state:
    st.session_state["map_drawn"] = False

if not st.session_state["map_drawn"]:
    st.session_state["map_drawn"] = True
    st_data = st_folium(
        m,
        width=1500,
        height=720,  # slightly shorter to align with new CSS vh scaling
        returned_objects=["last_active_drawing"]
    )
else:
    with st.spinner("Map stable. Hover and layer controls active..."):
        st_data = st_folium(
            m,
            width=1500,
            height=720,  # slightly shorter to align with new CSS vh scaling
            returned_objects=["last_active_drawing"]
        )

# === DIAGNOSTIC: Post-render check ===
print("Map rendered; checking post-render legend presence…")

# =========================================================
# PROFIT SUMMARY — BULLETPROOF STATIC TABLES (NO SCROLL)
# =========================================================
def render_profit_summary():
    st.header("Profit Summary")

    # ---------- helpers ----------
    def fmt_money(x):
        try:
            return f"${x:,.2f}"
        except Exception:
            return x

    def color_profit(val):
        try:
            v = float(val)
        except Exception:
            return "color:white;"
        if v > 0:
            return "color:limegreen;font-weight:bold;"
        elif v < 0:
            return "color:#ff4d4d;font-weight:bold;"
        else:
            return "font-weight:bold;color:white;"

    def render_static_table(df: pd.DataFrame, title: str):
        """Render static HTML table — no scrollbars, fully expanded."""
        if df is None or df.empty:
            st.info(f"No data available for {title}.")
            return
        # Build header
        html = f"<h5 style='margin-top:10px;margin-bottom:6px;'>{title}</h5>"
        html += """
        <style>
        .static-table { width:100%; border-collapse:collapse; font-size:0.9rem; }
        .static-table th, .static-table td {
            border:1px solid #444; padding:4px 6px; text-align:right;
        }
        .static-table th {
            background-color:#111; color:white; font-weight:600; text-align:left;
        }
        .static-table td:first-child, .static-table th:first-child { text-align:left; }
        </style>
        <table class='static-table'>
        <thead><tr>""" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr></thead><tbody>"

        # Build rows with conditional coloring
        for _, row in df.iterrows():
            html += "<tr>"
            for c, v in row.items():
                val = fmt_money(v) if isinstance(v, (int, float)) else v
                style = color_profit(v) if "Profit" in c or "Rate" in c else ""
                html += f"<td style='{style}'>{val}</td>"
            html += "</tr>"
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)

    # ---------- Safe defaults ----------
    expenses = st.session_state.get("expenses_dict", {})
    base_exp = float(
        st.session_state.get(
            "base_expenses_per_acre",
            sum(expenses.values()) if expenses else 0.0,
        )
    )
    corn_yield  = float(st.session_state.get("corn_yield", 200))
    corn_price  = float(st.session_state.get("corn_price", 5))
    bean_yield  = float(st.session_state.get("bean_yield", 60))
    bean_price  = float(st.session_state.get("bean_price", 12))
    target_yield = float(st.session_state.get("target_yield", 200))
    sell_price   = float(st.session_state.get("sell_price", corn_price))

    # ---------- Profit math ----------
    revenue_per_acre = target_yield * sell_price
    fixed_inputs = base_exp

    # variable-rate (from RX products)
    fert_costs_var = seed_costs_var = 0.0
    df_fert = st.session_state.get("fert_products")
    if isinstance(df_fert, pd.DataFrame) and "CostPerAcre" in df_fert.columns:
        fert_costs_var = pd.to_numeric(df_fert["CostPerAcre"], errors="coerce").sum()
    df_seed = st.session_state.get("seed_products")
    if isinstance(df_seed, pd.DataFrame) and "CostPerAcre" in df_seed.columns:
        seed_costs_var = pd.to_numeric(df_seed["CostPerAcre"], errors="coerce").sum()
    expenses_var = fixed_inputs + fert_costs_var + seed_costs_var
    profit_var   = revenue_per_acre - expenses_var

    # fixed-rate (from flat products)
    fert_costs_fix = seed_costs_fix = 0.0
    df_fix = st.session_state.get("fixed_products")
    if isinstance(df_fix, pd.DataFrame):
        if "Type" in df_fix.columns and "$/ac" in df_fix.columns:
            fert_costs_fix = pd.to_numeric(df_fix.loc[df_fix["Type"] == "Fertilizer", "$/ac"], errors="coerce").sum()
            seed_costs_fix = pd.to_numeric(df_fix.loc[df_fix["Type"] == "Seed", "$/ac"], errors="coerce").sum()
        elif "$/ac" in df_fix.columns:
            fert_costs_fix = seed_costs_fix = pd.to_numeric(df_fix["$/ac"], errors="coerce").sum()
    expenses_fix = fixed_inputs + fert_costs_fix + seed_costs_fix
    profit_fix   = revenue_per_acre - expenses_fix

    # ---------- Tables ----------
    comparison = pd.DataFrame(
        {
            "Metric": ["Revenue ($/ac)", "Expenses ($/ac)", "Profit ($/ac)"],
            "Breakeven Budget": [revenue_per_acre, fixed_inputs, revenue_per_acre - fixed_inputs],
            "Variable Rate":    [revenue_per_acre, expenses_var,  profit_var],
            "Fixed Rate":       [revenue_per_acre, expenses_fix,  profit_fix],
        }
    )

    cornsoy = pd.DataFrame(
        {
            "Crop": ["Corn", "Soybeans"],
            "Yield (bu/ac)":      [corn_yield, bean_yield],
            "Sell Price ($/bu)":  [corn_price, bean_price],
        }
    )
    cornsoy["Revenue ($/ac)"]      = cornsoy["Yield (bu/ac)"] * cornsoy["Sell Price ($/bu)"]
    cornsoy["Fixed Inputs ($/ac)"] = base_exp
    cornsoy["Profit ($/ac)"]       = cornsoy["Revenue ($/ac)"] - cornsoy["Fixed Inputs ($/ac)"]

    fixed_df = pd.DataFrame(list(expenses.items()), columns=["Expense", "$/ac"])
    if not fixed_df.empty:
        total_row = pd.DataFrame([{"Expense": "Total Fixed Costs", "$/ac": fixed_df["$/ac"].sum()}])
        fixed_df = pd.concat([fixed_df, total_row], ignore_index=True)

    # ---------- Layout ----------
    left, right = st.columns([2, 1], gap="large")

    with left:
        render_static_table(comparison, "Profit Comparison")

        with st.expander("Show Calculation Formulas", expanded=False):
            st.markdown("""
            <div style="display:flex;flex-wrap:wrap;gap:6px;
                        font-size:0.75rem;line-height:1.1rem;">
              <div style="flex:1;min-width:180px;border:1px solid #444;
                          border-radius:6px;padding:5px;background-color:#111;">
                <b>Breakeven Budget</b><br>
                (Target Yield × Sell Price) − Fixed Inputs
              </div>
              <div style="flex:1;min-width:180px;border:1px solid #444;
                          border-radius:6px;padding:5px;background-color:#111;">
                <b>Variable Rate</b><br>
                (Target Yield × Sell Price) − (Fixed Inputs + Var Seed + Var Fert)
              </div>
              <div style="flex:1;min-width:180px;border:1px solid #444;
                          border-radius:6px;padding:5px;background-color:#111;">
                <b>Fixed Rate</b><br>
                (Target Yield × Sell Price) − (Fixed Inputs + Fixed Seed + Fixed Fert)
              </div>
            </div>
            """, unsafe_allow_html=True)

        render_static_table(cornsoy, "Corn vs Soybean Profitability")

    with right:
        render_static_table(fixed_df, "Fixed Input Costs")

# ---------- render ----------
render_profit_summary()
