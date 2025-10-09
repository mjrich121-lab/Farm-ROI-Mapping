# Farm Profit Mapping Tool V4 — BACKUP DEFAULT (CLEAN, BULLETPROOF)
# Save this file as app_backup_default.py
# Minimal, defensive, and well-commented baseline that you can paste over your app

import io
import os
import zipfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point

# -------------------------------
# Page config + small utilities
# -------------------------------
st.set_page_config(page_title="Farm ROI Tool V4 - Backup", layout="wide")

# Clear risky caches (safe to run)
if hasattr(st, "cache_data"):
    try:
        st.cache_data.clear()
    except Exception:
        pass
if hasattr(st, "cache_resource"):
    try:
        st.cache_resource.clear()
    except Exception:
        pass

# -------------------------------
# Session-state defaults
# -------------------------------
if "uploads" not in st.session_state:
    st.session_state["uploads"] = {"yield": None, "zones": None, "prescriptions": None}
if "map_center" not in st.session_state:
    st.session_state["map_center"] = (39.8283, -98.5795)  # US center fallback
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 5
if "profit_summary" not in st.session_state:
    st.session_state["profit_summary"] = {}

# -------------------------------
# Helpers
# -------------------------------

def safe_read_csv(uploaded_file) -> Optional[pd.DataFrame]:
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


def df_to_gdf(df: pd.DataFrame, lat_col: str = None, lon_col: str = None, crs: str = "EPSG:4326") -> Optional[gpd.GeoDataFrame]:
    # Try to detect lat/lon columns robustly
    if df is None or df.empty:
        return None

    if lat_col is None or lon_col is None:
        # common candidates
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
        st.warning("Latitude/Longitude columns not found — returning GeoDataFrame is skipped.")
        return None

    try:
        gdf = gpd.GeoDataFrame(df.copy(), geometry=[Point(xy) for xy in zip(df[lon_col].astype(float), df[lat_col].astype(float))], crs=crs)
        return gdf
    except Exception as e:
        st.error(f"Failed to convert to GeoDataFrame: {e}")
        return None


# -------------------------------
# File upload UI
# -------------------------------
st.sidebar.header("Uploads — backup restore")
with st.sidebar.expander("Upload CSV / Shapefile (yield / prescriptions / zones)", expanded=True):
    yield_file = st.file_uploader("Yield CSV (must contain lat/lon or latitude/longitude)", type=["csv"], key="upload_yield")
    zones_file = st.file_uploader("Zones GeoJSON / Shapefile (zip) — recommended GeoJSON", type=["geojson", "json", "zip"], key="upload_zones")
    pres_file = st.file_uploader("Prescription CSV (optional)", type=["csv"], key="upload_pres")

# Process uploads defensively
if yield_file is not None:
    ydf = safe_read_csv(yield_file)
    st.session_state["uploads"]["yield"] = ydf
else:
    ydf = st.session_state["uploads"]["yield"]

if pres_file is not None:
    pdf = safe_read_csv(pres_file)
    st.session_state["uploads"]["prescriptions"] = pdf
else:
    pdf = st.session_state["uploads"]["prescriptions"]

# Zones: accept geojson or zipped shapefile
if zones_file is not None:
    try:
        # If zip — assume shapefile inside
        if zones_file.type == "application/zip" or (hasattr(zones_file, "name") and zones_file.name.endswith(".zip")):
            with open("/tmp/tmp_zones.zip", "wb") as f:
                f.write(zones_file.getbuffer())
            with zipfile.ZipFile("/tmp/tmp_zones.zip", "r") as z:
                z.extractall("/tmp/zones_shp")
            # find shapefile
            shp = None
            for root, dirs, files in os.walk("/tmp/zones_shp"):
                for file in files:
                    if file.endswith('.shp'):
                        shp = os.path.join(root, file)
                        break
                if shp:
                    break
            if shp:
                zgdf = gpd.read_file(shp)
            else:
                st.error("No .shp found inside uploaded zip.")
                zgdf = None
        else:
            zones_file.seek(0)
            zgdf = gpd.read_file(zones_file)
        st.session_state["uploads"]["zones"] = zgdf
    except Exception as e:
        st.error(f"Failed to load zones: {e}")
        st.session_state["uploads"]["zones"] = None
else:
    zgdf = st.session_state["uploads"]["zones"]

# -------------------------------
# Main layout: controls left, map center, summary right
# -------------------------------
left_col, map_col, right_col = st.columns([1, 2, 1])

with left_col:
    st.header("Controls")
    target_yield = st.number_input("Target Yield (bu/acre)", min_value=0.0, value=150.0, step=1.0)
    sell_price = st.number_input("Sell Price ($/bu)", min_value=0.0, value=4.50, step=0.01)
    fixed_costs = st.number_input("Fixed Costs ($/acre)", min_value=0.0, value=200.0, step=1.0)
    st.markdown("---")
    if st.button("Recompute Profit Summary"):
        st.session_state["profit_summary"] = {}

with map_col:
    st.header("Field Map")

    # create base folium map centered reasonably
    try:
        m = folium.Map(location=list(st.session_state["map_center"]), zoom_start=st.session_state["map_zoom"])
    except Exception:
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=5)

    # Add yield points if present
    if isinstance(ydf, pd.DataFrame) and not ydf.empty:
        g = df_to_gdf(ydf)
        if g is not None and not g.empty:
            # auto center on data
            try:
                bounds = g.total_bounds  # minx, miny, maxx, maxy
                center = ((bounds[1] + bounds[3]) / 2.0, (bounds[0] + bounds[2]) / 2.0)
                st.session_state["map_center"] = center
                st.session_state["map_zoom"] = 15
                m.location = center
                m.zoom_start = 15
            except Exception:
                pass

            # add simple circle markers
            for _, row in g.head(1000).iterrows():
                try:
                    folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=2, popup=str(row.get('yield', row.get('Yield', '')))).add_to(m)
                except Exception:
                    continue

    # Add zones if present
    if isinstance(zgdf, gpd.GeoDataFrame) and not getattr(zgdf, "empty", True):
        try:
            folium.GeoJson(zgdf.__geo_interface__, name="Zones").add_to(m)
        except Exception as e:
            st.warning(f"Failed to render zones on map: {e}")

    # Render map safely
    st_data = st_folium(m, width="100%", height=600)

with right_col:
    st.header("Profit Summary")

    # Basic profit calc: (target_yield * sell_price) - fixed_costs = gross profit per acre
    gross = target_yield * sell_price
    profit_per_acre = gross - fixed_costs

    st.metric("Target Yield (bu/acre)", f"{target_yield}")
    st.metric("Sell Price ($/bu)", f"${sell_price:.2f}")
    st.metric("Gross Revenue ($/acre)", f"${gross:.2f}")
    st.metric("Fixed Costs ($/acre)", f"${fixed_costs:.2f}")
    st.metric("Profit per Acre ($/acre)", f"${profit_per_acre:.2f}")

    st.markdown("---")
    st.subheader("Per-product cost table")
    # If prescriptions present, show simplistic table
    if isinstance(pdf, pd.DataFrame) and not pdf.empty:
        display_cols = pdf.columns.tolist()
        st.dataframe(pdf.head(200)[display_cols])
    else:
        st.info("No prescription / product CSV uploaded — upload to see per-product cost breakdown.")

# -------------------------------
# Utility: Export a simple backup zip of uploaded files (if any)
# -------------------------------
st.sidebar.markdown("---")
with st.sidebar.expander("Export / Backup", expanded=False):
    if st.button("Create backup zip of current uploads"):
        zname = "farm_roi_backups.zip"
        with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as zf:
            # yield
            if isinstance(ydf, pd.DataFrame) and not ydf.empty:
                buf = io.BytesIO()
                ydf.to_csv(buf, index=False)
                zf.writestr("yield_backup.csv", buf.getvalue())
            # prescriptions
            if isinstance(pdf, pd.DataFrame) and not pdf.empty:
                buf = io.BytesIO()
                pdf.to_csv(buf, index=False)
                zf.writestr("prescriptions_backup.csv", buf.getvalue())
            # zones -> geojson
            if isinstance(zgdf, gpd.GeoDataFrame) and not getattr(zgdf, "empty", True):
                try:
                    geojson = zgdf.to_json()
                    zf.writestr("zones_backup.geojson", geojson)
                except Exception:
                    pass
        with open(zname, "rb") as f:
            st.download_button("Download backup zip", data=f, file_name=zname, mime="application/zip")

# -------------------------------
# Footer: simple help
# -------------------------------
st.markdown("---")
st.caption("This is the BACKUP DEFAULT baseline version. Keep a copy before making iterative changes. If you want a more featureful restore (zone-based ROI, stacked inputs, legends, auto-coloring, or a drop-in replacement for your V4 code), tell me which sections to re-add and I'll splice them in.")
