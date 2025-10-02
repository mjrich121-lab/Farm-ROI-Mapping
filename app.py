import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from scipy.interpolate import griddata
import geopandas as gpd
import zipfile
import os
import matplotlib.pyplot as plt
import shutil

st.set_page_config(page_title="Farm ROI Tool", layout="wide")
st.title("🌱 Farm Profit Mapping Tool")

# ==================================================
# SECTION 1: Zone Map Upload
# ==================================================
st.header("Zone Map Upload")
zone_file = st.file_uploader("Upload Zone Map (GeoJSON or zipped Shapefile)",
                             type=["geojson", "json", "zip"], key="zone")

zones_gdf = None
if zone_file is not None:
    try:
        if zone_file.name.endswith(".geojson") or zone_file.name.endswith(".json"):
            zones_gdf = gpd.read_file(zone_file)
        elif zone_file.name.endswith(".zip"):
            with open("temp.zip", "wb") as f:
                f.write(zone_file.getbuffer())
            with zipfile.ZipFile("temp.zip", "r") as zip_ref:
                zip_ref.extractall("temp_shp")
            for f_name in os.listdir("temp_shp"):
                if f_name.endswith(".shp"):
                    shp_path = os.path.join("temp_shp", f_name)
                    zones_gdf = gpd.read_file(shp_path)
                    break
            os.remove("temp.zip")
            shutil.rmtree("temp_shp", ignore_errors=True)
        if zones_gdf is not None:
            st.success("✅ Zone map loaded successfully")
    except Exception as e:
        st.error(f"Error loading zone file: {e}")

# ==================================================
# SECTION 2: Yield Map Upload
# ==================================================
st.header("Yield Map Upload")
uploaded_files = st.file_uploader("Upload Yield Map CSV(s)",
                                  type="csv", accept_multiple_files=True)

# ==================================================
# SECTION 3: Expense Inputs (concise horizontal table)
# ==================================================
st.header("Expense Inputs (Per Acre)")

cols = st.columns(4)  # 4 columns to spread inputs horizontally

with cols[0]:
    sell_price = st.number_input("Sell Price ($/bu)", min_value=0.0, value=0.0, step=0.1)
    chemicals = st.number_input("Chemicals", value=0.0)
    insurance = st.number_input("Crop & Hail Insurance", value=0.0)

with cols[1]:
    insecticide = st.number_input("Insecticide/Fungicide", value=0.0)
    fertilizer = st.number_input("Fertilizer", value=0.0)
    machinery = st.number_input("Machinery", value=0.0)

with cols[2]:
    seed = st.number_input("Seed", value=0.0)
    cost_of_living = st.number_input("Cost of Living", value=0.0)
    extra_fuel = st.number_input("Extra Fuel for Corn", value=0.0)

with cols[3]:
    extra_interest = st.number_input("Extra Interest", value=0.0)
    truck_fuel = st.number_input("Truck Fuel", value=0.0)
    labor = st.number_input("Labor", value=0.0)
    cash_rent = st.number_input("Cash Rent", value=0.0)

# Sum up expenses
expenses_per_acre = sum([
    chemicals, insurance, insecticide, fertilizer, machinery,
    seed, cost_of_living, extra_fuel, extra_interest,
    truck_fuel, labor, cash_rent
])

# ==================================================
# SECTION 4: Map Setup (always visible)
# ==================================================
center = [40, -95]   # Default center on US
zoom = 5

# If zones uploaded, adjust center/zoom
if zones_gdf is not None:
    bounds = zones_gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    zoom = 14

m = folium.Map(location=center, zoom_start=zoom, tiles=None)

# Base layers
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Esri Satellite", overlay=False, control=False
).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Labels", overlay=True, control=False
).add_to(m)

# ==================================================
# SECTION 5: Zones (optional)
# ==================================================
if zones_gdf is not None:
    zone_layer = folium.FeatureGroup(name="Zones", show=True)
    static_zone_colors = {1: "#FF0000", 2: "#FF8000", 3: "#FFFF00", 4: "#80FF00", 5: "#008000"}
    zone_col = None
    for candidate in ["Zone", "zone", "ZONE", "Name", "name"]:
        if candidate in zones_gdf.columns:
            zone_col = candidate
            break
    if zone_col is None:
        zones_gdf["ZoneIndex"] = range(1, len(zones_gdf) + 1)
        zone_col = "ZoneIndex"

    for _, row in zones_gdf.iterrows():
        try:
            zone_value = int(row[zone_col])
        except:
            zone_value = row[zone_col]
        zone_color = static_zone_colors.get(zone_value, "#0000FF")
        folium.GeoJson(
            row["geometry"],
            name=f"Zone {zone_value}",
            style_function=lambda x, c=zone_color: {
                "fillOpacity": 0.3,
                "color": c,
                "weight": 3
            },
            tooltip=f"Zone: {zone_value}"
        ).add_to(zone_layer)
    zone_layer.add_to(m)

# ==================================================
# SECTION 6: Yield / Profit Heatmaps (if yield file exists)
# ==================================================
avg_yield = 0
revenue_per_acre = 0
net_profit_per_acre = 0
roi_percent = 0

if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        if "Latitude" in df.columns and "Longitude" in df.columns and "Yield" in df.columns:
            df["Revenue"] = df["Yield"] * sell_price
            df["NetProfit_per_acre"] = df["Revenue"] - expenses_per_acre

            avg_yield = df["Yield"].mean()
            revenue_per_acre = avg_yield * sell_price
            net_profit_per_acre = revenue_per_acre - expenses_per_acre
            roi_percent = (net_profit_per_acre / expenses_per_acre * 100) if expenses_per_acre > 0 else 0

            # Profit map
            grid_x, grid_y = np.mgrid[
                df["Longitude"].min():df["Longitude"].max():200j,
                df["Latitude"].min():df["Latitude"].max():200j
            ]
            grid_z_profit = griddata(
                (df["Longitude"], df["Latitude"]),
                df["NetProfit_per_acre"],
                (grid_x, grid_y),
                method="linear"
            )
            vmin, vmax = np.nanmin(df["NetProfit_per_acre"]), np.nanmax(df["NetProfit_per_acre"])
            cmap = plt.cm.get_cmap("RdYlGn")
            rgba_img = cmap((grid_z_profit - vmin) / (vmax - vmin))
            folium.raster_layers.ImageOverlay(
                image=np.uint8(rgba_img * 255),
                bounds=[[df["Latitude"].min(), df["Longitude"].min()],
                        [df["Latitude"].max(), df["Longitude"].max()]],
                opacity=0.7,
                name="Net Profit ($/ac)",
                show=True
            ).add_to(m)
# ==================================================
# SECTION 7: Render Map + Summary Table
# ==================================================
folium.LayerControl(collapsed=False).add_to(m)
st_map = st_folium(m, width=900, height=600)

# Summary calculations (already computed above)
summary_df = pd.DataFrame({
    "Metric": ["Revenue/acre ($)", "Expenses/acre ($)", "Net Profit/acre ($)", "ROI (%)"],
    "Profit": [
        round(revenue_per_acre, 2),
        round(expenses_per_acre, 2),
        round(net_profit_per_acre, 2),
        round(roi_percent, 2)
    ]
})

st.subheader("Profitability Summary")

# Apply red/green styling to Net Profit row
def highlight_profit(val, metric):
    if metric == "Net Profit/acre ($)":
        color = "green" if val >= 0 else "red"
        return f"color: {color}; font-weight: bold"
    return ""

styled_summary = summary_df.style.apply(
    lambda row: [highlight_profit(v, row["Metric"]) for v in row], axis=1
)

st.table(styled_summary)

