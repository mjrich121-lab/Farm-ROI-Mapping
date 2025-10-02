# =========================================================
# Farm Profit Mapping Tool (Base Framework v1)
# =========================================================
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

st.set_page_config(page_title="Farm ROI Tool", layout="wide")
st.title("Farm Profit Mapping Tool")

# =========================================================
# 1. ZONE MAP UPLOAD
# =========================================================
st.header("Zone Map Upload")
zone_file = st.file_uploader("Upload Zone Map (GeoJSON or zipped Shapefile)",
                             type=["geojson", "json", "zip"], key="zone")

zones_gdf = None
if zone_file is not None:
    if zone_file.name.endswith((".geojson", ".json")):
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
        import shutil
        shutil.rmtree("temp_shp", ignore_errors=True)
    if zones_gdf is not None:
        st.success("Zone map loaded successfully")

# =========================================================
# 2. YIELD MAP UPLOAD
# =========================================================
st.header("Yield Map Upload")
uploaded_files = st.file_uploader("Upload Yield Map CSV(s)", type="csv", accept_multiple_files=True)

# =========================================================
# 3. EXPENSE INPUTS (ALWAYS SHOW)
# =========================================================
st.header("Expense Inputs (Per Acre $)")

cols = st.columns(6)
sell_price = cols[0].number_input("Sell Price ($/bu)", min_value=0.0, value=0.0, step=0.1)
chemicals = cols[1].number_input("Chemicals ($/ac)", min_value=0.0, value=0.0, step=0.1)
insurance = cols[2].number_input("Insurance ($/ac)", min_value=0.0, value=0.0, step=0.1)
insecticide = cols[3].number_input("Insect/Fungicide ($/ac)", min_value=0.0, value=0.0, step=0.1)
fertilizer = cols[4].number_input("Fertilizer ($/ac)", min_value=0.0, value=0.0, step=0.1)
machinery = cols[5].number_input("Machinery ($/ac)", min_value=0.0, value=0.0, step=0.1)

cols2 = st.columns(6)
seed = cols2[0].number_input("Seed ($/ac)", min_value=0.0, value=0.0, step=0.1)
cost_of_living = cols2[1].number_input("Cost of Living ($/ac)", min_value=0.0, value=0.0, step=0.1)
extra_fuel = cols2[2].number_input("Extra Fuel ($/ac)", min_value=0.0, value=0.0, step=0.1)
extra_interest = cols2[3].number_input("Extra Interest ($/ac)", min_value=0.0, value=0.0, step=0.1)
truck_fuel = cols2[4].number_input("Truck Fuel ($/ac)", min_value=0.0, value=0.0, step=0.1)
labor = cols2[5].number_input("Labor ($/ac)", min_value=0.0, value=0.0, step=0.1)

cols3 = st.columns(6)
cash_rent = cols3[0].number_input("Cash Rent ($/ac)", min_value=0.0, value=0.0, step=0.1)

# Store expenses
expenses = {
    "Chemicals": chemicals,
    "Insurance": insurance,
    "Insecticide/Fungicide": insecticide,
    "Fertilizer": fertilizer,
    "Machinery": machinery,
    "Seed": seed,
    "Cost of Living": cost_of_living,
    "Extra Fuel": extra_fuel,
    "Extra Interest": extra_interest,
    "Truck Fuel": truck_fuel,
    "Labor": labor,
    "Cash Rent": cash_rent
}
expenses_per_acre = sum(expenses.values())

# =========================================================
# 4. CREATE MAP (ALWAYS SHOW BASE MAP)
# =========================================================
m = folium.Map(location=[40, -95], zoom_start=4, tiles=None)

# Base layers
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Esri Satellite", overlay=False, control=False
).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Labels", overlay=True, control=False
).add_to(m)

# =========================================================
# 5. ZONES (IF UPLOADED)
# =========================================================
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
            style_function=lambda x, c=zone_color: {"fillOpacity": 0.3, "color": c, "weight": 3},
            tooltip=f"Zone: {zone_value}"
        ).add_to(zone_layer)
    zone_layer.add_to(m)

# =========================================================
# 6. YIELD + PROFIT (IF FILE UPLOADED)
# =========================================================
df = None
if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        if "Latitude" in df.columns and "Longitude" in df.columns and "Yield" in df.columns:
            # Auto-zoom
            m.location = [df["Latitude"].mean(), df["Longitude"].mean()]
            m.zoom_start = 15

            # Revenue & Profit
            df["Revenue_per_acre"] = df["Yield"] * sell_price
            df["NetProfit_per_acre"] = df["Revenue_per_acre"] - expenses_per_acre

            # Profit Heatmap
            grid_x, grid_y = np.mgrid[
                df["Longitude"].min():df["Longitude"].max():200j,
                df["Latitude"].min():df["Latitude"].max():200j
            ]
            grid_z = griddata(
                (df["Longitude"], df["Latitude"]),
                df["NetProfit_per_acre"], (grid_x, grid_y), method="linear"
            )
            vmin, vmax = np.nanmin(df["NetProfit_per_acre"]), np.nanmax(df["NetProfit_per_acre"])
            cmap = plt.cm.get_cmap("RdYlGn")
            rgba_img = cmap((grid_z - vmin) / (vmax - vmin))
            rgba_img = np.nan_to_num(rgba_img, nan=0.0)
            folium.raster_layers.ImageOverlay(
                image=np.uint8(rgba_img * 255),
                bounds=[[df["Latitude"].min(), df["Longitude"].min()],
                        [df["Latitude"].max(), df["Longitude"].max()]],
                opacity=0.6, name="Net Profit ($/ac)", show=True
            ).add_to(m)

# =========================================================
# 7. DISPLAY MAP
# =========================================================
folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width=800, height=600)

# =========================================================
# 8. SUMMARY TABLE (ALWAYS SHOW BELOW MAP)
# =========================================================
st.header("Summary")
if df is not None:
    revenue_per_acre = df["Revenue_per_acre"].mean()
    net_profit_per_acre = df["NetProfit_per_acre"].mean()
    summary = pd.DataFrame({
        "Metric": ["Revenue ($/ac)", "Expenses ($/ac)", "Profit ($/ac)"],
        "Profit": [round(revenue_per_acre, 2), round(expenses_per_acre, 2), round(net_profit_per_acre, 2)]
    })
    st.dataframe(summary.style.applymap(
        lambda v: "color: green; font-weight: bold;" if v > 0 else
                  "color: red; font-weight: bold;" if v < 0 else "",
        subset=["Profit"]
    ))
else:
    st.write("Upload a yield map to see summary results.")
