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
# SECTION 3: Expense Inputs (always visible, AFTER yield upload)
# ==================================================
st.header("Expense Inputs (Per Acre)")
sell_price = st.number_input("Sell Price ($/bu)", min_value=0.0, value=0.0, step=0.1)

st.markdown("### Expenses per Acre ($)")
chemicals = st.number_input("Chemicals", value=0.0)
insurance = st.number_input("Crop & Hail Insurance", value=0.0)
insecticide = st.number_input("Insecticide/Fungicide", value=0.0)
fertilizer = st.number_input("Fertilizer", value=0.0)
machinery = st.number_input("Machinery", value=0.0)
seed = st.number_input("Seed", value=0.0)
cost_of_living = st.number_input("Cost of Living", value=0.0)
extra_fuel = st.number_input("Extra Fuel for Corn", value=0.0)
extra_interest = st.number_input("Extra Interest", value=0.0)
truck_fuel = st.number_input("Truck Fuel", value=0.0)
labor = st.number_input("Labor", value=0.0)
cash_rent = st.number_input("Cash Rent", value=0.0)

expenses_per_acre = sum([
    chemicals, insurance, insecticide, fertilizer, machinery,
    seed, cost_of_living, extra_fuel, extra_interest,
    truck_fuel, labor, cash_rent
])

# ==================================================
# SECTION 4: Map Setup
# ==================================================
if uploaded_files or zones_gdf is not None:
    center = [40, -95]
    zoom = 5

    if zones_gdf is not None:
        bounds = zones_gdf.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        zoom = 14

    m = folium.Map(location=center, zoom_start=zoom, tiles=None)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Satellite", overlay=False, control=False
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Labels", overlay=True, control=False
    ).add_to(m)

    # ==================================================
    # SECTION 5: Zones
    # ==================================================
    if zones_gdf is not None:
        zone_layer = folium.FeatureGroup(name="Zones", show=True)
        static_zone_colors = {
            1: "#FF0000",
            2: "#FF8000",
            3: "#FFFF00",
            4: "#80FF00",
            5: "#008000"
        }

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

        zone_legend_html = """
        <div style="position: fixed; bottom: 30px; right: 30px; width: 180px; 
                    background-color: white; z-index:9999; 
                    font-size:14px; border:2px solid grey; border-radius:5px;
                    padding: 10px;">
        <b>Zone Legend</b><br>
        <i style="background:#FF0000;width:20px;height:10px;display:inline-block;"></i> Zone 1<br>
        <i style="background:#FF8000;width:20px;height:10px;display:inline-block;"></i> Zone 2<br>
        <i style="background:#FFFF00;width:20px;height:10px;display:inline-block;"></i> Zone 3<br>
        <i style="background:#80FF00;width:20px;height:10px;display:inline-block;"></i> Zone 4<br>
        <i style="background:#008000;width:20px;height:10px;display:inline-block;"></i> Zone 5<br>
        </div>
        """
        m.get_root().html.add_child(folium.Element(zone_legend_html))

    # ==================================================
    # SECTION 6: Yield / Profit Heatmaps
    # ==================================================
    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file)
            if "Latitude" in df.columns and "Longitude" in df.columns and "Yield" in df.columns:
                df["Revenue"] = df["Yield"] * sell_price
                df["NetProfit_per_acre"] = df["Revenue"] - expenses_per_acre

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
                grid_z_yield = griddata(
                    (df["Longitude"], df["Latitude"]),
                    df["Yield"],
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

                profit_legend_html = f"""
                <div style="position: fixed; bottom: 30px; left: 30px; width: 200px; 
                            background-color: white; z-index:9999; 
                            font-size:14px; border:2px solid grey; border-radius:5px;
                            padding: 10px;">
                <b>Net Profit ($/ac)</b><br>
                Low: {vmin:.1f} &nbsp;&nbsp; High: {vmax:.1f}
                </div>
                """
                m.get_root().html.add_child(folium.Element(profit_legend_html))

                vmin_y, vmax_y = np.nanmin(df["Yield"]), np.nanmax(df["Yield"])
                rgba_yield = cmap((grid_z_yield - vmin_y) / (vmax_y - vmin_y))
                folium.raster_layers.ImageOverlay(
                    image=np.uint8(rgba_yield * 255),
                    bounds=[[df["Latitude"].min(), df["Longitude"].min()],
                            [df["Latitude"].max(), df["Longitude"].max()]],
                    opacity=0.5,
                    name="Yield (bu/ac)",
                    show=False
                ).add_to(m)

                yield_legend_html = f"""
                <div style="position: fixed; bottom: 100px; left: 30px; width: 200px; 
                            background-color: white; z-index:9999; 
                            font-size:14px; border:2px solid grey; border-radius:5px;
                            padding: 10px;">
                <b>Yield (bu/ac)</b><br>
                Low: {vmin_y:.1f} &nbsp;&nbsp; High: {vmax_y:.1f}
                </div>
                """
                m.get_root().html.add_child(folium.Element(yield_legend_html))

    # ==================================================
    # SECTION 7: Render
    # ==================================================
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=900, height=600)

