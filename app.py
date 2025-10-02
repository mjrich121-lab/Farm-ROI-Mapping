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

# ---------------- Zone Map Upload ----------------
st.header("Zone Map Upload")
zone_file = st.file_uploader("Upload Zone Map (GeoJSON or zipped Shapefile)",
                             type=["geojson", "json", "zip"], key="zone")

zones_gdf = None
if zone_file is not None:
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
        import shutil
        shutil.rmtree("temp_shp", ignore_errors=True)

    if zones_gdf is not None:
        st.success("Zone map loaded successfully")

# ---------------- Yield Map Upload ----------------
st.header("Yield Map Upload")
uploaded_files = st.file_uploader("Upload Yield Map CSV(s)", type="csv", accept_multiple_files=True)

if uploaded_files or zones_gdf is not None:
    m = folium.Map(location=[40, -95], zoom_start=5, tiles=None)

    # Satellite base
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=False
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Labels",
        overlay=True,
        control=False
    ).add_to(m)

    # --- Zones ---
    if zones_gdf is not None:
        zone_layer = folium.FeatureGroup(name="Zones", show=True)
        static_zone_colors = {
            1: "#FF0000",  # Red
            2: "#FF8000",  # Orange
            3: "#FFFF00",  # Yellow
            4: "#80FF00",  # Light Green
            5: "#008000"   # Dark Green
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

        # Zone Legend (bottom-right)
        zone_legend_html = """
        <div style="position: fixed; 
                    bottom: 30px; right: 30px; width: 180px; 
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

    # --- Yield ---
    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file)
            if "Latitude" in df.columns and "Longitude" in df.columns and "Yield" in df.columns:
                df["NetProfit_per_acre"] = df["Yield"] * 5  # placeholder until expenses linked

                # Heatmap
                grid_x, grid_y = np.mgrid[
                    df["Longitude"].min():df["Longitude"].max():200j,
                    df["Latitude"].min():df["Latitude"].max():200j
                ]
                grid_z = griddata(
                    (df["Longitude"], df["Latitude"]),
                    df["NetProfit_per_acre"],
                    (grid_x, grid_y),
                    method="linear"
                )
                vmin, vmax = np.nanmin(df["NetProfit_per_acre"]), np.nanmax(df["NetProfit_per_acre"])
                cmap = plt.cm.get_cmap("RdYlGn")
                rgba_img = cmap((grid_z - vmin) / (vmax - vmin))
                rgba_img = np.nan_to_num(rgba_img, nan=0.0)

                folium.raster_layers.ImageOverlay(
                    image=np.uint8(rgba_img * 255),
                    bounds=[[df["Latitude"].min(), df["Longitude"].min()],
                            [df["Latitude"].max(), df["Longitude"].max()]],
                    opacity=0.6,
                    name="Net Profit ($/ac)",
                    show=True
                ).add_to(m)

    # Controls
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=700, height=500)
