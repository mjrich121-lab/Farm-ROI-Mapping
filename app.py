import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import geopandas as gpd
import zipfile
import os

st.set_page_config(page_title="Farm ROI Tool", layout="wide")
st.title("Farm Profit Mapping Tool")

# ---------------- Yield CSV Upload ----------------
st.header("Yield Map Upload")
uploaded_files = st.file_uploader("Upload Yield Map CSV(s)", type="csv", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"Field: {file.name}")
        df = pd.read_csv(file)

        # ---------------- Field Inputs ----------------
        with st.expander("Field Inputs & Expenses", expanded=True):
            acres = st.number_input("Field Acres", min_value=1, value=100, key=f"{file.name}_acres")
            sell_price = st.number_input("Sell Price ($/bu)", min_value=0.0, value=5.0, key=f"{file.name}_sell_price")
            yield_ac = st.number_input("Yield (bu/ac)", min_value=0.0, value=200.0, key=f"{file.name}_yield")

            st.markdown("### Expenses per Acre ($)")
            chemicals = st.number_input("Chemicals", value=25.0, key=f"{file.name}_chemicals")
            insurance = st.number_input("Crop & Hail Insurance", value=50.0, key=f"{file.name}_insurance")
            insecticide = st.number_input("Insecticide/Fungicide", value=25.0, key=f"{file.name}_insecticide")
            fertilizer = st.number_input("Fertilizer", value=100.0, key=f"{file.name}_fertilizer")
            machinery = st.number_input("Machinery", value=150.0, key=f"{file.name}_machinery")
            seed = st.number_input("Seed", value=100.0, key=f"{file.name}_seed")
            cost_of_living = st.number_input("Cost of Living", value=25.0, key=f"{file.name}_cost_of_living")
            extra_fuel = st.number_input("Extra Fuel for Corn", value=20.0, key=f"{file.name}_extra_fuel")
            extra_interest = st.number_input("Extra Interest", value=20.0, key=f"{file.name}_extra_interest")
            truck_fuel = st.number_input("Truck Fuel", value=5.0, key=f"{file.name}_truck_fuel")
            labor = st.number_input("Labor", value=15.0, key=f"{file.name}_labor")
            cash_rent = st.number_input("Cash Rent", value=100.0, key=f"{file.name}_cash_rent")

        # ---------------- ROI Calculations ----------------
        expense_inputs = [chemicals, insurance, insecticide, fertilizer, machinery,
                          seed, cost_of_living, extra_fuel, extra_interest, truck_fuel,
                          labor, cash_rent]
        expenses_per_acre = sum(expense_inputs)
        revenue_per_acre = yield_ac * sell_price
        net_profit_per_acre = revenue_per_acre - expenses_per_acre
        roi_percent = (net_profit_per_acre / expenses_per_acre * 100) if expenses_per_acre else 0

        report = pd.DataFrame({
            "Metric": ["Acres", "Sell Price ($/bu)", "Yield (bu/ac)",
                       "Revenue per Acre ($)", "Expenses per Acre ($)",
                       "Net Profit per Acre ($)", "ROI (%)"],
            "Value": [acres, sell_price, yield_ac,
                      round(revenue_per_acre, 2), round(expenses_per_acre, 2),
                      round(net_profit_per_acre, 2), round(roi_percent, 2)]
        })
        st.table(report)

        # ---------------- Map Creation ----------------
        if "Latitude" in df.columns and "Longitude" in df.columns and "Yield" in df.columns:
            df["Revenue_per_acre"] = df["Yield"] * sell_price
            df["Expenses_per_acre"] = expenses_per_acre
            df["NetProfit_per_acre"] = df["Revenue_per_acre"] - df["Expenses_per_acre"]

            m = folium.Map(
                location=[df["Latitude"].mean(), df["Longitude"].mean()],
                zoom_start=15,
                tiles=None
            )

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

            # --- Heatmap function with dynamic legends ---
            def create_heatmap_layer(values, label, cmap_name, opacity, show, order_priority):
                grid_x, grid_y = np.mgrid[
                    df["Longitude"].min():df["Longitude"].max():200j,
                    df["Latitude"].min():df["Latitude"].max():200j
                ]
                grid_z = griddata(
                    (df["Longitude"], df["Latitude"]),
                    values,
                    (grid_x, grid_y),
                    method="linear"
                )
                vmin, vmax = np.nanmin(values), np.nanmax(values)
                cmap = plt.cm.get_cmap(cmap_name)
                rgba_img = cmap((grid_z - vmin) / (vmax - vmin))
                rgba_img = np.nan_to_num(rgba_img, nan=0.0)

                folium.raster_layers.ImageOverlay(
                    image=np.uint8(rgba_img * 255),
                    bounds=[[df["Latitude"].min(), df["Longitude"].min()],
                            [df["Latitude"].max(), df["Longitude"].max()]],
                    opacity=opacity,
                    name=label,
                    show=show
                ).add_to(m)

                # Legend with dynamic range
                colormap = cm.LinearColormap(colors=["red", "yellow", "green"], vmin=vmin, vmax=vmax)
                if "Profit" in label:
                    colormap.caption = f"🔥 {label} ({round(vmin,2)} – {round(vmax,2)})"
                else:
                    colormap.caption = f"{label} ({round(vmin,2)} – {round(vmax,2)})"
                colormap.add_to(m)

            # Profit heatmap (MAIN priority, ON by default, strong opacity)
            create_heatmap_layer(
                df["NetProfit_per_acre"],
                "Net Profit ($/ac)",
                cmap_name="RdYlGn",
                opacity=0.8,
                show=True,
                order_priority=1
            )

            # Yield heatmap (context, OFF by default, lighter)
            create_heatmap_layer(
                df["Yield"],
                "Yield (bu/ac)",
                cmap_name="YlGnBu",
                opacity=0.35,
                show=False,
                order_priority=2
            )
    
    # ---------------- Zone Upload ----------------
    st.markdown("---")
    st.header("Zone Map Upload")
    zone_file = st.file_uploader(
        "Upload Zone Map (GeoJSON or zipped Shapefile)",
        type=["geojson", "json", "zip"],
        key=f"{file.name}_zone"
    )

    if zone_file is not None:
        zones_gdf = None

        # --- Load GeoJSON or Shapefile ---
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

            # --- Cleanup temporary files ---
            os.remove("temp.zip")
            import shutil
            shutil.rmtree("temp_shp", ignore_errors=True)

        else:
            st.error("Unsupported file format. Please upload a GeoJSON or zipped Shapefile.")

        # --- Display Zones on Map ---
        if zones_gdf is not None:
            st.success("Zone map loaded successfully")
            zone_layer = folium.FeatureGroup(name="Zones", show=True)

            # Pick the correct zone column
            zone_col = None
            for candidate in ["Zone", "zone", "ZONE", "Name", "name"]:
                if candidate in zones_gdf.columns:
                    zone_col = candidate
                    break
            if zone_col is None:
                zones_gdf["ZoneIndex"] = range(1, len(zones_gdf) + 1)
                zone_col = "ZoneIndex"

            # Static zone color scheme
            static_zone_colors = {
                1: "#FF0000",  # Red
                2: "#FF8000",  # Orange
                3: "#FFFF00",  # Yellow
                4: "#80FF00",  # Light Green
                5: "#008000"   # Dark Green
            }

            # Add polygons
            for _, row in zones_gdf.iterrows():
                try:
                    zone_value = int(row[zone_col])
                except:
                    zone_value = row[zone_col]  # fallback if not numeric

                zone_color = static_zone_colors.get(zone_value, "#0000FF")  # fallback blue

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

            # --- Static Zone Legend (bottom-right, collapsible) ---
            zone_legend_html = """
            <div id="zone-legend" style="position: fixed; 
                        bottom: 30px; right: 30px; width: 180px; 
                        background-color: white; z-index:9999; 
                        font-size:14px; border:2px solid grey; border-radius:5px;
                        padding: 10px;">
            <b>Zone Legend</b>
            <button onclick="var x=document.getElementById('zone-legend-body'); 
                             if(x.style.display==='none'){x.style.display='block';this.innerText='-';} 
                             else{x.style.display='none';this.innerText='+';}" 
                    style="float:right;">-</button>
            <div id="zone-legend-body">
            <i style="background:#FF0000;width:20px;height:10px;display:inline-block;"></i> Zone 1<br>
            <i style="background:#FF8000;width:20px;height:10px;display:inline-block;"></i> Zone 2<br>
            <i style="background:#FFFF00;width:20px;height:10px;display:inline-block;"></i> Zone 3<br>
            <i style="background:#80FF00;width:20px;height:10px;display:inline-block;"></i> Zone 4<br>
            <i style="background:#008000;width:20px;height:10px;display:inline-block;"></i> Zone 5<br>
            </div>
            </div>
            """
            m.get_root().html.add_child(folium.Element(zone_legend_html))

            folium.LayerControl(collapsed=False).add_to(m)
            st_folium(m, width=700, height=500)

        else:
            st.warning("CSV must include Latitude, Longitude, and Yield columns.")
