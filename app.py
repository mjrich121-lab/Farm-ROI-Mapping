# =========================================================
# Farm ROI Tool V3
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

st.set_page_config(page_title="Farm ROI Tool V3", layout="wide")
st.title("Farm Profit Mapping Tool V3")

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
# 3. PRESCRIPTION MAP UPLOADS (multi-product supported)
# =========================================================
st.header("Prescription Map Uploads")
fert_file = st.file_uploader("Upload Fertilizer Prescription Map", type=["csv"], key="fert")
seed_file = st.file_uploader("Upload Seed Prescription Map", type=["csv"], key="seed")

def process_prescription(file):
    if file is None:
        return pd.DataFrame()
    df = pd.read_csv(file)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # We’ll accept a few schema variations
    # Must have product and acres_applied
    if "product" not in df.columns or "acres" not in df.columns:
        st.error("CSV must include columns: Product, Acres, [CostTotal or Price/Unit + Units]")
        return pd.DataFrame()

    # Derive total cost if not provided
    if "costtotal" in df.columns:
        df["cost_total"] = df["costtotal"]
    elif "price_per_unit" in df.columns and "units" in df.columns:
        df["cost_total"] = df["price_per_unit"] * df["units"]
    else:
        st.error("CSV must have either CostTotal OR Price_per_unit + Units")
        return pd.DataFrame()

    # Group by product
    grouped = df.groupby("product", as_index=False).agg(
        Acres=("acres", "sum"),
        CostTotal=("cost_total", "sum")
    )
    grouped["CostPerAcre"] = grouped["CostTotal"] / grouped["Acres"]
    return grouped

fert_products = process_prescription(fert_file)
seed_products = process_prescription(seed_file)

if not fert_products.empty:
    st.success("Fertilizer prescription uploaded and processed")
if not seed_products.empty:
    st.success("Seed prescription uploaded and processed")

# =========================================================
# 4. EXPENSE INPUTS (PER ACRE $)
# =========================================================
st.header("Expense Inputs (Per Acre $)")

cols = st.columns(6)
with cols[0]:
    st.markdown("**Sell Price ($/bu)**")
    sell_price = st.number_input("sell", min_value=0.0, value=0.0, step=0.1, label_visibility="collapsed")
with cols[1]:
    st.markdown("**Chemicals ($/ac)**")
    chemicals = st.number_input("chem", min_value=0.0, value=0.0, step=0.1, label_visibility="collapsed")
with cols[2]:
    st.markdown("**Insurance ($/ac)**")
    insurance = st.number_input("ins", min_value=0.0, value=0.0, step=0.1, label_visibility="collapsed")
with cols[3]:
    st.markdown("**Insect/Fungicide ($/ac)**")
    insecticide = st.number_input("insect", min_value=0.0, value=0.0, step=0.1, label_visibility="collapsed")
with cols[4]:
    st.markdown("**Fertilizer (Flat $/ac)**")
    fertilizer = st.number_input("fert", min_value=0.0, value=0.0, step=0.1, label_visibility="collapsed")
with cols[5]:
    st.markdown("**Seed (Flat $/ac)**")
    seed = st.number_input("seed", min_value=0.0, value=0.0, step=0.1, label_visibility="collapsed")

cols2 = st.columns(6)
with cols2[0]:
    st.markdown("**Machinery ($/ac)**")
    machinery = st.number_input("mach", min_value=0.0, value=0.0, step=0.1, label_visibility="collapsed")
with cols2[1]:
    st.markdown("**Labor ($/ac)**")
    labor = st.number_input("labor", min_value=0.0, value=0.0, step=0.1, label_visibility="collapsed")
with cols2[2]:
    st.markdown("**Cost of Living ($/ac)**")
    cost_of_living = st.number_input("col", min_value=0.0, value=0.0, step=0.1, label_visibility="collapsed")
with cols2[3]:
    st.markdown("**Extra Fuel ($/ac)**")
    extra_fuel = st.number_input("fuel", min_value=0.0, value=0.0, step=0.1, label_visibility="collapsed")
with cols2[4]:
    st.markdown("**Extra Interest ($/ac)**")
    extra_interest = st.number_input("int", min_value=0.0, value=0.0, step=0.1, label_visibility="collapsed")
with cols2[5]:
    st.markdown("**Truck Fuel ($/ac)**")
    truck_fuel = st.number_input("truck", min_value=0.0, value=0.0, step=0.1, label_visibility="collapsed")

cols3 = st.columns(6)
with cols3[0]:
    st.markdown("**Cash Rent ($/ac)**")
    cash_rent = st.number_input("rent", min_value=0.0, value=0.0, step=0.1, label_visibility="collapsed")

# Collect into dict
expenses = {
    "Chemicals": chemicals,
    "Insurance": insurance,
    "Insecticide/Fungicide": insecticide,
    "Fertilizer (Flat)": fertilizer,
    "Seed (Flat)": seed,
    "Machinery": machinery,
    "Labor": labor,
    "Cost of Living": cost_of_living,
    "Extra Fuel": extra_fuel,
    "Extra Interest": extra_interest,
    "Truck Fuel": truck_fuel,
    "Cash Rent": cash_rent
}
base_expenses_per_acre = sum(expenses.values())

# =========================================================
# 5. BASE MAP
# =========================================================
m = folium.Map(location=[40, -95], zoom_start=4, tiles=None)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Esri Satellite", overlay=False, control=False
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Labels", overlay=True, control=False
).add_to(m)

# =========================================================
# 6. ZONES
# =========================================================
if zones_gdf is not None:
    zone_layer = folium.FeatureGroup(name="Zones", show=True)
    static_zone_colors = {1: "#FF0000", 2: "#FF8000", 3: "#FFFF00", 4: "#80FF00", 5: "#008000"}
    zone_col = None
    for candidate in ["Zone","zone","ZONE","Name","name"]:
        if candidate in zones_gdf.columns:
            zone_col = candidate
            break
    if zone_col is None:
        zones_gdf["ZoneIndex"] = range(1, len(zones_gdf)+1)
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
            style_function=lambda x, c=zone_color: {"fillOpacity":0.3,"color":c,"weight":3},
            tooltip=f"Zone: {zone_value}"
        ).add_to(zone_layer)
    zone_layer.add_to(m)

# =========================================================
# 8. DISPLAY MAP
# =========================================================
folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width=900, height=600)
# =========================================================
# 7. YIELD + PROFIT
# =========================================================
df = None
if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        if {"Latitude","Longitude","Yield"}.issubset(df.columns):
            # Auto-zoom map
            m.location = [df["Latitude"].mean(), df["Longitude"].mean()]
            m.zoom_start = 15

            # Revenue per acre
            df["Revenue_per_acre"] = df["Yield"] * sell_price

            # Fertilizer + Seed costs (safe defaults if not uploaded)
            fert_costs = fert_products["CostPerAcre"].sum() if not fert_products.empty else 0
            seed_costs = seed_products["CostPerAcre"].sum() if not seed_products.empty else 0

            # Net profit per acre
            df["NetProfit_per_acre"] = (
                df["Revenue_per_acre"]
                - base_expenses_per_acre
                - fert_costs
                - seed_costs
            )

            # Profit Heatmap overlay
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
                opacity=0.6, name="Net Profit ($/ac)", show=True
            ).add_to(m)
# =========================================================
# 9. PROFIT SUMMARY
# =========================================================
st.header("Profit Summary")

if df is not None:
    revenue_per_acre = df["Revenue_per_acre"].mean()
    net_profit_per_acre = df["NetProfit_per_acre"].mean()

    st.subheader("Base Expenses (Flat $/ac)")
    st.write(pd.DataFrame(base_expenses.items(), columns=["Expense","$/ac"]))

    if not fert_products.empty:
        st.subheader("Fertilizer Costs (Per Product)")
        st.dataframe(fert_products[["Product","Acres","CostTotal","CostPerAcre"]])

    if not seed_products.empty:
        st.subheader("Seed Costs (Per Product)")
        st.dataframe(seed_products[["Product","Acres","CostTotal","CostPerAcre"]])

    st.subheader("Profit Metrics")
    summary = pd.DataFrame({
        "Metric":["Revenue ($/ac)","Expenses ($/ac)","Profit ($/ac)"],
        "Value":[round(revenue_per_acre,2),
                 round(base_expenses_per_acre,2),
                 round(net_profit_per_acre,2)]
    })

    def highlight_profit(val):
        if val > 0:
            return "color: green; font-weight: bold;"
        elif val < 0:
            return "color: red; font-weight: bold;"
        return ""
    st.dataframe(summary.style.applymap(highlight_profit, subset=["Value"]))
else:
    st.write("Upload a yield map to see profit results.")
