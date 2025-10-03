# =========================================================
# Farm ROI Tool V3 (with auto-zoom logic)
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
# HELPER: AUTO-ZOOM
# =========================================================
def auto_zoom_map(m, df=None, gdf=None):
    if df is not None and {"Latitude","Longitude"}.issubset(df.columns):
        m.location = [df["Latitude"].mean(), df["Longitude"].mean()]
        m.zoom_start = 15
    elif gdf is not None:
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return m

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
    df.columns = [c.strip().lower() for c in df.columns]

    if "product" not in df.columns or "acres" not in df.columns:
        return pd.DataFrame()

    if "costtotal" in df.columns:
        df["cost_total"] = df["costtotal"]
    elif "price_per_unit" in df.columns and "units" in df.columns:
        df["cost_total"] = df["price_per_unit"] * df["units"]
    else:
        return pd.DataFrame()

    grouped = df.groupby("product", as_index=False).agg(
        Acres=("acres", "sum"),
        CostTotal=("cost_total", "sum")
    )
    grouped["CostPerAcre"] = grouped["CostTotal"] / grouped["Acres"]
    return grouped

fert_products = process_prescription(fert_file)
seed_products = process_prescription(seed_file)

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
# 5. BASE MAP (rebuild clean each run but persist data state)
# =========================================================
def make_base_map():
    m = folium.Map(location=[40, -95], zoom_start=4, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Satellite", overlay=False, control=False
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Labels", overlay=True, control=False
    ).add_to(m)
    return m

# Initialize session state storage if not already there
if "zones_gdf" not in st.session_state:
    st.session_state["zones_gdf"] = None
if "yield_df" not in st.session_state:
    st.session_state["yield_df"] = None

# Always start with a fresh map each run
m = make_base_map()


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

    # Auto-zoom if no yield yet
    if not uploaded_files:
        m = auto_zoom_map(m, gdf=zones_gdf)
# =========================================================
# 7. YIELD + PROFIT (smooth heatmaps + stacked legends)
# =========================================================
df = None
if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)

        if {"Latitude", "Longitude", "Yield"}.issubset(df.columns):
            # --- Compute Revenue & Profit ---
            df["Revenue_per_acre"] = df["Yield"] * sell_price
            fert_costs = fert_products["CostPerAcre"].sum() if not fert_products.empty else 0
            seed_costs = seed_products["CostPerAcre"].sum() if not seed_products.empty else 0
            df["NetProfit_per_acre"] = (
                df["Revenue_per_acre"] - base_expenses_per_acre - fert_costs - seed_costs
            )

            # Warn if sell price is zero (prevents flat red heatmap)
            if sell_price == 0:
                st.warning("⚠️ Sell Price is 0 — profit heatmap will be flat. Set a non-zero $/bu.")

            # --- Fit map to data bounds ---
            south, north = df["Latitude"].min(), df["Latitude"].max()
            west,  east  = df["Longitude"].min(), df["Longitude"].max()
            m.fit_bounds([[south, west], [north, east]])

            # --- Helper: smooth heatmap overlay (linear + nearest fallback) ---
            import matplotlib.pyplot as plt
            from scipy.interpolate import griddata
            from scipy.spatial import cKDTree
            import numpy as np
            import folium

            def add_heatmap_overlay(values, name, show_default):
                # Grid covering the field
                n = 220
                lon_lin = np.linspace(west, east, n)
                lat_lin = np.linspace(south, north, n)
                lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)

                # Interpolate (linear) with nearest fallback to fill NaNs
                pts = (df["Longitude"].values, df["Latitude"].values)
                v   = values.astype(float)
                grid_lin = griddata(pts, v, (lon_grid, lat_grid), method="linear")
                grid_nn  = griddata(pts, v, (lon_grid, lat_grid), method="nearest")
                grid     = np.where(np.isnan(grid_lin), grid_nn, grid_lin)

                # Normalize to RdYlGn (red=low, green=high)
                vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))
                if vmin == vmax:  # avoid a flat palette
                    vmax = vmin + 1.0
                cmap = plt.cm.get_cmap("RdYlGn")
                rgba = cmap((grid - vmin) / (vmax - vmin))
                rgba = np.flipud(rgba)                     # correct north/south for Folium
                rgba = (rgba * 255).astype(np.uint8)

                folium.raster_layers.ImageOverlay(
                    image=rgba,
                    bounds=[[south, west], [north, east]],   # [ [S,W], [N,E] ]
                    opacity=0.60,
                    name=name,
                    show=show_default
                ).add_to(m)

                return (vmin, vmax)

            # Add layers (Yield off by default, Profit on)
            y_min, y_max = add_heatmap_overlay(df["Yield"].values,            "Yield (bu/ac)",       show_default=False)
            p_min, p_max = add_heatmap_overlay(df["NetProfit_per_acre"].values,"Net Profit ($/ac)",  show_default=True)
# --- Stacked legends (bottom-left) ---
from branca.element import Element

def rgba_to_hex(rgba_tuple):
    r, g, b, a = (int(round(255*x)) for x in rgba_tuple)
    return f"#{r:02x}{g:02x}{b:02x}"

stops = []
for i in range(0, 101, 10):
    color = plt.cm.get_cmap("RdYlGn")(i/100.0)
    stops.append(f"{rgba_to_hex(color)} {i}%")
gradient_css = ", ".join(stops)

legend_html = f"""
<div style="
    position: fixed; bottom: 20px; left: 20px; z-index: 9999;
    display: flex; flex-direction: column; gap: 6px;
    font-family: sans-serif; font-size: 12px; color: black;">
  
  <!-- Yield Legend -->
  <div style="background: rgba(255,255,255,0.6); padding: 4px 6px; border-radius: 4px;">
    <div style="font-weight: 600; margin-bottom: 2px;">Yield (bu/ac)</div>
    <div style="height: 14px; background: linear-gradient(90deg, {gradient_css}); border-radius: 2px;"></div>
    <div style="display:flex; justify-content: space-between; margin-top: 2px;">
      <span>{y_min:.1f}</span><span>{y_max:.1f}</span>
    </div>
  </div>

  <!-- Profit Legend -->
  <div style="background: rgba(255,255,255,0.6); padding: 4px 6px; border-radius: 4px;">
    <div style="font-weight: 600; margin-bottom: 2px;">Net Profit ($/ac)</div>
    <div style="height: 14px; background: linear-gradient(90deg, {gradient_css}); border-radius: 2px;"></div>
    <div style="display:flex; justify-content: space-between; margin-top: 2px;">
      <span>{p_min:.2f}</span><span>{p_max:.2f}</span>
    </div>
  </div>
</div>
"""

legend_element = Element(legend_html)
m.get_root().html.add_child(legend_element)

  # --- Add hover tooltips for Yield + Profit ---
try:
    # Use the same grid size as heatmaps for hover sampling
    n = 40  # fewer points = lighter map, more = denser hover coverage
    lon_lin = np.linspace(west, east, n)
    lat_lin = np.linspace(south, north, n)
    lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)

    # Interpolate yield + profit values at these grid points
    pts = (df["Longitude"].values, df["Latitude"].values)
    y_vals = griddata(pts, df["Yield"].values, (lon_grid, lat_grid), method="linear")
    p_vals = griddata(pts, df["NetProfit_per_acre"].values, (lon_grid, lat_grid), method="linear")

    # Add invisible markers with tooltips
    for i in range(n):
        for j in range(n):
            if not np.isnan(y_vals[i, j]) and not np.isnan(p_vals[i, j]):
                folium.CircleMarker(
                    location=[lat_grid[i, j], lon_grid[i, j]],
                    radius=0.1,  # essentially invisible
                    color="transparent",
                    fill=False,
                    tooltip=f"Yield: {y_vals[i,j]:.1f} bu/ac<br>Profit: ${p_vals[i,j]:.2f}/ac"
                ).add_to(m)
except Exception as e:
    st.warning(f"Could not add hover tooltips: {e}")
       

# =========================================================
# 8. DISPLAY MAP
# =========================================================
folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width=900, height=600)

# =========================================================
# 9. PROFIT SUMMARY
# =========================================================
st.header("Profit Summary")

if df is not None:
    revenue_per_acre = df["Revenue_per_acre"].mean()
    net_profit_per_acre = df["NetProfit_per_acre"].mean()

    st.subheader("Base Expenses (Flat $/ac)")
    st.write(pd.DataFrame(expenses.items(), columns=["Expense","$/ac"]))

    if not fert_products.empty:
        st.subheader("Fertilizer Costs (Per Product)")
        st.dataframe(fert_products[["product","Acres","CostTotal","CostPerAcre"]])

    if not seed_products.empty:
        st.subheader("Seed Costs (Per Product)")
        st.dataframe(seed_products[["product","Acres","CostTotal","CostPerAcre"]])

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
    st.write("Upload a yield map (or zone file) to see profit results.")
