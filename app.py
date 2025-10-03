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
# Make the app responsive to screen size
st.markdown(
    """
    <style>
    /* Force all Streamlit containers to be responsive */
    .block-container {
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Dataframes and tables fit screen width */
    .dataframe {
        width: 100% !important;
        overflow-x: auto !important;
    }

    /* Compact table styling */
    .compact-table td, .compact-table th {
        padding: 4px 8px !important;
        font-size: 12px !important;
        white-space: nowrap;
    }

    /* Map responsiveness */
    iframe[title="st_folium"] {
        width: 100% !important;
        height: 70vh !important;
    }

    /* Make headers tighter */
    h1, h2, h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# =========================================================
# HELPER FUNCTION: Load Shapefiles or GeoJSON
# =========================================================
def load_vector_file(uploaded_file):
    """Load shapefile (.zip), GeoJSON, or JSON into a GeoDataFrame"""
    gdf = None
    if uploaded_file.name.endswith((".geojson", ".json")):
        gdf = gpd.read_file(uploaded_file)
    elif uploaded_file.name.endswith(".zip"):
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        with zipfile.ZipFile("temp.zip", "r") as zip_ref:
            zip_ref.extractall("temp_shp")
        for f_name in os.listdir("temp_shp"):
            if f_name.endswith(".shp"):
                gdf = gpd.read_file(os.path.join("temp_shp", f_name))
                break
        os.remove("temp.zip")
        import shutil
        shutil.rmtree("temp_shp", ignore_errors=True)
    return gdf

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
yield_file = st.file_uploader("Upload Yield Map", type=["csv","geojson","json","zip"], key="yield")

df = None
if yield_file is not None:
    if yield_file.name.endswith(".csv"):
        df = pd.read_csv(yield_file)
        if {"Latitude","Longitude","Yield"}.issubset(df.columns):
            st.success("Yield CSV loaded successfully")
        else:
            st.error("CSV must include Latitude, Longitude, and Yield columns")
    else:
        gdf = load_vector_file(yield_file)
        if gdf is not None:
            gdf["Longitude"] = gdf.geometry.centroid.x
            gdf["Latitude"] = gdf.geometry.centroid.y
            # Try to find yield column
            yield_col = [c for c in gdf.columns if "yield" in c.lower()]
            if yield_col:
                gdf.rename(columns={yield_col[0]: "Yield"}, inplace=True)
                df = pd.DataFrame(gdf.drop(columns="geometry"))
                st.success("Yield shapefile/geojson loaded successfully")
            else:
                st.error("No yield column found in uploaded file")

# =========================================================
# 3. PRESCRIPTION MAP UPLOADS
# =========================================================
st.header("Prescription Map Uploads")

fert_file = st.file_uploader("Upload Fertilizer Prescription Map", type=["csv","geojson","json","zip"], key="fert")
seed_file = st.file_uploader("Upload Seed Prescription Map", type=["csv","geojson","json","zip"], key="seed")

def process_prescription(file, prescrip_type="fertilizer"):
    if file is None:
        return pd.DataFrame()

    # --- Load CSV or shapefile ---
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        gdf = load_vector_file(file)
        if gdf is None:
            return pd.DataFrame()
        gdf["Longitude"] = gdf.geometry.centroid.x
        gdf["Latitude"] = gdf.geometry.centroid.y

        # Auto-calc acres if not provided
        if "acres" not in gdf.columns:
            gdf["acres"] = gdf.geometry.area * 0.000247105  # m² → acres

        df = pd.DataFrame(gdf.drop(columns="geometry"))

    # --- Normalize column names ---
    df.columns = [c.strip().lower() for c in df.columns]

    # --- Detect product column ---
    if "product" not in df.columns:
        for candidate in ["variety","hybrid","type","name","material"]:
            if candidate in df.columns:
                df.rename(columns={candidate: "product"}, inplace=True)
                break
        else:
            df["product"] = prescrip_type.capitalize()

    # --- Ensure acres column exists and allow manual override ---
    if "acres" not in df.columns:
        df["acres"] = 0.0  # placeholder if missing

    # Manual override (per-upload average acres input)
    avg_acres_override = st.number_input(
        f"Override Acres Per Polygon for {prescrip_type.capitalize()} Map",
        min_value=0.0, value=0.0, step=0.1
    )
    if avg_acres_override > 0:
        df["acres"] = avg_acres_override

    # --- Calculate costs ---
    if "product" in df.columns and "acres" in df.columns:
        if "costtotal" not in df.columns:
            if "price_per_unit" in df.columns and "units" in df.columns:
                df["costtotal"] = df["price_per_unit"] * df["units"]
            elif "rate" in df.columns and "price" in df.columns:
                df["costtotal"] = df["rate"] * df["price"]
            else:
                df["costtotal"] = 0

        grouped = df.groupby("product", as_index=False).agg(
            Acres=("acres","sum"),
            CostTotal=("costtotal","sum")
        )
        grouped["CostPerAcre"] = grouped["CostTotal"] / grouped["Acres"]
        return grouped

    return pd.DataFrame()

# Store into session state
if fert_file is not None:
    st.session_state["fert_products"] = process_prescription(fert_file, "fertilizer")
if seed_file is not None:
    st.session_state["seed_products"] = process_prescription(seed_file, "seed")

# Feedback
if not st.session_state["fert_products"].empty:
    st.success("Fertilizer prescription uploaded successfully")
if not st.session_state["seed_products"].empty:
    st.success("Seed prescription uploaded successfully")


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
    # Save into session state
    st.session_state["zones_gdf"] = zones_gdf

# Draw from session state if available
if st.session_state["zones_gdf"] is not None:
    zone_layer = folium.FeatureGroup(name="Zones", show=True)
    static_zone_colors = {1: "#FF0000", 2: "#FF8000", 3: "#FFFF00", 4: "#80FF00", 5: "#008000"}
    zone_col = None
    for candidate in ["Zone","zone","ZONE","Name","name"]:
        if candidate in st.session_state["zones_gdf"].columns:
            zone_col = candidate
            break
    if zone_col is None:
        st.session_state["zones_gdf"]["ZoneIndex"] = range(1, len(st.session_state["zones_gdf"])+1)
        zone_col = "ZoneIndex"
    for _, row in st.session_state["zones_gdf"].iterrows():
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
# 7. YIELD + PROFIT
# =========================================================
df = None
if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        if {"Latitude", "Longitude", "Yield"}.issubset(df.columns):
            st.session_state["yield_df"] = df

# Draw overlays from session state if available
if st.session_state["yield_df"] is not None:
    df = st.session_state["yield_df"]

    # --- Revenue & Profit ---
    df["Revenue_per_acre"] = df["Yield"] * sell_price
    fert_costs = st.session_state["fert_products"]["CostPerAcre"].sum() if not st.session_state["fert_products"].empty else 0
    seed_costs = st.session_state["seed_products"]["CostPerAcre"].sum() if not st.session_state["seed_products"].empty else 0
    df["NetProfit_per_acre"] = (
        df["Revenue_per_acre"] - base_expenses_per_acre - fert_costs - seed_costs
    )

    if sell_price == 0:
        st.warning("⚠️ Sell Price is 0 — profit heatmap will be flat. Set a non-zero $/bu.")

    # --- Auto-zoom to data bounds ---
    south, north = df["Latitude"].min(), df["Latitude"].max()
    west,  east  = df["Longitude"].min(), df["Longitude"].max()
    m.fit_bounds([[south, west], [north, east]])

    # --- Heatmap overlays (Yield + Profit) ---
    def add_heatmap_overlay(values, name, show_default):
        n = 200
        lon_lin = np.linspace(west, east, n)
        lat_lin = np.linspace(south, north, n)
        lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)

        # Interpolate with linear + nearest fallback
        pts = (df["Longitude"].values, df["Latitude"].values)
        grid_lin = griddata(pts, values, (lon_grid, lat_grid), method="linear")
        grid_nn  = griddata(pts, values, (lon_grid, lat_grid), method="nearest")
        grid     = np.where(np.isnan(grid_lin), grid_nn, grid_lin)

        vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))
        if vmin == vmax:
            vmax = vmin + 1
        cmap = plt.cm.get_cmap("RdYlGn")
        rgba = cmap((grid - vmin) / (vmax - vmin))
        rgba = np.flipud(rgba)
        rgba = (rgba * 255).astype(np.uint8)

        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[south, west], [north, east]],
            opacity=0.5,
            name=name,
            show=show_default
        ).add_to(m)

        return (vmin, vmax)

    # Add both overlays
    y_min, y_max = add_heatmap_overlay(df["Yield"].values, "Yield (bu/ac)", show_default=False)
    p_min, p_max = add_heatmap_overlay(df["NetProfit_per_acre"].values, "Net Profit ($/ac)", show_default=True)

    # --- Legends (bottom left stacked) ---
    from branca.element import Element
    def rgba_to_hex(rgba_tuple):
        r, g, b, a = (int(round(255*x)) for x in rgba_tuple)
        return f"#{r:02x}{g:02x}{b:02x}"
    stops = [f"{rgba_to_hex(plt.cm.get_cmap('RdYlGn')(i/100.0))} {i}%" for i in range(0, 101, 10)]
    gradient_css = ", ".join(stops)

    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                display: flex; flex-direction: column; gap: 6px; font-family: sans-serif; font-size: 12px;">
      <div style="background: rgba(255,255,255,0.6); padding: 4px 6px; border-radius: 4px;">
        <div style="font-weight: 600; margin-bottom: 2px;">Yield (bu/ac)</div>
        <div style="height: 14px; background: linear-gradient(90deg, {gradient_css}); border-radius: 2px;"></div>
        <div style="display:flex; justify-content: space-between; margin-top: 2px;">
          <span>{y_min:.1f}</span><span>{y_max:.1f}</span>
        </div>
      </div>
      <div style="background: rgba(255,255,255,0.6); padding: 4px 6px; border-radius: 4px;">
        <div style="font-weight: 600; margin-bottom: 2px;">Net Profit ($/ac)</div>
        <div style="height: 14px; background: linear-gradient(90deg, {gradient_css}); border-radius: 2px;"></div>
        <div style="display:flex; justify-content: space-between; margin-top: 2px;">
          <span>{p_min:.2f}</span><span>{p_max:.2f}</span>
        </div>
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))


# =========================================================
# 8. DISPLAY MAP
# =========================================================
folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, use_container_width=True, height=700)

# =========================================================
# 9. PROFIT SUMMARY (always visible, even without yield map)
# =========================================================
st.header("Profit Summary")

# Default values if no yield map
revenue_per_acre = 0.0
net_profit_per_acre = 0.0

if df is not None:
    revenue_per_acre = df["Revenue_per_acre"].mean()
    net_profit_per_acre = df["NetProfit_per_acre"].mean()

# --- Profit Metrics ---
summary = pd.DataFrame({
    "Metric": ["Revenue ($/ac)", "Expenses ($/ac)", "Profit ($/ac)"],
    "Value": [revenue_per_acre, base_expenses_per_acre, net_profit_per_acre]
})
summary["Value"] = summary["Value"].map("${:,.2f}".format)
summary = summary.set_index("Metric")

def style_profit(val, metric):
    if metric == "Profit ($/ac)":
        num = float(val.replace("$","").replace(",",""))
        if num > 0:
            return "color: green; font-weight: bold;"
        elif num < 0:
            return "color: red; font-weight: bold;"
        else:
            return "font-weight: bold;"
    return ""

# --- Fixed Input Costs ---
fixed_df = pd.DataFrame(expenses.items(), columns=["Expense", "$/ac"])
total_fixed = sum(expenses.values())
fixed_df.loc[len(fixed_df)] = ["Total Fixed Costs", total_fixed]
fixed_df["$/ac"] = fixed_df["$/ac"].map("${:,.2f}".format)
fixed_df = fixed_df.set_index("Expense")

def style_totals(val, idx):
    if "Total" in idx:
        return "font-weight: bold;"
    return ""

# --- Variable Rate Input Costs ---
variable_list = []
if not fert_products.empty:
    fert_display = fert_products[["product", "CostPerAcre"]].copy()
    fert_display.rename(columns={"product": "Product", "CostPerAcre": "$/ac"}, inplace=True)
    variable_list.append(fert_display)
if not seed_products.empty:
    seed_display = seed_products[["product", "CostPerAcre"]].copy()
    seed_display.rename(columns={"product": "Product", "CostPerAcre": "$/ac"}, inplace=True)
    variable_list.append(seed_display)

if variable_list:
    variable_df = pd.concat(variable_list, ignore_index=True)
    total_var = variable_df["$/ac"].sum()
    variable_df.loc[len(variable_df)] = ["Total Variable Costs", total_var]
else:
    variable_df = pd.DataFrame({
        "Product": ["Seed", "Fertilizer 1", "Fertilizer 2", "Fertilizer 3", "Total Variable Costs"],
        "$/ac": [0, 0, 0, 0, 0]
    })

variable_df["$/ac"] = variable_df["$/ac"].apply(lambda x: f"${x:,.2f}")
variable_df = variable_df.set_index("Product")

# --- Layout ---
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("Profit Metrics")
    st.dataframe(
        summary.style.apply(
            lambda col: [style_profit(v, col.name) for v in col],
            axis=0
        )
    )

    st.subheader("Variable Rate Input Costs")
    st.dataframe(
        variable_df.style.apply(
            lambda col: [style_totals(v, idx) for idx, v in zip(variable_df.index, col)],
            axis=0
        )
    )
with right_col:
    st.subheader("Fixed Input Costs")
    st.dataframe(
        fixed_df.style.apply(
            lambda col: [style_totals(v, idx) for idx, v in zip(fixed_df.index, col)],
            axis=0
        ),
        use_container_width=True,              # expand horizontally
        height=(len(fixed_df) * 35 + 40)      # auto-fit vertically (no scroll)
    )
