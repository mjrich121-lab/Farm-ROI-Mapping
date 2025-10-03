# =========================================================
# Farm Profit Mapping Tool V4 
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
import shutil

st.set_page_config(page_title="Farm ROI Tool V3", layout="wide")
st.title("Farm Profit Mapping Tool V3")

# --- Initialize session state defaults once here ---
if "fert_products" not in st.session_state:
    st.session_state["fert_products"] = pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])
if "seed_products" not in st.session_state:
    st.session_state["seed_products"] = pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])
if "zones_gdf" not in st.session_state:
    st.session_state["zones_gdf"] = None
if "yield_df" not in st.session_state:
    st.session_state["yield_df"] = None
if "fixed_products" not in st.session_state:
    st.session_state["fixed_products"] = pd.DataFrame(columns=["Type","Product","Rate","CostPerUnit","$/ac"])

# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
    .block-container { max-width: 100% !important; ... }
    ...
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
zone_file = st.file_uploader(
    "Upload Zone Map",
    type=["geojson", "json", "zip"],
    key="zone"
)
st.markdown(
    "_Accepted formats: **GeoJSON, JSON, or a zipped Shapefile (.zip containing .shp, .shx, .dbf, .prj)**. "
    "⚠️ Uploading just a single .shp file will not work._"
)

zones_gdf = None
if zone_file is not None:
    try:
        # --- Load into GeoDataFrame ---
        if zone_file.name.endswith((".geojson", ".json")):
            zones_gdf = gpd.read_file(zone_file)
        elif zone_file.name.endswith(".zip"):
            with open("temp.zip", "wb") as f:
                f.write(zone_file.getbuffer())
            with zipfile.ZipFile("temp.zip", "r") as zip_ref:
                zip_ref.extractall("temp_shp")
            for f_name in os.listdir("temp_shp"):
                if f_name.endswith(".shp"):
                    zones_gdf = gpd.read_file(os.path.join("temp_shp", f_name))
                    break
            os.remove("temp.zip")
            shutil.rmtree("temp_shp", ignore_errors=True)

        if zones_gdf is not None and not zones_gdf.empty:
            st.success(f"✅ Zone map loaded successfully with {len(zones_gdf)} zones.")

            # --- Find existing zone name column OR make one ---
            zone_col = None
            for candidate in ["Zone", "zone", "ZONE", "Name", "name"]:
                if candidate in zones_gdf.columns:
                    zone_col = candidate
                    break
            if zone_col is None:
                zones_gdf["ZoneIndex"] = range(1, len(zones_gdf) + 1)
                zone_col = "ZoneIndex"

            # Ensure we **actually have a 'Zone' column** in the GDF for mapping
            zones_gdf["Zone"] = zones_gdf[zone_col]

            # --- Acre calculation in equal-area CRS ---
            gdf_area = zones_gdf.copy()
            if gdf_area.crs is None:
                gdf_area.set_crs(epsg=4326, inplace=True)  # assume WGS84
            if gdf_area.crs.is_geographic:
                gdf_area = gdf_area.to_crs(epsg=5070)       # Albers Equal Area (USA)

            zones_gdf["Calculated Acres"] = (gdf_area.geometry.area * 0.000247105).astype(float)
            zones_gdf["Override Acres"]   = zones_gdf["Calculated Acres"].astype(float)

            # --- Keep geometry in EPSG:4326 for Folium ---
            if zones_gdf.crs is None or zones_gdf.crs.to_string() != "EPSG:4326":
                zones_gdf = zones_gdf.to_crs(epsg=4326)

            # --- Editable overrides (centered) ---
            display_df = zones_gdf[["Zone", "Calculated Acres", "Override Acres"]].copy()

            c1, c2, c3 = st.columns([1,2,1])
            with c2:
                edited = st.data_editor(
                    display_df,
                    num_rows="fixed",
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Zone": st.column_config.TextColumn(disabled=True),
                        "Calculated Acres": st.column_config.NumberColumn(format="%.2f", disabled=True),
                        "Override Acres": st.column_config.NumberColumn(format="%.2f"),
                    },
                    key="zone_acres_editor",
                )

                # sanitize: blanks/None -> Calculated
                edited["Override Acres"] = pd.to_numeric(edited["Override Acres"], errors="coerce")
                edited["Override Acres"] = edited["Override Acres"].fillna(edited["Calculated Acres"])

                # totals
                total_calc     = float(zones_gdf["Calculated Acres"].sum())
                total_override = float(edited["Override Acres"].sum())
                st.markdown(f"**Total Acres → Calculated: {total_calc:,.2f} | Override: {total_override:,.2f}**")

            # push overrides back into the GDF (keep columns for tooltip)
            zones_gdf["Override Acres"] = edited["Override Acres"].astype(float).values

            # save for downstream
            st.session_state["zones_gdf"] = zones_gdf

        else:
            st.error("❌ Could not load zone map. Please check file format.")

    except Exception as e:
        st.error(f"❌ Error processing zone map: {e}")

# =========================================================
# 2. YIELD MAP UPLOAD
# =========================================================
st.header("Yield Map Upload")
yield_file = st.file_uploader(
    "Upload Yield Map",
    type=["csv", "geojson", "json", "zip"],
    key="yield"
)
st.markdown(
    "_Accepted formats: **CSV, GeoJSON, JSON, or a zipped Shapefile "
    "(.zip containing .shp, .shx, .dbf, .prj)**. ⚠️ Uploading just a single .shp file will not work._"
)

df = None
if yield_file is not None:
    try:
        if yield_file.name.endswith(".csv"):
            df = pd.read_csv(yield_file)

            # --- Normalize column names ---
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # --- Debug: show available columns ---
            with st.expander("📂 Columns detected in uploaded CSV"):
                st.write(list(df.columns))

            # --- Prefer dry yield volume ---
            dry_candidates = [c for c in df.columns if "yld" in c and "vol" in c and "dr" in c]

            if dry_candidates:
                chosen = dry_candidates[0]
                df.rename(columns={chosen: "Yield"}, inplace=True)
                st.success(f"✅ Yield CSV loaded successfully (using dry yield column '{chosen}').")
            else:
                st.error("❌ No usable dry yield column found (expected something like 'yld_vol_dr').")

        else:
            gdf = load_vector_file(yield_file)
            if gdf is not None and not gdf.empty:
                # --- Normalize column names ---
                gdf.columns = [c.strip().lower().replace(" ", "_") for c in gdf.columns]

                # --- Debug: show available columns ---
                with st.expander("📂 Columns detected in uploaded shapefile"):
                    st.write(list(gdf.columns))

                gdf["Longitude"] = gdf.geometry.centroid.x
                gdf["Latitude"] = gdf.geometry.centroid.y

                # --- Prefer dry yield volume ---
                dry_candidates = [c for c in gdf.columns if "yld" in c and "vol" in c and "dr" in c]

                if dry_candidates:
                    chosen = dry_candidates[0]
                    gdf.rename(columns={chosen: "Yield"}, inplace=True)
                    df = pd.DataFrame(gdf.drop(columns="geometry"))
                    st.success(f"✅ Yield shapefile loaded successfully (using dry yield column '{chosen}').")
                else:
                    st.error("❌ No usable dry yield column found (expected something like 'yld_vol_dr').")

            else:
                st.error("❌ Could not read shapefile/geojson")

    except Exception as e:
        st.error(f"❌ Error processing yield file: {e}")

# Save to session state
if df is not None:
    st.session_state["yield_df"] = df

# =========================================================
# 3. PRESCRIPTION MAP UPLOADS
# =========================================================
st.header("Prescription Map Uploads")

fert_file = st.file_uploader(
    "Upload Fertilizer Prescription Map",
    type=["csv", "geojson", "json", "zip"],
    key="fert"
)
st.markdown(
    "_Accepted formats: **CSV, GeoJSON, JSON, or a zipped Shapefile (.zip containing .shp, .shx, .dbf, .prj)**. ⚠️ Uploading just a single .shp file will not work._"
)

seed_file = st.file_uploader(
    "Upload Seed Prescription Map",
    type=["csv", "geojson", "json", "zip"],
    key="seed"
)
st.markdown(
    "_Accepted formats: **CSV, GeoJSON, JSON, or a zipped Shapefile (.zip containing .shp, .shx, .dbf, .prj)**. ⚠️ Uploading just a single .shp file will not work._"
)


def process_prescription(file, prescrip_type="fertilizer"):
    """Process fertilizer/seed prescription maps safely."""
    if file is None:
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])

    try:
        # --- Load CSV or shapefile ---
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            gdf = load_vector_file(file)
            if gdf is None or gdf.empty:
                return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])

            gdf["Longitude"] = gdf.geometry.centroid.x
            gdf["Latitude"] = gdf.geometry.centroid.y

            # Auto-calc acres if not provided
            if "acres" not in gdf.columns:
                gdf["acres"] = gdf.geometry.area * 0.000247105  # m² → acres

            df = pd.DataFrame(gdf.drop(columns="geometry"))

    except Exception as e:
        st.error(f"❌ Error processing {prescrip_type} map: {e}")
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])

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

    # --- Ensure acres column exists ---
    if "acres" not in df.columns:
        df["acres"] = 0.0

    # --- Manual override (per-upload acres) ---
    avg_acres_override = st.number_input(
        f"Override Acres Per Polygon for {prescrip_type.capitalize()} Map",
        min_value=0.0, value=0.0, step=0.1
    )
    if avg_acres_override > 0:
        df["acres"] = avg_acres_override

    # --- Calculate costs ---
    if "costtotal" not in df.columns:
        if "price_per_unit" in df.columns and "units" in df.columns:
            df["costtotal"] = df["price_per_unit"] * df["units"]
        elif "rate" in df.columns and "price" in df.columns:
            df["costtotal"] = df["rate"] * df["price"]
        else:
            df["costtotal"] = 0

    if not df.empty:
        grouped = df.groupby("product", as_index=False).agg(
            Acres=("acres","sum"),
            CostTotal=("costtotal","sum")
        )
        grouped["CostPerAcre"] = grouped.apply(
            lambda x: x["CostTotal"] / x["Acres"] if x["Acres"] > 0 else 0, axis=1
        )
        return grouped

    return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])


# --- Store results in session state safely ---
if "fert_products" not in st.session_state:
    st.session_state["fert_products"] = pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])
if "seed_products" not in st.session_state:
    st.session_state["seed_products"] = pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])

if fert_file is not None:
    st.session_state["fert_products"] = process_prescription(fert_file, "fertilizer")

if seed_file is not None:
    st.session_state["seed_products"] = process_prescription(seed_file, "seed")

# --- Feedback (safe checks) ---
if not st.session_state["fert_products"].empty:
    st.success("✅ Fertilizer prescription uploaded successfully")

if not st.session_state["seed_products"].empty:
    st.success("✅ Seed prescription uploaded successfully")


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
# 4B. FIXED RATE INPUTS (Manual Seed & Fertilizer Table)
# =========================================================
st.header("Fixed Rate Prescription Inputs")

with st.expander("Fixed Rate Inputs (Seed & Fertilizer)", expanded=False):
    # Initialize session state if missing
    if "fixed_products" not in st.session_state:
        st.session_state["fixed_products"] = pd.DataFrame(
            {
                "Type": ["Seed", "Fertilizer"],
                "Product": ["", ""],
                "Rate": [0.0, 0.0],
                "CostPerUnit": [0.0, 0.0],
                "$/ac": [0.0, 0.0]
            }
        )

    # Show editable table
    fixed_entries = st.data_editor(
        st.session_state["fixed_products"],
        num_rows="dynamic",
        use_container_width=True,
        key="fixed_editor"
    )

    # Safely store back into session state
    st.session_state["fixed_products"] = (
        fixed_entries.copy().reset_index(drop=True)
    )
# =========================================================
# 4C. VARIABLE RATE INPUTS (Summary Tables)
# =========================================================
st.header("Variable Rate Prescription Inputs")

with st.expander("Variable Rate Inputs (Seed & Fertilizer)", expanded=False):
    fert_df = st.session_state["fert_products"]
    seed_df = st.session_state["seed_products"]

    if fert_df is not None and not fert_df.empty:
        st.subheader("Fertilizer Products (Variable Rate)")
        st.dataframe(
            fert_df.style.format({
                "Acres": "{:,.1f}",
                "CostTotal": "${:,.2f}",
                "CostPerAcre": "${:,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    if seed_df is not None and not seed_df.empty:
        st.subheader("Seed Products (Variable Rate)")
        st.dataframe(
            seed_df.style.format({
                "Acres": "{:,.1f}",
                "CostTotal": "${:,.2f}",
                "CostPerAcre": "${:,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    if (fert_df is None or fert_df.empty) and (seed_df is None or seed_df.empty):
        st.info("No variable rate prescription maps uploaded yet.")
# =========================================================
# 4D. OPTIONAL: Compare Crop Profitability Before Mapping
# =========================================================
st.header("Compare Crop Profitability (Optional)")

with st.expander("Enter Corn & Soybean Price/Yield Assumptions", expanded=False):
    st.markdown("_Any values you enter here will also be reflected in the Profit Summary section below._")

    st.session_state["corn_yield"] = st.number_input(
        "Corn Yield Goal (bu/ac)", 
        min_value=0.0, 
        value=st.session_state.get("corn_yield", 200.0), 
        step=1.0
    )
    st.session_state["corn_price"] = st.number_input(
        "Corn Sell Price ($/bu)", 
        min_value=0.0, 
        value=st.session_state.get("corn_price", 5.0), 
        step=0.1
    )
    st.session_state["bean_yield"] = st.number_input(
        "Soybean Yield Goal (bu/ac)", 
        min_value=0.0, 
        value=st.session_state.get("bean_yield", 60.0), 
        step=1.0
    )
    st.session_state["bean_price"] = st.number_input(
        "Soybean Sell Price ($/bu)", 
        min_value=0.0, 
        value=st.session_state.get("bean_price", 12.0), 
        step=0.1
    )
# --- Preview chart with same highlighting as Section 9 ---
preview_df = pd.DataFrame({
    "Crop": ["Corn", "Soybeans"],
    "Yield Goal (bu/ac)": [st.session_state["corn_yield"], st.session_state["bean_yield"]],
    "Sell Price ($/bu)": [st.session_state["corn_price"], st.session_state["bean_price"]],
    "Revenue ($/ac)": [
        st.session_state["corn_yield"] * st.session_state["corn_price"],
        st.session_state["bean_yield"] * st.session_state["bean_price"]
    ],
    "Fixed Inputs ($/ac)": [base_expenses_per_acre, base_expenses_per_acre],
    "Breakeven Budget ($/ac)": [
        (st.session_state["corn_yield"] * st.session_state["corn_price"]) - base_expenses_per_acre,
        (st.session_state["bean_yield"] * st.session_state["bean_price"]) - base_expenses_per_acre
    ]
})

def highlight_budget(val):
    if isinstance(val, (int, float)):
        if val > 0:
            return "color: green; font-weight: bold;"
        elif val < 0:
            return "color: red; font-weight: bold;"
    return "font-weight: bold;"

st.dataframe(
    preview_df.style.applymap(
        highlight_budget,
        subset=["Breakeven Budget ($/ac)"]
    ).format({
        "Yield Goal (bu/ac)": "{:,.1f}",
        "Sell Price ($/bu)": "${:,.2f}",
        "Revenue ($/ac)": "${:,.2f}",
        "Fixed Inputs ($/ac)": "${:,.2f}",
        "Breakeven Budget ($/ac)": "${:,.2f}"
    }),
    use_container_width=True,
    hide_index=True
)
# =========================================================
# 5. BASE MAP (rebuild clean each run but persist data state)
# =========================================================
from branca.element import MacroElement, Template, Element

def make_base_map():
    m = folium.Map(
        location=[39.5, -98.35],  # Center of continental US
        zoom_start=5,             # Default zoom
        min_zoom=2,               # Prevent zooming too far out
        tiles=None,
        scrollWheelZoom=False,    # Disabled by default
        prefer_canvas=True
    )

    # Add locked base layers (satellite + labels)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", overlay=False, control=False
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", overlay=True, control=False
    ).add_to(m)

    # ✅ Click-to-enable scroll wheel (disable again on mouseout)
    template = Template("""
        {% macro script(this, kwargs) %}
        var map = {{this._parent.get_name()}};
        map.scrollWheelZoom.disable();
        map.on('click', function() {
            map.scrollWheelZoom.enable();
        });
        map.on('mouseout', function() {
            map.scrollWheelZoom.disable();
        });
        {% endmacro %}
    """)
    macro = MacroElement()
    macro._template = template
    m.get_root().add_child(macro)

    return m

# Always start with a fresh map each run
m = make_base_map()
# =========================================================
# 6. MAP DISPLAY (Zones overlay + legend)
# =========================================================
if "zones_gdf" in st.session_state and st.session_state["zones_gdf"] is not None:
    zones_gdf = st.session_state["zones_gdf"].to_crs(epsg=4326)

    # Center map on field bounds without re-creating the map
    bounds = zones_gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    m.location = [center_lat, center_lon]
    m.zoom_start = 15

    # Zone colors
    zone_colors = {
        1: "#FF0000",   # Red
        2: "#FF8C00",   # Orange
        3: "#FFFF00",   # Yellow
        4: "#32CD32",   # Light Green
        5: "#006400"    # Dark Green
    }

    # Zones overlay (transparent so yield/heatmaps show beneath)
    folium.GeoJson(
        zones_gdf,
        name="Zones",
        style_function=lambda feature: {
            "fillColor": zone_colors.get(int(feature["properties"]["Zone"]), "#808080"),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.1,
        },
        tooltip=folium.GeoJsonTooltip(fields=["Zone", "Calculated Acres", "Override Acres"])
    ).add_to(m)

    # ✅ Zone legend in bottom-left
    unique_zones = sorted(zones_gdf["Zone"].unique())
    zone_legend_html = """
    <div style="
        position: absolute; 
        bottom: 20px; left: 20px; 
        z-index:9999; 
        background-color: rgba(0,0,0,0.6); 
        padding: 6px 10px; 
        border-radius: 5px; 
        font-size: 13px; 
        color: white;
    ">
    <b>Zone Colors</b><br>
    """
    for z in unique_zones:
        color = zone_colors.get(int(z), "#808080")
        zone_legend_html += f"<div style='display:flex; align-items:center; margin:2px 0;'> \
                        <div style='background:{color}; width:14px; height:14px; margin-right:6px;'></div> Zone {z}</div>"
    zone_legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(zone_legend_html))

    # ✅ Keep overlay toggles
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

# =========================================================
# 7. YIELD + PROFIT (Variable + Fixed Rate overlays)
# =========================================================
if st.session_state["yield_df"] is not None and not st.session_state["yield_df"].empty:
    df = st.session_state["yield_df"].copy()

    # 🔍 Detect yield column
    yield_col_priority = ["Dry_Yield","DryYield","DryYld","Yld_Dry",
                          "YIELD","Yield","Yld","YLD","YLD_BuAc","USBU_AC","WET_YLD"]
    yield_col = None
    for candidate in yield_col_priority:
        for col in df.columns:
            if candidate.lower() in col.lower():
                yield_col = col
                break
        if yield_col: break

    if yield_col is None:
        st.error("❌ No recognized yield column found in uploaded file.")
    else:
        df = df.rename(columns={yield_col:"Yield"})
        st.success(f"✅ Using `{yield_col}` column for Yield calculations (renamed to `Yield`).")
        st.session_state["yield_df"] = df
        df["Revenue_per_acre"] = df["Yield"] * sell_price

        # Variable rate profit
        fert_costs_var = st.session_state["fert_products"]["CostPerAcre"].sum() if not st.session_state["fert_products"].empty else 0
        seed_costs_var = st.session_state["seed_products"]["CostPerAcre"].sum() if not st.session_state["seed_products"].empty else 0
        df["NetProfit_per_acre_variable"] = df["Revenue_per_acre"] - base_expenses_per_acre - fert_costs_var - seed_costs_var

        # Fixed rate profit
        fixed_costs = 0
        if not st.session_state["fixed_products"].empty:
            fixed_df = st.session_state["fixed_products"].copy()
            fixed_df["$/ac"] = fixed_df.apply(
                lambda x: x["Rate"] * x["CostPerUnit"] if x["Rate"] > 0 and x["CostPerUnit"] > 0 else 0, axis=1
            )
            fixed_costs = fixed_df["$/ac"].sum()
        df["NetProfit_per_acre_fixed"] = df["Revenue_per_acre"] - base_expenses_per_acre - fixed_costs

        # ✅ Auto-zoom to combined bounds
        bounds_list = []
        if "zones_gdf" in st.session_state and st.session_state["zones_gdf"] is not None:
            zb = st.session_state["zones_gdf"].total_bounds
            bounds_list.append([[zb[1], zb[0]], [zb[3], zb[2]]])
        if {"Latitude","Longitude"}.issubset(df.columns):
            bounds_list.append([[df["Latitude"].min(), df["Longitude"].min()],
                                [df["Latitude"].max(), df["Longitude"].max()]])

        if bounds_list:
            south = min(b[0][0] for b in bounds_list)
            west  = min(b[0][1] for b in bounds_list)
            north = max(b[1][0] for b in bounds_list)
            east  = max(b[1][1] for b in bounds_list)
            m.fit_bounds([[south, west], [north, east]])
        else:
            # fallback center of US if no bounds found
            south, west, north, east = 25, -125, 49, -66

        # ✅ Heatmap helper
        def add_heatmap_overlay(values, name, show_default):
            if len(values) == 0:  # safety check
                return (0, 0)
            n = 200
            lon_lin = np.linspace(west, east, n)
            lat_lin = np.linspace(south, north, n)
            lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)
            pts = (df["Longitude"].values, df["Latitude"].values)
            grid_lin = griddata(pts, values, (lon_grid, lat_grid), method="linear")
            grid_nn = griddata(pts, values, (lon_grid, lat_grid), method="nearest")
            grid = np.where(np.isnan(grid_lin), grid_nn, grid_lin)
            vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))
            if vmin == vmax: vmax = vmin + 1
            cmap = plt.cm.get_cmap("RdYlGn")
            rgba = cmap((grid - vmin) / (vmax - vmin))
            rgba = np.flipud(rgba)
            rgba = (rgba * 255).astype(np.uint8)

            overlay = folium.raster_layers.ImageOverlay(
                image=rgba,
                bounds=[[south, west], [north, east]],
                opacity=0.5,
                name=name,
                overlay=True,
                show=show_default
            )
            overlay.add_to(m)
            return (vmin, vmax)

        # Add overlays
        y_min, y_max = add_heatmap_overlay(df["Yield"].values, "Yield (bu/ac)", show_default=False)
        v_min, v_max = add_heatmap_overlay(df["NetProfit_per_acre_variable"].values, "Variable Rate Profit ($/ac)", show_default=True)
        f_min, f_max = add_heatmap_overlay(df["NetProfit_per_acre_fixed"].values, "Fixed Rate Profit ($/ac)", show_default=False)

        # ✅ Legend only if overlays were created
        if y_min != y_max:
            def rgba_to_hex(rgba_tuple):
                r, g, b, a = (int(round(255*x)) for x in rgba_tuple)
                return f"#{r:02x}{g:02x}{b:02x}"

            stops = [f"{rgba_to_hex(plt.cm.get_cmap('RdYlGn')(i/100.0))} {i}%" for i in range(0, 101, 10)]
            gradient_css = ", ".join(stops)

            profit_legend_html = f"""
            <div style="position: fixed; top: 20px; left: 20px; z-index: 9999;
                        display: flex; flex-direction: column; gap: 10px; 
                        font-family: sans-serif; font-size: 12px; color: white;">

              <div>
                <div style="font-weight: 600; margin-bottom: 2px;">Yield (bu/ac)</div>
                <div style="height: 14px; background: linear-gradient(90deg, {gradient_css}); border-radius: 2px;"></div>
                <div style="display:flex; justify-content: space-between; margin-top: 2px;">
                  <span>{y_min:.1f}</span><span>{y_max:.1f}</span>
                </div>
              </div>

              <div>
                <div style="font-weight: 600; margin-bottom: 2px;">Variable Rate Profit ($/ac)</div>
                <div style="height: 14px; background: linear-gradient(90deg, {gradient_css}); border-radius: 2px;"></div>
                <div style="display:flex; justify-content: space-between; margin-top: 2px;">
                  <span>{v_min:.2f}</span><span>{v_max:.2f}</span>
                </div>
              </div>

              <div>
                <div style="font-weight: 600; margin-bottom: 2px;">Fixed Rate Profit ($/ac)</div>
                <div style="height: 14px; background: linear-gradient(90deg, {gradient_css}); border-radius: 2px;"></div>
                <div style="display:flex; justify-content: space-between; margin-top: 2px;">
                  <span>{f_min:.2f}</span><span>{f_max:.2f}</span>
                </div>
              </div>
            </div>
            """
            m.get_root().html.add_child(Element(profit_legend_html))


# =========================================================
# 8. DISPLAY MAP
# =========================================================
st_folium(m, use_container_width=True, height=600)


# =========================================================
# 9. PROFIT SUMMARY
# =========================================================
st.header("Profit Summary")

# --- Ensure session state keys always exist ---
if "fert_products" not in st.session_state:
    st.session_state["fert_products"] = pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])
if "seed_products" not in st.session_state:
    st.session_state["seed_products"] = pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])
if "zones_gdf" not in st.session_state:
    st.session_state["zones_gdf"] = None
if "yield_df" not in st.session_state:
    st.session_state["yield_df"] = None

# --- Safe defaults ---
revenue_per_acre = 0.0
net_profit_per_acre = 0.0
expenses_per_acre = base_expenses_per_acre if "base_expenses_per_acre" in locals() else 0.0

if st.session_state["yield_df"] is not None and not st.session_state["yield_df"].empty:
    df = st.session_state["yield_df"]
    if "Revenue_per_acre" in df.columns:
        revenue_per_acre = df["Revenue_per_acre"].mean()
    if "NetProfit_per_acre" in df.columns:
        net_profit_per_acre = df["NetProfit_per_acre"].mean()

# --- Layout (two columns) ---
col_left, col_right = st.columns([2, 2])

# --------------------------
# LEFT SIDE = Breakeven + Profit Comparison
# --------------------------
with col_left:
    st.subheader("Breakeven Budget Tool (Corn vs Beans)")
    st.markdown("_These values are linked to the 'Compare Crop Profitability' section above the map._")

    # --- Pull values from session_state (set in 4D or fallback defaults) ---
    corn_yield = st.session_state.get("corn_yield", 200.0)
    corn_price = st.session_state.get("corn_price", 5.0)
    bean_yield = st.session_state.get("bean_yield", 60.0)
    bean_price = st.session_state.get("bean_price", 12.0)

    # --- Calculate breakeven budgets ---
    corn_revenue = corn_yield * corn_price
    bean_revenue = bean_yield * bean_price

    corn_budget = corn_revenue - expenses_per_acre
    bean_budget = bean_revenue - expenses_per_acre

    breakeven_df = pd.DataFrame({
        "Crop": ["Corn", "Soybeans"],
        "Yield Goal (bu/ac)": [corn_yield, bean_yield],
        "Sell Price ($/bu)": [corn_price, bean_price],
        "Revenue ($/ac)": [corn_revenue, bean_revenue],
        "Fixed Inputs ($/ac)": [expenses_per_acre, expenses_per_acre],
        "Breakeven Budget ($/ac)": [corn_budget, bean_budget]
    })

    # --- Styling for color highlight ---
    def highlight_budget(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: green; font-weight: bold;"
            elif val < 0:
                return "color: red; font-weight: bold;"
        return "font-weight: bold;"

    st.dataframe(
        breakeven_df.style.applymap(
            highlight_budget,
            subset=["Breakeven Budget ($/ac)"]
        ).format({
            "Yield Goal (bu/ac)": "{:,.1f}",
            "Sell Price ($/bu)": "${:,.2f}",
            "Revenue ($/ac)": "${:,.2f}",
            "Fixed Inputs ($/ac)": "${:,.2f}",
            "Breakeven Budget ($/ac)": "${:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --- Note to direct users back to 4D ---
    st.caption("To adjust Corn and Soybean assumptions, edit values in the **Compare Crop Profitability (Optional)** section above the map.")

    # --- Profit Metrics Comparison ---
    st.subheader("Profit Metrics Comparison")

    # Variable Rate Profit
    var_profit = 0.0
    if st.session_state["yield_df"] is not None and not st.session_state["yield_df"].empty:
        df = st.session_state["yield_df"]

        fert_costs = st.session_state["fert_products"]["CostPerAcre"].sum() if not st.session_state["fert_products"].empty else 0
        seed_costs = st.session_state["seed_products"]["CostPerAcre"].sum() if not st.session_state["seed_products"].empty else 0

        revenue_var = df["Revenue_per_acre"].mean() if "Revenue_per_acre" in df.columns else 0.0
        expenses_var = base_expenses_per_acre + fert_costs + seed_costs
        var_profit = revenue_var - expenses_var
    else:
        revenue_var, expenses_var = 0.0, 0.0

    # Fixed Rate Profit
    fixed_profit = 0.0
    if "fixed_products" in st.session_state and not st.session_state["fixed_products"].empty:
        fert_fixed_costs = st.session_state["fixed_products"][st.session_state["fixed_products"]["Type"]=="Fertilizer"]["$/ac"].sum()
        seed_fixed_costs = st.session_state["fixed_products"][st.session_state["fixed_products"]["Type"]=="Seed"]["$/ac"].sum()
        revenue_fixed = revenue_var
        expenses_fixed = base_expenses_per_acre + fert_fixed_costs + seed_fixed_costs
        fixed_profit = revenue_fixed - expenses_fixed
    else:
        revenue_fixed, expenses_fixed = 0.0, 0.0

    # Breakeven Budget (was Overall)
    revenue_overall = revenue_per_acre
    expenses_overall = expenses_per_acre
    profit_overall = net_profit_per_acre

    # Build numeric-only comparison table
    comparison = pd.DataFrame({
        "Metric": ["Revenue ($/ac)", "Expenses ($/ac)", "Profit ($/ac)"],
        "Breakeven Budget": [round(revenue_overall,2), round(expenses_overall,2), round(profit_overall,2)],
        "Variable Rate": [round(revenue_var,2), round(expenses_var,2), round(var_profit,2)],
        "Fixed Rate": [round(revenue_fixed,2), round(expenses_fixed,2), round(fixed_profit,2)]
    })

    def highlight_profit(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: green; font-weight: bold;"
            elif val < 0:
                return "color: red; font-weight: bold;"
        return "font-weight: bold;"

    st.dataframe(
        comparison.style.applymap(
            highlight_profit,
            subset=["Breakeven Budget","Variable Rate","Fixed Rate"]
        ).format({
            "Breakeven Budget":"${:,.2f}",
            "Variable Rate":"${:,.2f}",
            "Fixed Rate":"${:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Collapsible formulas shown separately
    with st.expander("Show Calculation Formulas", expanded=False):
        st.markdown("""
        <div style="border:1px solid #444; border-radius:6px; padding:10px; margin-bottom:8px; background-color:#111;">
            <b>Breakeven Budget</b><br>
            (Target Yield × Sell Price) − Fixed Inputs
        </div>
        <div style="border:1px solid #444; border-radius:6px; padding:10px; margin-bottom:8px; background-color:#111;">
            <b>Variable Rate</b><br>
            (Avg Yield × Sell Price) − (Fixed Inputs + Var Seed + Var Fert)
        </div>
        <div style="border:1px solid #444; border-radius:6px; padding:10px; margin-bottom:8px; background-color:#111;">
            <b>Fixed Rate</b><br>
            (Avg Yield × Sell Price) − (Fixed Inputs + Fixed Seed + Fixed Fert)
        </div>
        """, unsafe_allow_html=True)

# --------------------------
# RIGHT SIDE = Fixed Inputs
# --------------------------
with col_right:
    st.subheader("Fixed Input Costs")
    fixed_df = pd.DataFrame(list(expenses.items()), columns=["Expense", "$/ac"])
    total_fixed = pd.DataFrame([{"Expense": "Total Fixed Costs", "$/ac": fixed_df["$/ac"].sum()}])
    fixed_df = pd.concat([fixed_df, total_fixed], ignore_index=True)

    styled_fixed = fixed_df.style.format({"$/ac": "${:,.2f}"}).apply(
        lambda x: ["font-weight: bold;" if v == "Total Fixed Costs" else "" for v in x],
        subset=["Expense"]
    ).apply(
        lambda x: ["font-weight: bold;" if i == len(fixed_df) - 1 else "" for i in range(len(x))],
        subset=["$/ac"]
    )

    row_height = 34
    header_buffer = 50
    table_height = len(fixed_df) * row_height + header_buffer

    st.dataframe(
        styled_fixed,
        use_container_width=True,
        hide_index=True,
        height=table_height
    )
