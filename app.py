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
# HELPER FUNCTION: Load Shapefiles or GeoJSON (bulletproof)
# =========================================================
import tempfile

def load_vector_file(uploaded_file):
    """
    Load shapefile (.zip), GeoJSON, or JSON into a GeoDataFrame safely.
    - Handles missing .prj by assuming WGS84.
    - Guarantees temp cleanup.
    - Always returns WGS84 (EPSG:4326).
    """
    try:
        # GeoJSON / JSON direct
        if uploaded_file.name.lower().endswith((".geojson", ".json")):
            gdf = gpd.read_file(uploaded_file)
        elif uploaded_file.name.lower().endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "in.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmpdir)

                shp_path = None
                for f_name in os.listdir(tmpdir):
                    if f_name.lower().endswith(".shp"):
                        shp_path = os.path.join(tmpdir, f_name)
                        break
                if shp_path is None:
                    return None  # no .shp inside

                gdf = gpd.read_file(shp_path)
        else:
            # Support stray .shp (rare, but safer to catch)
            if uploaded_file.name.lower().endswith(".shp"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    shp_path = os.path.join(tmpdir, uploaded_file.name)
                    with open(shp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    gdf = gpd.read_file(shp_path)
            else:
                return None

        if gdf is None or gdf.empty:
            return None

        # CRS safety: assume WGS84 if missing; convert to 4326 for folium
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        gdf = gdf.to_crs(epsg=4326)
        return gdf

    except Exception:
        return None

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
# 2. YIELD MAP UPLOAD  (multi-file + crash-proof)
# =========================================================
st.header("Yield Map Upload")

yield_files = st.file_uploader(
    "Upload Yield Map(s)",
    type=["csv", "geojson", "json", "zip"],
    key="yield",
    accept_multiple_files=True
)
st.markdown(
    "_Accepted formats: **CSV, GeoJSON, JSON, or a zipped Shapefile "
    "(.zip containing .shp, .shx, .dbf, .prj)**. ⚠️ Uploading just a single .shp file will not work._"
)

# --- Initialize persistent list in session state ---
st.session_state.setdefault("yield_files_list", [])

if yield_files:
    for yf in yield_files:
        try:
            df_temp = None

            # --- CSV case ---
            if yf.name.lower().endswith(".csv"):
                df_temp = pd.read_csv(yf)
            else:
                # --- Vector file case ---
                gdf_temp = load_vector_file(yf)
                if gdf_temp is not None and not gdf_temp.empty:
                    df_temp = pd.DataFrame(gdf_temp.drop(columns="geometry", errors="ignore"))

            # --- Validate dataframe ---
            if df_temp is not None and not df_temp.empty:
                # normalize column names
                df_temp.columns = [c.strip().lower().replace(" ", "_") for c in df_temp.columns]

                # --- Intelligent yield column detection ---
                yield_candidates = []

                # Priority 1 – Dry yield (best accuracy)
                for key in ["yld_vol_dr", "yld_mass_dr", "yield_dry", "dry_yield"]:
                    if key in df_temp.columns:
                        yield_candidates.append(key)

                # Priority 2 – Other yield-like columns
                if not yield_candidates:
                    for key in ["yield", "yld_vol_wt", "yld_mass_wt", "wet_yield"]:
                        if key in df_temp.columns:
                            yield_candidates.append(key)

                # Apply first match, else placeholder
                if yield_candidates:
                    chosen = yield_candidates[0]
                    df_temp.rename(columns={chosen: "Yield"}, inplace=True)
                else:
                    df_temp["Yield"] = 0.0  # triggers manual Target Yield later

                # record successful load
                st.session_state["yield_files_list"].append(
                    {"name": yf.name, "rows": len(df_temp)}
                )
                st.success(f"✅ Loaded {yf.name} ({len(df_temp)} rows)")

            else:
                st.warning(f"⚠️ {yf.name} contained no usable data.")

        except Exception as e:
            st.warning(f"⚠️ Skipped {yf.name}: {e}")

# --- Show currently loaded yield files ---
if st.session_state["yield_files_list"]:
    st.info("### Loaded Yield Files")
    for f in st.session_state["yield_files_list"]:
        st.markdown(f"- **{f['name']}** ({f['rows']} rows)")
else:
    st.caption("No yield files loaded yet.")

# --- Keep backward compatibility for downstream sections ---
st.session_state["yield_df"] = None  # ensures map won't error even if no yield selected

        
# =========================================================
# 3. PRESCRIPTION MAP UPLOADS  (multi-file + crash-proof)
# =========================================================
st.header("Prescription Map Uploads")

# --- Multi-file uploaders ---
fert_files = st.file_uploader(
    "Upload Fertilizer Prescription Map(s)",
    type=["csv", "geojson", "json", "zip"],
    key="fert",
    accept_multiple_files=True
)
seed_files = st.file_uploader(
    "Upload Seed Prescription Map(s)",
    type=["csv", "geojson", "json", "zip"],
    key="seed",
    accept_multiple_files=True
)

st.markdown(
    "_Accepted formats: **CSV, GeoJSON, JSON, or a zipped Shapefile "
    "(.zip containing .shp, .shx, .dbf, .prj)**. ⚠️ Uploading just a single .shp file will not work._"
)

# --- Persistent stores for all uploaded layers ---
st.session_state.setdefault("fert_layers_store", {})
st.session_state.setdefault("seed_layers_store", {})

# =========================================================
# HELPER: Process One Prescription File (existing logic kept)
# =========================================================
def process_prescription(file, prescrip_type="fertilizer"):
    """Process fertilizer/seed prescription maps safely (CSV or polygons)."""
    if file is None:
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])

    # --- Handle vector files (shapefile/geojson/json/zip) ---
    if file.name.lower().endswith((".geojson",".json",".zip",".shp")):
        gdf = load_vector_file(file)
        if gdf is None or gdf.empty:
            st.error(f"❌ Could not read {prescrip_type} prescription map.")
            return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])

        gdf.columns = [c.strip().lower().replace(" ", "_") for c in gdf.columns]

        # Ensure CRS WGS84
        try:
            gdf = gdf.to_crs(epsg=4326)
        except Exception:
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)

        # Add centroids for later (map bounds + tooltips)
        gdf["Longitude"] = gdf.geometry.representative_point().x
        gdf["Latitude"]  = gdf.geometry.representative_point().y

        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    else:
        # --- CSV handling ---
        df = pd.read_csv(file)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

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
        f"Override Acres Per Polygon for {prescrip_type.capitalize()} Map ({file.name})",
        min_value=0.0, value=0.0, step=0.1, key=f"{prescrip_type}_{file.name}_override"
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

    # --- Aggregate ---
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

# =========================================================
# PROCESS ALL UPLOADED FILES  (multi-file support)
# =========================================================

st.session_state.setdefault("fert_products", pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"]))
st.session_state.setdefault("seed_products", pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"]))

# --- Fertilizer files ---
if fert_files:
    for f in fert_files:
        try:
            grouped = process_prescription(f, "fertilizer")
            if not grouped.empty:
                layer_key = os.path.splitext(f.name)[0].lower().replace(" ", "_")
                st.session_state["fert_layers_store"][layer_key] = grouped
                st.success(f"✅ Fertilizer file loaded: {f.name}")
        except Exception as e:
            st.warning(f"⚠️ Skipped fertilizer file {f.name}: {e}")

# --- Seed files ---
if seed_files:
    for f in seed_files:
        try:
            grouped = process_prescription(f, "seed")
            if not grouped.empty:
                layer_key = os.path.splitext(f.name)[0].lower().replace(" ", "_")
                st.session_state["seed_layers_store"][layer_key] = grouped
                st.success(f"✅ Seed file loaded: {f.name}")
        except Exception as e:
            st.warning(f"⚠️ Skipped seed file {f.name}: {e}")

# =========================================================
# FEEDBACK LISTS (summary of all uploads)
# =========================================================
if st.session_state["fert_layers_store"]:
    st.info("### Loaded Fertilizer Maps")
    for k in st.session_state["fert_layers_store"].keys():
        st.markdown(f"- {k}")

if st.session_state["seed_layers_store"]:
    st.info("### Loaded Seed Maps")
    for k in st.session_state["seed_layers_store"].keys():
        st.markdown(f"- {k}")


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
# 5. BASE MAP
# =========================================================
from branca.element import MacroElement, Template

def make_base_map():
    try:
        m = folium.Map(
            location=[39.5, -98.35],  # Continental US center
            zoom_start=5,
            min_zoom=2,
            tiles=None,
            scrollWheelZoom=False,
            prefer_canvas=True
        )

        # Base layers (safe guarded)
        try:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", overlay=False, control=False
            ).add_to(m)
        except Exception:
            pass
        try:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", overlay=True, control=False
            ).add_to(m)
        except Exception:
            pass

        # Enable scrollwheel only on click
        template = Template("""
            {% macro script(this, kwargs) %}
            var map = {{this._parent.get_name()}};
            map.scrollWheelZoom.disable();
            map.on('click', function() { map.scrollWheelZoom.enable(); });
            map.on('mouseout', function() { map.scrollWheelZoom.disable(); });
            {% endmacro %}
        """)
        macro = MacroElement()
        macro._template = template
        m.get_root().add_child(macro)
        return m
    except Exception as e:
        st.error(f"❌ Failed to build base map: {e}")
        return folium.Map(location=[39.5, -98.35], zoom_start=4)

# Init base map
m = make_base_map()
st.session_state["layer_control_added"] = False


# =========================================================
# 6. ZONES OVERLAY
# =========================================================
def add_zones_overlay(m):
    zones_gdf = st.session_state.get("zones_gdf")
    if zones_gdf is None or zones_gdf.empty:
        return m

    try:
        zones_gdf = zones_gdf.to_crs(epsg=4326)

        if "Zone" not in zones_gdf.columns:
            zones_gdf["Zone"] = range(1, len(zones_gdf) + 1)

        zb = zones_gdf.total_bounds
        m.location = [(zb[1] + zb[3]) / 2, (zb[0] + zb[2]) / 2]
        m.zoom_start = 15

        palette = ["#FF0000","#FF8C00","#FFFF00","#32CD32","#006400",
                   "#1E90FF","#8A2BE2","#FFC0CB","#A52A2A","#00CED1"]
        unique_vals = list(dict.fromkeys(sorted(list(zones_gdf["Zone"].astype(str).unique()))))
        color_map = {z: palette[i % len(palette)] for i, z in enumerate(unique_vals)}

        folium.GeoJson(
            zones_gdf,
            name="Zones",
            style_function=lambda feature: {
                "fillColor": color_map.get(str(feature["properties"].get("Zone","")), "#808080"),
                "color": "black", "weight": 1, "fillOpacity": 0.08,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["Zone","Calculated Acres","Override Acres"] if c in zones_gdf.columns]
            )
        ).add_to(m)

              # --- Zone legend bottom-right, collapsible ---
        unique_vals = list(dict.fromkeys(sorted(list(zones_gdf["Zone"].astype(str).unique()))))
        color_map = {z: palette[i % len(palette)] for i, z in enumerate(unique_vals)}

        legend_items = "".join([
            f"<div style='display:flex; align-items:center; margin:2px 0;'>"
            f"<div style='background:{color_map[z]}; width:14px; height:14px; margin-right:6px;'></div>{z}</div>"
            for z in unique_vals
        ])

   legend_html = f"""
<div id="zone-legend" style="position:absolute; bottom:20px; right:20px; z-index:9999;
             font-family:sans-serif; font-size:13px; color:white;
             background-color: rgba(0,0,0,0.65); padding:6px 10px; border-radius:5px;
             width:160px;">
  <div style="font-weight:600; margin-bottom:4px; cursor:pointer;" onclick="
      var x = document.getElementById('zone-legend-items');
      if (x.style.display === 'none') {{ x.style.display = 'block'; }} 
      else {{ x.style.display = 'none'; }}">
    Zone Colors ▼
  </div>
  <div id="zone-legend-items" style="display:block;">
    {legend_items}
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# =========================================================
# 7A. HELPERS + BOUNDS
# =========================================================
from matplotlib import colors as mpl_colors

def detect_rate_type(gdf):
    """Detect fixed vs variable rate from rate-like columns."""
    try:
        rate_col = None
        for c in gdf.columns:
            if "tgt" in c.lower() or "rate" in c.lower():
                rate_col = c; break
        if rate_col and gdf[rate_col].nunique(dropna=True) == 1:
            return "Fixed Rate"
    except Exception:
        pass
    return "Variable Rate"

def compute_bounds_for_heatmaps():
    """Safely collect bounds from all available layers."""
    try:
        bnds = []
        for key in ["zones_gdf","seed_gdf"]:
            g = st.session_state.get(key)
            if g is not None and not g.empty:
                tb = g.total_bounds
                if tb is not None and len(tb) == 4 and not any(pd.isna(tb)):
                    bnds.append([[tb[1], tb[0]], [tb[3], tb[2]]])
        for _k, fg in st.session_state.get("fert_gdfs", {}).items():
            if fg is not None and not fg.empty:
                tb = fg.total_bounds
                if tb is not None and len(tb) == 4 and not any(pd.isna(tb)):
                    bnds.append([[tb[1], tb[0]], [tb[3], tb[2]]])
        ydf = st.session_state.get("yield_df")
        if ydf is not None and not ydf.empty and {"Latitude","Longitude"}.issubset(ydf.columns):
            bnds.append([[ydf["Latitude"].min(), ydf["Longitude"].min()],
                         [ydf["Latitude"].max(), ydf["Longitude"].max()]])
        if bnds:
            south = min(b[0][0] for b in bnds)
            west  = min(b[0][1] for b in bnds)
            north = max(b[1][0] for b in bnds)
            east  = max(b[1][1] for b in bnds)
            return south, west, north, east
    except Exception:
        pass
    return 25.0, -125.0, 49.0, -66.0  # fallback

def safe_fit_bounds(m, bounds):
    try:
        south, west, north, east = bounds
        m.fit_bounds([[south, west],[north, east]])
    except Exception:
        pass


# =========================================================
# 7B. PRESCRIPTION OVERLAYS (Seed + Fert) + Gradient Legends
# =========================================================
def add_gradient_legend(name, vmin, vmax, cmap, index):
    """Add a gradient legend of consistent size, stacked in the top-left."""
    top_offset = 20 + (index * 80)
    stops = [f"{mpl_colors.rgb2hex(cmap(i/100.0)[:3])} {i}%" for i in range(0, 101, 10)]
    gradient_css = ", ".join(stops)
    legend_html = f"""
    <div style="position:absolute; top:{top_offset}px; left:10px; z-index:9999;
                font-family:sans-serif; font-size:12px; color:white;
                background-color: rgba(0,0,0,0.65); padding:6px 10px; border-radius:5px;
                width:180px;">
      <div style="font-weight:600; margin-bottom:4px;">{name}</div>
      <div style="height:14px; background:linear-gradient(90deg, {gradient_css});
                  border-radius:2px; margin-bottom:4px;"></div>
      <div style="display:flex; justify-content:space-between;">
        <span>{vmin:.1f}</span><span>{vmax:.1f}</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def infer_unit(gdf, rate_col, product_col):
    for cand in ["unit", "units", "uom", "rate_uom", "rateunit", "rate_unit"]:
        if cand in gdf.columns:
            vals = gdf[cand].dropna().astype(str).str.strip()
            if not vals.empty and vals.iloc[0] != "":
                return vals.iloc[0]

    rc = str(rate_col or "").lower()
    if any(k in rc for k in ["gpa", "gal", "uan"]): return "gal/ac"
    if any(k in rc for k in ["lb", "lbs", "dry", "nh3", "ammonia"]): return "lb/ac"
    if "kg" in rc: return "kg/ha"
    if any(k in rc for k in ["seed", "pop", "plant", "ksds", "kseed", "kseeds"]):
        try:
            med = pd.to_numeric(gdf[rate_col], errors="coerce").median()
            if 10 <= float(med) <= 90:
                return "k seeds/ac"
        except Exception:
            pass
        return "seeds/ac"

    prod_val = ""
    if product_col and product_col in gdf.columns:
        try:
            prod_val = str(gdf[product_col].dropna().astype(str).iloc[0]).lower()
        except Exception:
            prod_val = ""
    if "uan" in prod_val or "10-34-0" in prod_val: return "gal/ac"
    return None


def add_prescription_overlay(gdf, name, cmap, index):
    """Add Seed/Fert prescription polygons with gradient fill + tooltip + legend."""
    if gdf is None or gdf.empty:
        return

    gdf = gdf.copy()

    # detect columns
    product_col, rate_col = None, None
    for c in gdf.columns:
        cl = str(c).lower()
        if product_col is None and "product" in cl:
            product_col = c
        if rate_col is None and ("tgt" in cl or "rate" in cl):
            rate_col = c

    gdf["RateType"] = detect_rate_type(gdf)

    if rate_col and pd.to_numeric(gdf[rate_col], errors="coerce").notna().any():
        vals = pd.to_numeric(gdf[rate_col], errors="coerce").dropna()
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmin == vmax:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    unit = infer_unit(gdf, rate_col, product_col)
    rate_alias = f"Target Rate ({unit})" if unit else "Target Rate"
    legend_name = f"{name} — {rate_alias}"

    def style_fn(feat):
        val = feat["properties"].get(rate_col) if rate_col else None
        if val is None or pd.isna(val):
            fill = "#808080"
        else:
            try:
                norm = (float(val) - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                norm = max(0.0, min(1.0, norm))
                fill = mpl_colors.rgb2hex(cmap(norm)[:3])
            except Exception:
                fill = "#808080"
        return {"stroke": False, "opacity": 0, "weight": 0,
                "fillColor": fill, "fillOpacity": 0.55}

    fields, aliases = [], []
    if product_col: fields.append(product_col); aliases.append("Product")
    if rate_col:    fields.append(rate_col);    aliases.append(rate_alias)
    fields.append("RateType"); aliases.append("Type")

    folium.GeoJson(
        gdf,
        name=name,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases)
    ).add_to(m)

    add_gradient_legend(legend_name, vmin, vmax, cmap, index)


# --- Draw prescription layers ---
st.session_state["legend_index"] = 0  # unified counter start

seed_gdf    = st.session_state.get("seed_gdf")
fert_layers = st.session_state.get("fert_gdfs", {})

if seed_gdf is not None and not seed_gdf.empty:
    add_prescription_overlay(seed_gdf, "Seed RX", plt.cm.Greens, st.session_state["legend_index"])
    st.session_state["legend_index"] += 1

for k, fgdf in fert_layers.items():
    if fgdf is not None and not fgdf.empty:
        add_prescription_overlay(fgdf, f"Fertilizer RX: {k}", plt.cm.Blues, st.session_state["legend_index"])
        st.session_state["legend_index"] += 1


# =========================================================
# 7C. YIELD + PROFIT HEATMAPS (Crash-Proof)
# =========================================================
from scipy.interpolate import griddata

def add_heatmap_overlay(df, values, name, cmap, show_default, bounds):
    """Add a rasterized heatmap overlay to folium map."""
    try:
        if df is None or df.empty:
            return None, None
        south, west, north, east = bounds
        vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        if vals.empty: return None, None
        mask = (df[["Latitude", "Longitude"]].applymap(np.isfinite).all(axis=1)) & vals.notna()
        if mask.sum() < 3: return None, None

        pts_lon = df.loc[mask, "Longitude"].astype(float).values
        pts_lat = df.loc[mask, "Latitude"].astype(float).values
        vals_ok = vals.loc[mask].astype(float).values

        n = 200
        lon_lin = np.linspace(west, east, n)
        lat_lin = np.linspace(south, north, n)
        lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)

        grid_lin = griddata((pts_lon, pts_lat), vals_ok, (lon_grid, lat_grid), method="linear")
        grid_nn  = griddata((pts_lon, pts_lat), vals_ok, (lon_grid, lat_grid), method="nearest")
        grid = np.where(np.isnan(grid_lin), grid_nn, grid_lin)
        if grid is None or np.all(np.isnan(grid)): return None, None

        vmin = float(np.nanpercentile(vals_ok, 5)) if len(vals_ok) > 0 else 0.0
        vmax = float(np.nanpercentile(vals_ok, 95)) if len(vals_ok) > 0 else 1.0
        if vmin == vmax: vmax = vmin + 1.0

        rgba = cmap((grid - vmin) / (vmax - vmin))
        rgba = np.flipud(rgba)
        rgba = (rgba * 255).astype(np.uint8)

        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[south, west],[north, east]],
            opacity=0.5,
            name=name,
            overlay=True,
            show=show_default
        ).add_to(m)
        return vmin, vmax
    except Exception as e:
        st.warning(f"⚠️ Skipping heatmap {name}: {e}")
        return None, None


# --- MAIN EXECUTION FOR HEATMAPS ---
bounds = compute_bounds_for_heatmaps()
df = None
yield_df = st.session_state.get("yield_df")
if yield_df is not None and not yield_df.empty and "Yield" in yield_df.columns and {"Latitude","Longitude"}.issubset(yield_df.columns):
    df = yield_df.copy()

if df is None or df.empty:
    lat_center = (bounds[0] + bounds[2]) / 2.0
    lon_center = (bounds[1] + bounds[3]) / 2.0
    target_yield = st.number_input("Set Target Yield (bu/ac)", min_value=0.0, value=200.0, step=1.0)
    df = pd.DataFrame({"Yield":[target_yield],"Latitude":[lat_center],"Longitude":[lon_center]})

try:
    df["Revenue_per_acre"] = df["Yield"].astype(float) * float(sell_price or 0)
    fert_var = float(st.session_state["fert_products"]["CostPerAcre"].sum()) if not st.session_state["fert_products"].empty else 0.0
    seed_var = float(st.session_state["seed_products"]["CostPerAcre"].sum()) if not st.session_state["seed_products"].empty else 0.0
    df["NetProfit_per_acre_variable"] = df["Revenue_per_acre"] - (float(base_expenses_per_acre or 0) + fert_var + seed_var)
    fixed_costs = 0.0
    if "fixed_products" in st.session_state and not st.session_state["fixed_products"].empty:
        fx = st.session_state["fixed_products"].copy()
        fx["$/ac"] = fx.apply(lambda r: (r.get("Rate",0) or 0)*(r.get("CostPerUnit",0) or 0), axis=1)
        fixed_costs = float(fx["$/ac"].sum())
    df["NetProfit_per_acre_fixed"] = df["Revenue_per_acre"] - (float(base_expenses_per_acre or 0) + fixed_costs)
except Exception:
    st.warning("⚠️ Could not compute profit metrics, using defaults.")
    df["Revenue_per_acre"] = 0.0
    df["NetProfit_per_acre_variable"] = 0.0
    df["NetProfit_per_acre_fixed"] = 0.0

# --- Overlays with legends (unified stacking) ---
if "legend_index" not in st.session_state:
    st.session_state["legend_index"] = 0

y_min, y_max = add_heatmap_overlay(df, df["Yield"].values, "Yield (bu/ac)", plt.cm.RdYlGn, False, bounds)
if y_min is not None:
    add_gradient_legend("Yield (bu/ac)", y_min, y_max, plt.cm.RdYlGn, st.session_state["legend_index"])
    st.session_state["legend_index"] += 1

v_min, v_max = add_heatmap_overlay(df, df["NetProfit_per_acre_variable"].values, "Variable Rate Profit ($/ac)", plt.cm.RdYlGn, True, bounds)
if v_min is not None:
    add_gradient_legend("Variable Rate Profit ($/ac)", v_min, v_max, plt.cm.RdYlGn, st.session_state["legend_index"])
    st.session_state["legend_index"] += 1

f_min, f_max = add_heatmap_overlay(df, df["NetProfit_per_acre_fixed"].values, "Fixed Rate Profit ($/ac)", plt.cm.RdYlGn, False, bounds)
if f_min is not None:
    add_gradient_legend("Fixed Rate Profit ($/ac)", f_min, f_max, plt.cm.RdYlGn, st.session_state["legend_index"])
    st.session_state["legend_index"] += 1


# =========================================================
# 7D. LAYER CONTROL
# =========================================================
try:
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
except Exception:
    pass


# =========================================================
# 8. DISPLAY MAP
# =========================================================
safe_fit_bounds(m, compute_bounds_for_heatmaps())
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
