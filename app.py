# =========================================================
# Farm Profit Mapping Tool V4  — Ultra-Compact (Functionality Intact)
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
import tempfile
from branca.element import MacroElement, Template
from matplotlib import colors as mpl_colors

st.set_page_config(page_title="Farm ROI Tool V4", layout="wide")
st.title("Farm Profit Mapping Tool V4")

# =========================================================
# GLOBAL COMPACT UI STYLING
# =========================================================
st.markdown(
    """
    <style>
    /* Tighten column gutters (left/right padding) */
    div[data-testid="column"] { padding-left: .15rem !important; padding-right: .15rem !important; }

    /* Reduce vertical spacing between stacked blocks */
    section[data-testid="stVerticalBlock"] > div { padding-top: .15rem !important; padding-bottom: .15rem !important; }

    /* Compact headers */
    h1 { margin: .25rem 0 .25rem 0 !important; font-size: 1.2rem !important; }
    h2, h3 { margin: .2rem 0 .15rem 0 !important; font-size: 1rem !important; }

    /* Super-compact expander headers and bodies */
    div[data-testid="stExpander"] details summary { padding: .2rem .4rem !important; font-size: .8rem !important; }
    div[data-testid="stExpander"] details > div { padding: .2rem .4rem !important; }

    /* Compact number inputs (height + font + +/- buttons) */
    div[data-testid="stNumberInput"] label { font-size: .72rem !important; margin-bottom: .05rem !important; }
    div[data-testid="stNumberInput"] div[role="spinbutton"] {
        min-height: 22px !important; height: 22px !important; padding: 0 4px !important; font-size: .75rem !important;
    }
    div[data-testid="stNumberInput"] button { padding: 0 !important; min-width: 18px !important; }

    /* Compact select/text inputs */
    div[data-baseweb="select"] div[role="combobox"],
    input[type="text"] { min-height: 22px !important; height: 22px !important; font-size: .8rem !important; }

    /* File uploaders */
    div[data-testid="stFileUploaderDropzone"] { padding: .2rem !important; min-height: 32px !important; }
    div[data-testid="stFileUploaderDropzone"] p { font-size: .65rem !important; margin: 0 !important; }

    /* Dataframes / editors (Excel-like row height) */
    div[data-testid="stDataFrame"] table { font-size: .75rem !important; }
    div[data-testid="stDataFrame"] th, div[data-testid="stDataFrame"] td {
        padding: 1px 4px !important; line-height: 1rem !important;
    }
    div[data-testid="stDataEditor"] table { font-size: .75rem !important; }
    div[data-testid="stDataEditor"] th, div[data-testid="stDataEditor"] td {
        padding: 1px 4px !important; line-height: 1rem !important;
    }

    /* Captions tiny */
    div[data-testid="stCaptionContainer"] { margin: .1rem 0 !important; font-size: .7rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)
# Make any remaining number inputs physically narrow
st.markdown(
    """
    <style>
    /* Narrow number inputs site-wide (corn/soy assumptions, etc.) */
    div[data-testid="stNumberInput"] { width: 132px !important; max-width: 132px !important; }
    div[data-testid="stNumberInput"] div[role="spinbutton"] {
        min-height: 20px !important; height: 20px !important; padding: 0 4px !important; font-size: .78rem !important;
    }
    div[data-testid="stNumberInput"] button { padding: 0 !important; min-width: 16px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HELPERS
# =========================================================
def load_vector_file(uploaded_file):
    try:
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
                    return None
                gdf = gpd.read_file(shp_path)
        elif uploaded_file.name.lower().endswith(".shp"):
            with tempfile.TemporaryDirectory() as tmpdir:
                shp_path = os.path.join(tmpdir, uploaded_file.name)
                with open(shp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                gdf = gpd.read_file(shp_path)
        else:
            return None

        if gdf is None or gdf.empty:
            return None
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception:
        return None


def process_prescription(file, prescrip_type="fertilizer"):
    if file is None:
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])
    try:
        if file.name.lower().endswith((".geojson",".json",".zip",".shp")):
            gdf = load_vector_file(file)
            if gdf is None or gdf.empty:
                st.error(f"❌ Could not read {prescrip_type} map.")
                return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])
            gdf.columns = [c.strip().lower().replace(" ", "_") for c in gdf.columns]
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            else:
                gdf = gdf.to_crs(epsg=4326)
            gdf["Longitude"] = gdf.geometry.representative_point().x
            gdf["Latitude"]  = gdf.geometry.representative_point().y
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        else:
            df = pd.read_csv(file)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    except Exception as e:
        st.warning(f"⚠️ Failed to read {file.name}: {e}")
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])

    # normalize
    if "product" not in df.columns:
        for c in ["variety","hybrid","type","name","material"]:
            if c in df.columns:
                df.rename(columns={c: "product"}, inplace=True)
                break
        else:
            df["product"] = prescrip_type.capitalize()
    if "acres" not in df.columns:
        df["acres"] = 0.0

    # ultra-compact options expander (left as-is functionally)
    with st.expander(f"⚙️ {prescrip_type.capitalize()} Map Options — {file.name}", expanded=False):
        override = st.number_input(
            "Override Acres Per Polygon",
            min_value=0.0, value=0.0, step=0.1,
            key=f"{prescrip_type}_{file.name}_override"
        )
        if override > 0:
            df["acres"] = override

    if "costtotal" not in df.columns:
        if {"price_per_unit","units"}.issubset(df.columns):
            df["costtotal"] = df["price_per_unit"] * df["units"]
        elif {"rate","price"}.issubset(df.columns):
            df["costtotal"] = df["rate"] * df["price"]
        else:
            df["costtotal"] = 0

    if not df.empty:
        grouped = df.groupby("product", as_index=False).agg(
            Acres=("acres","sum"),
            CostTotal=("costtotal","sum")
        )
        grouped["CostPerAcre"] = grouped.apply(
            lambda r: r["CostTotal"]/r["Acres"] if r["Acres"]>0 else 0, axis=1
        )
        return grouped
    return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])


def _mini_num(label: str, key: str, default: float = 0.0, step: float = 0.1):
    st.caption(label)
    return st.number_input(key, min_value=0.0, value=float(default), step=step, label_visibility="collapsed")


# =========================================================
# 2–3. FILE UPLOADS — COMPACT 4-UP ROW
# =========================================================
st.markdown("### Upload Maps")

c1, c2, c3, c4 = st.columns(4)

# ---------------------------
# ZONE MAP (moved into c1)
# ---------------------------
with c1:
    zone_file = st.file_uploader("Zone", type=["geojson", "json", "zip"], key="zone_file", accept_multiple_files=False)
    if zone_file:
        try:
            zones_gdf = load_vector_file(zone_file)

            if zones_gdf is not None and not zones_gdf.empty:
                st.caption(f"✅ Zones: {len(zones_gdf)} polys")

                # Detect or create zone column
                zone_col = None
                for cand in ["Zone", "zone", "ZONE", "Name", "name"]:
                    if cand in zones_gdf.columns:
                        zone_col = cand
                        break
                if zone_col is None:
                    zones_gdf["ZoneIndex"] = range(1, len(zones_gdf) + 1)
                    zone_col = "ZoneIndex"
                zones_gdf["Zone"] = zones_gdf[zone_col]

                # Acre calculation
                gdf_area = zones_gdf.copy()
                if gdf_area.crs is None:
                    gdf_area.set_crs(epsg=4326, inplace=True)
                if gdf_area.crs.is_geographic:
                    gdf_area = gdf_area.to_crs(epsg=5070)  # Equal Area
                zones_gdf["Calculated Acres"] = (gdf_area.geometry.area * 0.000247105).astype(float)
                zones_gdf["Override Acres"]   = zones_gdf["Calculated Acres"].astype(float)

                # Compact override editor
                display_df = zones_gdf[["Zone", "Calculated Acres", "Override Acres"]].copy()
                edited = st.data_editor(
                    display_df,
                    num_rows="fixed",
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Zone": st.column_config.TextColumn(disabled=True),
                        "Calculated Acres": st.column_config.NumberColumn(format="%.2f", disabled=True),
                        "Override Acres": st.column_config.NumberColumn(format="%.2f"),
                    },
                    key="zone_acres_editor",
                )
                edited["Override Acres"] = pd.to_numeric(edited["Override Acres"], errors="coerce")
                edited["Override Acres"] = edited["Override Acres"].fillna(edited["Calculated Acres"])

                # Totals line under editor
                total_calc = float(zones_gdf["Calculated Acres"].sum())
                total_override = float(edited["Override Acres"].sum())
                st.caption(f"Total Acres — Calc: {total_calc:,.2f} | Override: {total_override:,.2f}")

                zones_gdf["Override Acres"] = edited["Override Acres"].astype(float).values
                st.session_state["zones_gdf"] = zones_gdf
            else:
                st.error("❌ Could not load zone map. File is empty or invalid.")
        except Exception as e:
            st.error(f"❌ Error processing zone map: {e}")
    else:
        st.caption("No zone file")

# ---------------------------
# YIELD MAPS (c2)
# ---------------------------
with c2:
    yield_files = st.file_uploader("Yield", type=["csv", "geojson", "json", "zip"], key="yield", accept_multiple_files=True)
    st.session_state.setdefault("yield_files_list", [])

    if yield_files:
        st.session_state["yield_files_list"].clear()
        summary = []
        for yf in yield_files:
            try:
                df_temp = None
                if yf.name.lower().endswith(".csv"):
                    df_temp = pd.read_csv(yf)
                else:
                    gdf_temp = load_vector_file(yf)
                    if gdf_temp is not None and not gdf_temp.empty:
                        df_temp = pd.DataFrame(gdf_temp.drop(columns="geometry", errors="ignore"))

                if df_temp is not None and not df_temp.empty:
                    df_temp.columns = [c.strip().lower().replace(" ", "_") for c in df_temp.columns]
                    yield_cols = [c for c in df_temp.columns if any(k in c for k in ["yld_vol_dr","yld_mass_dr","yield_dry","dry_yield"])]
                    if not yield_cols:
                        yield_cols = [c for c in df_temp.columns if any(k in c for k in ["yield","yld_vol_wt","yld_mass_wt","wet_yield"])]
                    if yield_cols:
                        df_temp.rename(columns={yield_cols[0]: "Yield"}, inplace=True)
                    else:
                        df_temp["Yield"] = 0.0

                    summary.append({"File": yf.name, "Rows": len(df_temp)})
                else:
                    st.warning(f"⚠️ {yf.name} had no usable data.")
            except Exception as e:
                st.warning(f"⚠️ Skipped {yf.name}: {e}")

        if summary:
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    else:
        st.caption("No yield files")

st.session_state["yield_df"] = None

# ---------------------------
# FERTILIZER MAPS (c3)
# ---------------------------
with c3:
    fert_files = st.file_uploader("Fert", type=["csv","geojson","json","zip"], key="fert", accept_multiple_files=True)
    st.session_state["fert_layers_store"] = {}
    if fert_files:
        summary = []
        for f in fert_files:
            grouped = process_prescription(f, "fertilizer")
            if not grouped.empty:
                key = os.path.splitext(f.name)[0].lower().replace(" ", "_")
                st.session_state["fert_layers_store"][key] = grouped
                summary.append({"File": f.name, "Products": len(grouped)})
        if summary:
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    else:
        st.caption("No fert files")

# ---------------------------
# SEED MAPS (c4)
# ---------------------------
with c4:
    seed_files = st.file_uploader("Seed", type=["csv","geojson","json","zip"], key="seed", accept_multiple_files=True)
    st.session_state["seed_layers_store"] = {}
    if seed_files:
        summary = []
        for f in seed_files:
            grouped = process_prescription(f, "seed")
            if not grouped.empty:
                key = os.path.splitext(f.name)[0].lower().replace(" ", "_")
                st.session_state["seed_layers_store"][key] = grouped
                summary.append({"File": f.name, "Products": len(grouped)})
        if summary:
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    else:
        st.caption("No seed files")

# =========================================================
# 4A. Expense Inputs (Ultra-compact grid editor)
#    - Drop-in replacement for the 12 number inputs
#    - Keeps outputs: `expenses` dict and `base_expenses_per_acre` float
# =========================================================

st.markdown("#### 4A. Expense Inputs (Per Acre $)")

# Build defaults once, then preserve user edits in session
_default_expense_rows = [
    ("Chemicals", 0.0),
    ("Insurance", 0.0),
    ("Insecticide/Fungicide", 0.0),
    ("Fertilizer (Flat)", 0.0),
    ("Seed (Flat)", 0.0),
    ("Cash Rent", 0.0),
    ("Machinery", 0.0),
    ("Labor", 0.0),
    ("Cost of Living", 0.0),
    ("Extra Fuel", 0.0),
    ("Extra Interest", 0.0),
    ("Truck Fuel", 0.0),
]

# Initialize or refresh structure if needed
if "exp_df" not in st.session_state:
    st.session_state["exp_df"] = pd.DataFrame(_default_expense_rows, columns=["Expense", "$/ac"])
else:
    # Ensure we have the same 12 rows in correct order (robust to older state)
    cur = st.session_state["exp_df"]
    names = [r[0] for r in _default_expense_rows]
    if set(cur["Expense"].tolist()) != set(names):
        st.session_state["exp_df"] = pd.DataFrame(_default_expense_rows, columns=["Expense", "$/ac"])

# Render super-compact data editor (no scroll)
exp_df = st.data_editor(
    st.session_state["exp_df"],
    hide_index=True,
    num_rows="fixed",
    use_container_width=True,
    key="exp_editor",
    column_config={
        "Expense": st.column_config.TextColumn(disabled=True),
        "$/ac": st.column_config.NumberColumn(format="%.2f", step=1.0, help="Per-acre cost"),
    },
    height=df_height(12, row_h=24, header=30, pad=0),  # exact height = no internal scroll
)

# Sanitize numeric and persist
exp_df["$/ac"] = pd.to_numeric(exp_df["$/ac"], errors="coerce").fillna(0.0)
st.session_state["exp_df"] = exp_df.copy()

# Outputs used elsewhere in your app (unchanged names/types)
expenses = dict(zip(exp_df["Expense"], exp_df["$/ac"]))
base_expenses_per_acre = float(exp_df["$/ac"].sum())



# ---- 4B–4D split layout (left: products, right: crop assumptions + preview) ----
left, right = st.columns([1, 1])

with left:
    with st.expander("Fixed Rate Inputs", expanded=False):
        if "fixed_products" not in st.session_state or st.session_state["fixed_products"].empty:
            st.session_state["fixed_products"] = pd.DataFrame(
                {"Type":["Seed","Fertilizer"], "Product":["",""], "Rate":[0.0,0.0],
                 "CostPerUnit":[0.0,0.0], "$/ac":[0.0,0.0]}
            )
        fixed_entries = st.data_editor(
            st.session_state["fixed_products"], num_rows="dynamic",
            use_container_width=True, key="fixed_editor"
        )
        st.session_state["fixed_products"] = fixed_entries.copy().reset_index(drop=True)

    with st.expander("Variable Rate Inputs", expanded=False):
        fert_df = st.session_state.get("fert_products", pd.DataFrame())
        seed_df = st.session_state.get("seed_products", pd.DataFrame())
        if not fert_df.empty:
            st.caption("Fertilizer (VR)")
            st.dataframe(fert_df, use_container_width=True, hide_index=True)
        if not seed_df.empty:
            st.caption("Seed (VR)")
            st.dataframe(seed_df, use_container_width=True, hide_index=True)
        if fert_df.empty and seed_df.empty:
            st.caption("— No VR inputs —")

with right:
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Corn Yield (bu/ac)")
        st.session_state["corn_yield"] = st.number_input(
            "corn_yld", min_value=0.0, value=st.session_state.get("corn_yield", 200.0),
            step=1.0, label_visibility="collapsed"
        )
        st.caption("Corn Price ($/bu)")
        st.session_state["corn_price"] = st.number_input(
            "corn_px", min_value=0.0, value=st.session_state.get("corn_price", 5.0),
            step=0.1, label_visibility="collapsed"
        )
    with c2:
        st.caption("Soy Yield (bu/ac)")
        st.session_state["bean_yield"] = st.number_input(
            "bean_yld", min_value=0.0, value=st.session_state.get("bean_yield", 60.0),
            step=1.0, label_visibility="collapsed"
        )
        st.caption("Soy Price ($/bu)")
        st.session_state["bean_price"] = st.number_input(
            "bean_px", min_value=0.0, value=st.session_state.get("bean_price", 12.0),
            step=0.1, label_visibility="collapsed"
        )

    # Compact preview table (linked to session_state values)
    preview_df = pd.DataFrame({
        "Crop": ["Corn", "Soybeans"],
        "Yield": [st.session_state["corn_yield"], st.session_state["bean_yield"]],
        "Price": [st.session_state["corn_price"], st.session_state["bean_price"]],
        "Revenue": [
            st.session_state["corn_yield"] * st.session_state["corn_price"],
            st.session_state["bean_yield"] * st.session_state["bean_price"]
        ],
        "Fixed": [base_expenses_per_acre, base_expenses_per_acre],
    })
    preview_df["Breakeven"] = preview_df["Revenue"] - preview_df["Fixed"]

    st.dataframe(
        preview_df.style.format({
            "Yield":"{:.0f}", "Price":"${:.2f}",
            "Revenue":"${:,.0f}", "Fixed":"${:,.0f}", "Breakeven":"${:,.0f}"
        }),
        use_container_width=True, hide_index=True
    )


# =========================================================
# 5. BASE MAP  (UNCHANGED FUNCTIONALITY)
# =========================================================
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
# 6. ZONES OVERLAY  (UNCHANGED)
# =========================================================
def add_zones_overlay(m):
    zones_gdf = st.session_state.get("zones_gdf")
    if zones_gdf is None or zones_gdf.empty:
        return m

    try:
        # Ensure projection
        zones_gdf = zones_gdf.to_crs(epsg=4326)

        # Guarantee 'Zone' column
        if "Zone" not in zones_gdf.columns:
            zones_gdf["Zone"] = range(1, len(zones_gdf) + 1)

        # Auto center and zoom
        zb = zones_gdf.total_bounds
        m.location = [(zb[1] + zb[3]) / 2, (zb[0] + zb[2]) / 2]
        m.zoom_start = 15

        # Color mapping
        palette = ["#FF0000","#FF8C00","#FFFF00","#32CD32","#006400",
                   "#1E90FF","#8A2BE2","#FFC0CB","#A52A2A","#00CED1"]
        unique_vals = list(dict.fromkeys(sorted(list(zones_gdf["Zone"].astype(str).unique()))))
        color_map = {z: palette[i % len(palette)] for i, z in enumerate(unique_vals)}

        # Add overlay
        folium.GeoJson(
            zones_gdf,
            name="Zones",
            style_function=lambda feature: {
                "fillColor": color_map.get(str(feature["properties"].get("Zone", "")), "#808080"),
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.08,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["Zone","Calculated Acres","Override Acres"] if c in zones_gdf.columns]
            )
        ).add_to(m)

        # Legend
        legend_items = ""
        for z in unique_vals:
            legend_items += (
                f"<div style='display:flex; align-items:center; margin:2px 0;'>"
                f"<div style='background:{color_map[z]}; width:14px; height:14px; margin-right:6px;'></div>"
                f"{z}</div>"
            )
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

    except Exception as e:
        st.warning(f"⚠️ Skipping zones overlay: {e}")

    return m

# =========================================================
# 7A. HELPERS + BOUNDS  (UNCHANGED)
# =========================================================
def detect_rate_type(gdf):
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
# 7B. PRESCRIPTION OVERLAYS + LEGENDS  (UNCHANGED)
# =========================================================
def add_gradient_legend(name, vmin, vmax, cmap, index):
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

# --- Draw prescription layers (UNCHANGED) ---
st.session_state["legend_index"] = 0

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
# 7C. YIELD + PROFIT HEATMAPS (Crash-Proof, Compact Fixes)
# =========================================================
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
bounds = st.session_state.get("map_bounds", compute_bounds_for_heatmaps())
df = None
yield_df = st.session_state.get("yield_df")
if yield_df is not None and not yield_df.empty and "Yield" in yield_df.columns and {"Latitude","Longitude"}.issubset(yield_df.columns):
    df = yield_df.copy()

# Only show Target Yield when no yield map present
if df is None or df.empty:
    lat_center = (bounds[0] + bounds[2]) / 2.0
    lon_center = (bounds[1] + bounds[3]) / 2.0
    target_yield = st.number_input("Set Target Yield (bu/ac)", min_value=0.0, value=200.0, step=1.0)
    df = pd.DataFrame({"Yield":[target_yield],"Latitude":[lat_center],"Longitude":[lon_center]})

# IMPORTANT FIX: define sell_price even though 4A removed it (use corn price assumption)
sell_price = float(st.session_state.get("corn_price", 5.0))

try:
    df["Revenue_per_acre"] = df["Yield"].astype(float) * sell_price
    fert_var = float(st.session_state.get("fert_products", pd.DataFrame(columns=["CostPerAcre"]))["CostPerAcre"].sum()) \
               if "fert_products" in st.session_state and not st.session_state["fert_products"].empty else 0.0
    seed_var = float(st.session_state.get("seed_products", pd.DataFrame(columns=["CostPerAcre"]))["CostPerAcre"].sum()) \
               if "seed_products" in st.session_state and not st.session_state["seed_products"].empty else 0.0
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

# --- Final map fit ---
bounds = compute_bounds_for_heatmaps()
safe_fit_bounds(m, bounds)

# =========================================================
# 7D. LAYER CONTROL  (UNCHANGED)
# =========================================================
try:
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
except Exception:
    pass

# =========================================================
# 8. DISPLAY MAP  (UNCHANGED)
# =========================================================
safe_fit_bounds(m, compute_bounds_for_heatmaps())
st_folium(m, use_container_width=True, height=550)

# =========================================================
# 9. PROFIT SUMMARY  — COMPACT, SAME FUNCTIONALITY
# =========================================================
st.header("Profit Summary")

# Ensure session state keys exist
if "fert_products" not in st.session_state:
    st.session_state["fert_products"] = pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])
if "seed_products" not in st.session_state:
    st.session_state["seed_products"] = pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])
if "zones_gdf" not in st.session_state:
    st.session_state["zones_gdf"] = None
if "yield_df" not in st.session_state:
    st.session_state["yield_df"] = None

# Safe defaults
revenue_per_acre = 0.0
net_profit_per_acre = 0.0
expenses_per_acre = base_expenses_per_acre if "base_expenses_per_acre" in locals() else 0.0

if st.session_state["yield_df"] is not None and not st.session_state["yield_df"].empty:
    df_y = st.session_state["yield_df"]
    if "Revenue_per_acre" in df_y.columns:
        revenue_per_acre = df_y["Revenue_per_acre"].mean()
    if "NetProfit_per_acre" in df_y.columns:
        net_profit_per_acre = df_y["NetProfit_per_acre"].mean()

# Layout (compact side-by-side)
col_left, col_right = st.columns([1.6, 1])

# LEFT: Breakeven + Profit Comparison
with col_left:
    # Pull values from session_state (set in Section 4)
    corn_yield = st.session_state.get("corn_yield", 200.0)
    corn_price = st.session_state.get("corn_price", 5.0)
    bean_yield = st.session_state.get("bean_yield", 60.0)
    bean_price = st.session_state.get("bean_price", 12.0)

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

    def highlight_budget(val):
        if isinstance(val, (int, float)):
            if val > 0: return "color: limegreen; font-weight: 600;"
            if val < 0: return "color: crimson; font-weight: 600;"
        return "font-weight: 600;"

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
    hide_index=True,
    height=df_height(len(breakeven_df))   # <-- no scroll
)

    # Profit Metrics Comparison
    var_profit = 0.0
    if st.session_state["yield_df"] is not None and not st.session_state["yield_df"].empty:
        df_y = st.session_state["yield_df"]
        fert_costs = st.session_state["fert_products"]["CostPerAcre"].sum() if not st.session_state["fert_products"].empty else 0
        seed_costs = st.session_state["seed_products"]["CostPerAcre"].sum() if not st.session_state["seed_products"].empty else 0
        revenue_var = df_y["Revenue_per_acre"].mean() if "Revenue_per_acre" in df_y.columns else 0.0
        expenses_var = base_expenses_per_acre + fert_costs + seed_costs
        var_profit = revenue_var - expenses_var
    else:
        revenue_var, expenses_var = 0.0, 0.0

    fixed_profit = 0.0
    if "fixed_products" in st.session_state and not st.session_state["fixed_products"].empty:
        fert_fixed_costs = st.session_state["fixed_products"][st.session_state["fixed_products"]["Type"]=="Fertilizer"]["$/ac"].sum()
        seed_fixed_costs = st.session_state["fixed_products"][st.session_state["fixed_products"]["Type"]=="Seed"]["$/ac"].sum()
        revenue_fixed = revenue_var
        expenses_fixed = base_expenses_per_acre + fert_fixed_costs + seed_fixed_costs
        fixed_profit = revenue_fixed - expenses_fixed
    else:
        revenue_fixed, expenses_fixed = 0.0, 0.0

    revenue_overall = revenue_per_acre
    expenses_overall = expenses_per_acre
    profit_overall = net_profit_per_acre

    comparison = pd.DataFrame({
        "Metric": ["Revenue ($/ac)", "Expenses ($/ac)", "Profit ($/ac)"],
        "Breakeven Budget": [round(revenue_overall,2), round(expenses_overall,2), round(profit_overall,2)],
        "Variable Rate": [round(revenue_var,2), round(expenses_var,2), round(var_profit,2)],
        "Fixed Rate": [round(revenue_fixed,2), round(expenses_fixed,2), round(fixed_profit,2)]
    })

    def highlight_profit(val):
        if isinstance(val, (int, float)):
            if val > 0: return "color: limegreen; font-weight: 600;"
            if val < 0: return "color: crimson; font-weight: 600;"
        return "font-weight: 600;"

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
    hide_index=True,
    height=df_height(len(comparison))     # <-- no scroll
)


    with st.expander("Show Calculation Formulas", expanded=False):
        st.markdown("""
        <div style="font-size:.85rem;">
        • <b>Breakeven Budget</b> = (Target Yield × Sell Price) − Fixed Inputs<br/>
        • <b>Variable Rate</b> = (Avg Yield × Sell Price) − (Fixed Inputs + Var Seed + Var Fert)<br/>
        • <b>Fixed Rate</b> = (Avg Yield × Sell Price) − (Fixed Inputs + Fixed Seed + Fixed Fert)
        </div>
        """, unsafe_allow_html=True)

# RIGHT: Fixed Inputs
fixed_df = pd.concat([fixed_df, total_fixed], ignore_index=True)

styled_fixed = fixed_df.style.format({"$/ac": "${:,.2f}"}).apply(
    lambda x: ["font-weight: bold;" if v == "Total Fixed Costs" else "" for v in x],
    subset=["Expense"]
).apply(
    lambda x: ["font-weight: bold;" if i == len(fixed_df) - 1 else "" for i in range(len(x))],
    subset=["$/ac"]
)

fixed_df = pd.concat([fixed_df, total_fixed], ignore_index=True)

styled_fixed = fixed_df.style.format({"$/ac": "${:,.2f}"}).apply(
    lambda x: ["font-weight: bold;" if v == "Total Fixed Costs" else "" for v in x],
    subset=["Expense"]
).apply(
    lambda x: ["font-weight: bold;" if i == len(fixed_df) - 1 else "" for i in range(len(x))],
    subset=["$/ac"]
)

st.dataframe(
    styled_fixed,
    use_container_width=True,
    hide_index=True,
    height=df_height(len(fixed_df))       # <-- replaces manual math
)


