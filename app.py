# =========================================================
# Farm Profit Mapping Tool V4 — COMPACT + BULLETPROOF
# =========================================================
import os
import zipfile
import tempfile
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
import streamlit as st
import folium
from streamlit_folium import st_folium
from branca.element import MacroElement, Template
from matplotlib import colors as mpl_colors

# =========================================================
# GLOBAL THEME + LAYOUT OVERRIDES (STREAMLIT v1.39+)
# =========================================================
st.markdown("""
<style>

/* ---------- GLOBAL BASE: PREVENT ALL SCROLL ---------- */
html, body, [data-testid="stAppViewContainer"], [data-testid="block-container"] {
    overflow-x: hidden !important;
    overflow-y: hidden !important;
}

/* ---------- MAP CONTAINER (85% WIDTH CENTERED) ---------- */
iframe[title="st_folium"] {
    max-width: 85vw !important;
    display: block;
    margin: 0 auto !important;
    border-radius: 8px;
}

/* ---------- SECTION 9: PROFIT SUMMARY (50% WIDTH CENTERED) ---------- */
div[data-testid="block-container"] h2:has(span:contains("Profit Summary")),
div[data-testid="block-container"] h3:has(span:contains("Profit Summary")) {
    text-align: center !important;
}

div[data-testid="block-container"] > div:has(h2:has(span:contains("Profit Summary"))) {
    max-width: 50vw !important;
    margin-left: auto !important;
    margin-right: auto !important;
    overflow: hidden !important;
    border-radius: 8px;
}

/* ---------- CORN vs SOY EXPANDER ---------- */
div[data-testid="stExpander"]:has(div:contains("Corn vs Soy")),
div[data-testid="stExpander"]:has(div:contains("Compare Crop Profitability")) {
    max-width: 50vw !important;
    margin-left: auto !important;
    margin-right: auto !important;
    border-radius: 8px;
    overflow: hidden !important;
}

/* ---------- TABLES CENTERED ---------- */
.stDataFrame {
    overflow: visible !important;
}
.stDataFrame table {
    margin-left: auto !important;
    margin-right: auto !important;
}

</style>
""", unsafe_allow_html=True)

# ===========================
# COMPACT THEME + LAYOUT
# ===========================
def apply_compact_theme():
    st.set_page_config(page_title="Farm ROI Tool V4", layout="wide")
    st.title("Farm Profit Mapping Tool V4")
# =========================================================
# GLOBAL STYLE OVERRIDES — THEME SAFE + BULLETPROOF
# =========================================================
st.markdown("""
<style>

/* ---------- THEME VARIABLES (auto adapts) ---------- */
:root {
  --card-bg: var(--secondary-background-color);
  --text-color: var(--text-color);
}

/* ---------- Section 9: Profit Summary ---------- */
section[data-testid="stVerticalBlock"]:has(h2:contains('Profit Summary')),
section[data-testid="stVerticalBlock"]:has(h3:contains('Profit Summary')) {
  max-width: 50vw !important;
  margin-left: auto !important;
  margin-right: auto !important;
  background-color: var(--card-bg) !important;
  border-radius: 8px !important;
  padding: 10px 16px 14px 16px !important;
  overflow: hidden !important;
  box-shadow: 0 0 12px rgba(0,0,0,0.25);
}

/* center all inner DataFrames + remove scrolls */
.stDataFrame {overflow: visible !important;}
.stDataFrame table {margin-left:auto;margin-right:auto;}
.stDataFrame [data-testid="stHorizontalBlock"] {
  justify-content: center !important;
}

/* ---------- Corn vs Soy collapsible chart ---------- */
div[data-testid="stExpander"]:has(div:contains('Corn vs Soy')) {
  max-width: 50vw !important;
  margin-left: auto !important;
  margin-right: auto !important;
  background-color: var(--card-bg) !important;
  border-radius: 8px !important;
  box-shadow: 0 0 8px rgba(0,0,0,0.3);
}
div[data-testid="stExpander"] summary {
  font-weight: 600 !important;
  color: var(--text-color) !important;
}
div[data-testid="stExpander"] > div[role="region"] {
  padding-top: 0.4rem !important;
  padding-bottom: 0.6rem !important;
}

/* ---------- Optional visual cohesion ---------- */
section[data-testid="stVerticalBlock"]:has(iframe[title="st_folium"]) {
  display: flex;
  justify-content: center;
}
iframe[title="st_folium"] {
  border-radius: 8px !important;
  max-width: 85vw !important;
  margin: 0 auto !important;
}

</style>
""", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
          .block-container{ padding-top:.28rem !important; }

          /* tighter column gutters */
          div[data-testid="column"]{
            padding-left:.22rem !important; padding-right:.22rem !important;
          }

          /* reduce vertical block spacing */
          section[data-testid="stVerticalBlock"] > div{
            padding-top:.18rem !important; padding-bottom:.18rem !important;
          }

          /* compact headers + captions */
          h1{ margin:.32rem 0 .28rem 0 !important; font-size:1.22rem !important; }
          h2,h3{ margin:.24rem 0 .16rem 0 !important; font-size:1.0rem !important; }
          div[data-testid="stCaptionContainer"]{ margin:.12rem 0 !important; }

          /* number inputs */
          div[data-testid="stNumberInput"] label{
            font-size:.78rem !important; margin-bottom:.10rem !important;
          }
          div[data-testid="stNumberInput"] div[role="spinbutton"]{
            min-height:26px !important; height:26px !important; padding:0 6px !important; font-size:.86rem !important;
          }
          div[data-testid="stNumberInput"] button{ padding:0 !important; min-width:20px !important; }

          /* make number boxes narrower so 12 fit on one line on most screens */
          div[data-testid="stNumberInput"]{ width:112px !important; max-width:112px !important; }

          /* uploader footprint */
          div[data-testid="stFileUploader"]{ margin-top:.10rem !important; }
          div[data-testid="stFileUploaderDropzone"]{ padding:.22rem !important; min-height:40px !important; }
          div[data-testid="stFileUploaderDropzone"] p{ margin:0 !important; font-size:.78rem !important; }

          /* compact DataFrame & DataEditor cells (no wasted space) */
          div[data-testid="stDataFrame"] table, div[data-testid="stDataEditor"] table{ font-size:.86rem !important; }
          div[data-testid="stDataFrame"] th, div[data-testid="stDataFrame"] td,
          div[data-testid="stDataEditor"] th, div[data-testid="stDataEditor"] td{
            padding:2px 6px !important; line-height:1.12rem !important;
          }

          /* compact static tables (st.table) */
          table { border-collapse:collapse; }
          thead th, tbody td { padding:4px 8px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ===========================
# HELPERS
# ===========================
def df_px_height(nrows: int, row_h: int = 28, header: int = 34, pad: int = 2) -> int:
    """Exact pixel height so tables/editors show with NO internal scroll."""
    return int(header + max(1, nrows) * row_h + pad)


def find_col(df: pd.DataFrame, names) -> Optional[str]:
    """Return first matching column (case-insensitive); supports underscores."""
    if df is None or df.empty:
        return None
    norm = {c.lower().replace(" ", "_"): c for c in df.columns}
    for n in names:
        key = n.lower().replace(" ", "_")
        if key in norm:
            return norm[key]
    return None


def load_vector_file(uploaded_file):
    """Read GeoJSON/JSON/ZIP(SHP)/SHP into EPSG:4326 GeoDataFrame."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith((".geojson", ".json")):
            gdf = gpd.read_file(uploaded_file)
        elif name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zpath = os.path.join(tmpdir, "in.zip")
                with open(zpath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(tmpdir)
                shp_path = None
                for fn in os.listdir(tmpdir):
                    if fn.lower().endswith(".shp"):
                        shp_path = os.path.join(tmpdir, fn)
                        break
                if not shp_path:
                    return None
                gdf = gpd.read_file(shp_path)
        elif name.endswith(".shp"):
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
    """
    Returns: (grouped_table, original_gdf_or_None)
    grouped_table columns: product, Acres, CostTotal, CostPerAcre
    """
    if file is None:
        return pd.DataFrame(columns=["product", "Acres", "CostTotal", "CostPerAcre"]), None
    try:
        name = file.name.lower()
        gdf_orig = None
        if name.endswith((".geojson", ".json", ".zip", ".shp")):
            gdf = load_vector_file(file)
            if gdf is None or gdf.empty:
                st.error(f"❌ Could not read {prescrip_type} map.")
                return pd.DataFrame(columns=["product", "Acres", "CostTotal", "CostPerAcre"]), None
            gdf.columns = [c.strip().lower().replace(" ", "_") for c in gdf.columns]
            gdf_orig = gdf.copy()
            gdf["Longitude"] = gdf.geometry.representative_point().x
            gdf["Latitude"] = gdf.geometry.representative_point().y
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        else:
            df = pd.read_csv(file)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        if "product" not in df.columns:
            for c in ["variety", "hybrid", "type", "name", "material"]:
                if c in df.columns:
                    df.rename(columns={c: "product"}, inplace=True)
                    break
            else:
                df["product"] = prescrip_type.capitalize()
        if "acres" not in df.columns:
            df["acres"] = 0.0

        if "costtotal" not in df.columns:
            if {"price_per_unit", "units"}.issubset(df.columns):
                df["costtotal"] = df["price_per_unit"] * df["units"]
            elif {"rate", "price"}.issubset(df.columns):
                df["costtotal"] = df["rate"] * df["price"]
            else:
                df["costtotal"] = 0.0

        if df.empty:
            return pd.DataFrame(columns=["product", "Acres", "CostTotal", "CostPerAcre"]), gdf_orig

        grouped = df.groupby("product", as_index=False).agg(
            Acres=("acres", "sum"), CostTotal=("costtotal", "sum")
        )
        grouped["CostPerAcre"] = grouped.apply(
            lambda r: r["CostTotal"] / r["Acres"] if r["Acres"] > 0 else 0, axis=1
        )
        return grouped, gdf_orig
    except Exception as e:
        st.warning(f"⚠️ Failed to read {file.name}: {e}")
        return pd.DataFrame(columns=["product", "Acres", "CostTotal", "CostPerAcre"]), None


def _bootstrap_defaults():
    """Ensure all keys exist so nothing crashes."""
    for k in ["chem", "ins", "insect", "fert", "seed", "rent", "mach", "labor", "col", "fuel", "int", "truck"]:
        st.session_state.setdefault(k, 0.0)
    st.session_state.setdefault("corn_yield", 200.0)
    st.session_state.setdefault("corn_price", 5.0)
    st.session_state.setdefault("bean_yield", 60.0)
    st.session_state.setdefault("bean_price", 12.0)
    st.session_state.setdefault("sell_price", st.session_state["corn_price"])
    st.session_state.setdefault("target_yield", 200.0)
    st.session_state.setdefault("fixed_products", pd.DataFrame(
        {"Type": ["Seed", "Fertilizer"], "Product": ["", ""], "Rate": [0.0, 0.0],
         "CostPerUnit": [0.0, 0.0], "$/ac": [0.0, 0.0]}
    ))
    st.session_state.setdefault("yield_df", pd.DataFrame())
    st.session_state.setdefault("fert_layers_store", {})
    st.session_state.setdefault("seed_layers_store", {})
    st.session_state.setdefault("fert_gdfs", {})
    st.session_state.setdefault("seed_gdf", None)
    st.session_state.setdefault("expenses_dict", {})
    st.session_state.setdefault("base_expenses_per_acre", 0.0)


# ===========================
# UI: Uploaders row + summaries
# ===========================
def render_uploaders():
    st.subheader("Upload Maps")

    u1, u2, u3, u4 = st.columns(4)

    # --- Zones
    with u1:
        st.caption("Zone Map · GeoJSON/JSON/ZIP(SHP)")
        zone_file = st.file_uploader("Zone", type=["geojson", "json", "zip"], key="up_zone", accept_multiple_files=False)
        if zone_file:
            zones_gdf = load_vector_file(zone_file)
            if zones_gdf is not None and not zones_gdf.empty:
                zone_col = None
                for cand in ["Zone", "zone", "ZONE", "Name", "name"]:
                    if cand in zones_gdf.columns:
                        zone_col = cand
                        break
                if zone_col is None:
                    zones_gdf["ZoneIndex"] = range(1, len(zones_gdf) + 1)
                    zone_col = "ZoneIndex"
                zones_gdf["Zone"] = zones_gdf[zone_col]

                # Equal-area acres
                g2 = zones_gdf.copy()
                if g2.crs is None:
                    g2.set_crs(epsg=4326, inplace=True)
                if g2.crs.is_geographic:
                    g2 = g2.to_crs(epsg=5070)
                zones_gdf["Calculated Acres"] = (g2.geometry.area * 0.000247105).astype(float)
                zones_gdf["Override Acres"] = zones_gdf["Calculated Acres"].astype(float)

                disp = zones_gdf[["Zone", "Calculated Acres", "Override Acres"]].copy()
                edited = st.data_editor(
                    disp,
                    num_rows="fixed",
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Zone": st.column_config.TextColumn(disabled=True),
                        "Calculated Acres": st.column_config.NumberColumn(format="%.2f", disabled=True),
                        "Override Acres": st.column_config.NumberColumn(format="%.2f"),
                    },
                    height=df_px_height(len(disp)),
                )
                edited["Override Acres"] = pd.to_numeric(edited["Override Acres"], errors="coerce") \
                    .fillna(edited["Calculated Acres"])
                zones_gdf["Override Acres"] = edited["Override Acres"].astype(float).values
                st.caption(
                    f"✅ Zones: {len(zones_gdf)}  |  Calc: {zones_gdf['Calculated Acres'].sum():,.2f} ac  |  "
                    f"Override: {zones_gdf['Override Acres'].sum():,.2f} ac"
                )
                st.session_state["zones_gdf"] = zones_gdf
            else:
                st.error("❌ Could not read zone file.")
        else:
            st.caption("No zone file uploaded.")

    # --- Yield
    with u2:
        st.caption("Yield Map(s) · CSV/GeoJSON/JSON/ZIP(SHP)")
        yield_files = st.file_uploader("Yield", type=["csv", "geojson", "json", "zip"], key="up_yield",
                                       accept_multiple_files=True)
        st.session_state["yield_df"] = pd.DataFrame()
        if yield_files:
            frames, summary = [], []
            for yf in yield_files:
                try:
                    name = yf.name.lower()
                    if name.endswith(".csv"):
                        df = pd.read_csv(yf)
                    else:
                        yg = load_vector_file(yf)
                        df = pd.DataFrame(yg.drop(columns="geometry", errors="ignore")) if yg is not None else pd.DataFrame()
                    if not df.empty:
                        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                        ycol = find_col(df, ["yield", "yld_vol_dr", "yld_mass_dr", "yield_dry", "dry_yield", "wet_yield"])
                        if ycol and ycol != "Yield":
                            df.rename(columns={ycol: "Yield"}, inplace=True)
                        elif not ycol:
                            df["Yield"] = 0.0
                        # try to keep lat/lon as Latitude/Longitude if present
                        latc = find_col(df, ["latitude"])
                        lonc = find_col(df, ["longitude"])
                        if latc and latc != "Latitude":
                            df.rename(columns={latc: "Latitude"}, inplace=True)
                        if lonc and lonc != "Longitude":
                            df.rename(columns={lonc: "Longitude"}, inplace=True)

                        frames.append(df)
                        summary.append({"File": yf.name, "Rows": len(df)})
                except Exception as e:
                    st.warning(f"⚠️ {yf.name}: {e}")
            st.session_state["yield_df"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if summary:
                st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True,
                             height=df_px_height(len(summary)))
        else:
            st.caption("No yield files uploaded.")

    # --- Fert
    with u3:
        st.caption("Fertilizer RX · CSV/GeoJSON/JSON/ZIP(SHP)")
        fert_files = st.file_uploader("Fert", type=["csv", "geojson", "json", "zip"], key="up_fert",
                                      accept_multiple_files=True)
        st.session_state["fert_layers_store"] = {}
        st.session_state["fert_gdfs"] = {}
        if fert_files:
            summ = []
            for f in fert_files:
                grouped, gdf_orig = process_prescription(f, "fertilizer")
                if not grouped.empty:
                    key = os.path.splitext(f.name)[0].lower().replace(" ", "_")
                    st.session_state["fert_layers_store"][key] = grouped
                    if gdf_orig is not None and not gdf_orig.empty:
                        st.session_state["fert_gdfs"][key] = gdf_orig
                    summ.append({"File": f.name, "Products": len(grouped)})
            if summ:
                st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True,
                             height=df_px_height(len(summ)))
        else:
            st.caption("No fertilizer files uploaded.")

    # --- Seed
    with u4:
        st.caption("Seed RX · CSV/GeoJSON/JSON/ZIP(SHP)")
        seed_files = st.file_uploader("Seed", type=["csv", "geojson", "json", "zip"], key="up_seed",
                                      accept_multiple_files=True)
        st.session_state["seed_layers_store"] = {}
        st.session_state["seed_gdf"] = None
        if seed_files:
            summ = []
            last_seed_gdf = None
            for f in seed_files:
                grouped, gdf_orig = process_prescription(f, "seed")
                if not grouped.empty:
                    key = os.path.splitext(f.name)[0].lower().replace(" ", "_")
                    st.session_state["seed_layers_store"][key] = grouped
                    if gdf_orig is not None and not gdf_orig.empty:
                        last_seed_gdf = gdf_orig
                    summ.append({"File": f.name, "Products": len(grouped)})
            if last_seed_gdf is not None and not last_seed_gdf.empty:
                st.session_state["seed_gdf"] = last_seed_gdf
            if summ:
                st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True,
                             height=df_px_height(len(summ)))
        else:
            st.caption("No seed files uploaded.")


# ===========================
# UI: Fixed inputs + Corn/Soy strip
# ===========================
def _mini_num(label: str, key: str, default: float = 0.0, step: float = 0.1):
    st.caption(label)
    return st.number_input(key, min_value=0.0, value=float(st.session_state.get(key, default)),
                           step=step, label_visibility="collapsed")

def render_fixed_inputs_and_strip():
    st.markdown("### Fixed Inputs ($/ac)")

    r = st.columns(12, gap="small")
    with r[0]:  st.session_state["chem"]  = _mini_num("Chem ($/ac)",        "chem",  0.0, 1.0)
    with r[1]:  st.session_state["ins"]   = _mini_num("Insur ($/ac)",       "ins",   0.0, 1.0)
    with r[2]:  st.session_state["insect"]= _mini_num("Insect/Fung ($/ac)", "insect",0.0, 1.0)
    with r[3]:  st.session_state["fert"]  = _mini_num("Fert Flat ($/ac)",   "fert",  0.0, 1.0)
    with r[4]:  st.session_state["seed"]  = _mini_num("Seed Flat ($/ac)",   "seed",  0.0, 1.0)
    with r[5]:  st.session_state["rent"]  = _mini_num("Cash Rent ($/ac)",   "rent",  0.0, 1.0)
    with r[6]:  st.session_state["mach"]  = _mini_num("Mach ($/ac)",        "mach",  0.0, 1.0)
    with r[7]:  st.session_state["labor"] = _mini_num("Labor ($/ac)",       "labor", 0.0, 1.0)
    with r[8]:  st.session_state["col"]   = _mini_num("Living ($/ac)",      "col",   0.0, 1.0)
    with r[9]:  st.session_state["fuel"]  = _mini_num("Fuel ($/ac)",        "fuel",  0.0, 1.0)
    with r[10]: st.session_state["int"]   = _mini_num("Interest ($/ac)",    "int",   0.0, 1.0)
    with r[11]: st.session_state["truck"] = _mini_num("Truck Fuel ($/ac)",  "truck", 0.0, 1.0)

    # store totals
    expenses = {
        "Chemicals": st.session_state["chem"],
        "Insurance": st.session_state["ins"],
        "Insecticide/Fungicide": st.session_state["insect"],
        "Fertilizer (Flat)": st.session_state["fert"],
        "Seed (Flat)": st.session_state["seed"],
        "Cash Rent": st.session_state["rent"],
        "Machinery": st.session_state["mach"],
        "Labor": st.session_state["labor"],
        "Cost of Living": st.session_state["col"],
        "Extra Fuel": st.session_state["fuel"],
        "Extra Interest": st.session_state["int"],
        "Truck Fuel": st.session_state["truck"],
    }
    base_expenses_per_acre = sum(expenses.values())
    st.session_state["expenses_dict"] = expenses
    st.session_state["base_expenses_per_acre"] = base_expenses_per_acre

    # --- collapsible Corn vs Soy chart ---
    with st.expander("Corn vs Soy Profitability (Click to Expand)", expanded=False):
        st.markdown("""
        <style>
        .stDataFrame {overflow: visible !important;}
        .stDataFrame table {margin-left:auto;margin-right:auto;}
        div[data-testid="stExpander"]{max-width:50vw;margin-left:auto;margin-right:auto;}
        </style>
        """, unsafe_allow_html=True)

        prev = pd.DataFrame({
            "Crop": ["Corn", "Soybeans"],
            "Yield (bu/ac)": [st.session_state["corn_yield"], st.session_state["bean_yield"]],
            "Sell Price ($/bu)": [st.session_state["corn_price"], st.session_state["bean_price"]],
        })
        prev["Revenue ($/ac)"] = prev["Yield (bu/ac)"] * prev["Sell Price ($/bu)"]
        prev["Fixed Inputs ($/ac)"] = base_expenses_per_acre
        prev["Breakeven ($/ac)"] = prev["Revenue ($/ac)"] - prev["Fixed Inputs ($/ac)"]

        def _hl(v):
            try:
                v=float(v)
                if v>0: return "color:#22c55e;font-weight:700;"
                if v<0: return "color:#ef4444;font-weight:700;"
            except: pass
            return "font-weight:700;"

        st.dataframe(
            prev.style.applymap(_hl, subset=["Breakeven ($/ac)"]).format({
                "Yield (bu/ac)":"{:,.0f}",
                "Sell Price ($/bu)":"${:,.2f}",
                "Revenue ($/ac)":"${:,.0f}",
                "Fixed Inputs ($/ac)":"${:,.0f}",
                "Breakeven ($/ac)":"${:,.0f}",
            }),
            use_container_width=True, hide_index=True,
            height=df_px_height(len(prev))
        )


# ===========================
# Map helpers / overlays
# ===========================
def make_base_map():
    try:
        m = folium.Map(
            location=[39.5, -98.35], zoom_start=5, min_zoom=2, tiles=None,
            scrollWheelZoom=False, prefer_canvas=True
        )
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

        template = Template("""
            {% macro script(this, kwargs) %}
            var map = {{this._parent.get_name()}};
            map.scrollWheelZoom.disable();
            map.on('click', function(){ map.scrollWheelZoom.enable(); });
            map.on('mouseout', function(){ map.scrollWheelZoom.disable(); });
            {% endmacro %}
        """)
        macro = MacroElement()
        macro._template = template
        m.get_root().add_child(macro)
        return m
    except Exception as e:
        st.error(f"❌ Failed to build base map: {e}")
        return folium.Map(location=[39.5, -98.35], zoom_start=4)


def add_gradient_legend(m, name, vmin, vmax, cmap, index):
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


def detect_rate_type(gdf):
    try:
        rate_col = None
        for c in gdf.columns:
            if "tgt" in c.lower() or "rate" in c.lower():
                rate_col = c
                break
        if rate_col and gdf[rate_col].nunique(dropna=True) == 1:
            return "Fixed Rate"
    except Exception:
        pass
    return "Variable Rate"


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


def add_prescription_overlay(m, gdf, name, cmap, index):
    if gdf is None or gdf.empty:
        return
    gdf = gdf.copy()

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
        gdf, name=name, style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases)
    ).add_to(m)

    add_gradient_legend(m, legend_name, vmin, vmax, cmap, index)


def compute_bounds_for_heatmaps():
    try:
        bnds = []
        for key in ["zones_gdf", "seed_gdf"]:
            g = st.session_state.get(key)
            if g is not None and not getattr(g, "empty", True):
                tb = g.total_bounds
                if tb is not None and len(tb) == 4 and not any(pd.isna(tb)):
                    bnds.append([[tb[1], tb[0]], [tb[3], tb[2]]])
        for _k, fg in st.session_state.get("fert_gdfs", {}).items():
            if fg is not None and not fg.empty:
                tb = fg.total_bounds
                if tb is not None and len(tb) == 4 and not any(pd.isna(tb)):
                    bnds.append([[tb[1], tb[0]], [tb[3], tb[2]]])
        ydf = st.session_state.get("yield_df")
        if ydf is not None and not ydf.empty:
            latc = find_col(ydf, ["latitude"]) or "Latitude"
            lonc = find_col(ydf, ["longitude"]) or "Longitude"
            if latc in ydf.columns and lonc in ydf.columns:
                bnds.append([[ydf[latc].min(), ydf[lonc].min()],
                             [ydf[latc].max(), ydf[lonc].max()]])
        if bnds:
            south = min(b[0][0] for b in bnds)
            west = min(b[0][1] for b in bnds)
            north = max(b[1][0] for b in bnds)
            east = max(b[1][1] for b in bnds)
            return south, west, north, east
    except Exception:
        pass
    return 25.0, -125.0, 49.0, -66.0  # fallback USA


def add_heatmap_overlay(m, df, values, name, cmap, show_default, bounds):
    try:
        if df is None or df.empty:
            return None, None
        south, west, north, east = bounds

        vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        if vals.empty:
            return None, None

        latc = find_col(df, ["latitude"]) or "Latitude"
        lonc = find_col(df, ["longitude"]) or "Longitude"
        if latc not in df.columns or lonc not in df.columns:
            return None, None

        mask = df[[latc, lonc]].applymap(np.isfinite).all(axis=1) & ~pd.to_numeric(df[values.name], errors="coerce").isna()
        if mask.sum() < 3:
            return None, None

        pts_lon = df.loc[mask, lonc].astype(float).values
        pts_lat = df.loc[mask, latc].astype(float).values
        vals_ok = pd.to_numeric(df.loc[mask, values.name], errors="coerce").astype(float).values

        n = 200
        lon_lin = np.linspace(west, east, n)
        lat_lin = np.linspace(south, north, n)
        lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)

        grid_lin = griddata((pts_lon, pts_lat), vals_ok, (lon_grid, lat_grid), method="linear")
        grid_nn = griddata((pts_lon, pts_lat), vals_ok, (lon_grid, lat_grid), method="nearest")
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
            bounds=[[south, west], [north, east]],
            opacity=0.5,
            name=name,
            overlay=True,
            show=show_default
        ).add_to(m)
        return vmin, vmax
    except Exception as e:
        st.warning(f"⚠️ Skipping heatmap {name}: {e}")
        return None, None


# ===========================
# Profit Summary (compact)
# ===========================
def render_profit_summary():
    st.header("Profit Summary")

    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"]{
        max-width:50vw !important;
        margin-left:auto !important;
        margin-right:auto !important;
    }
    .stDataFrame {overflow:visible!important;}
    .stDataFrame table{margin-left:auto;margin-right:auto;}
    </style>
    """, unsafe_allow_html=True)

    def _h(n): return df_px_height(n, row_h=28, header=34, pad=4)

    # session defaults
    expenses = st.session_state.get("expenses_dict", {})
    base_exp = float(st.session_state.get("base_expenses_per_acre", sum(expenses.values())))
    corn_yield = float(st.session_state.get("corn_yield", 200))
    corn_price = float(st.session_state.get("corn_price", 5))
    bean_yield = float(st.session_state.get("bean_yield", 60))
    bean_price = float(st.session_state.get("bean_price", 12))
    target_yield = float(st.session_state.get("target_yield", 200))
    sell_price = float(st.session_state.get("sell_price", corn_price))

    breakeven_df = pd.DataFrame({
        "Crop":["Corn","Soybeans"],
        "Yield Goal (bu/ac)":[corn_yield,bean_yield],
        "Sell Price ($/bu)":[corn_price,bean_price],
    })
    breakeven_df["Revenue ($/ac)"]=[corn_yield*corn_price,bean_yield*bean_price]
    breakeven_df["Fixed Inputs ($/ac)"]=base_exp
    breakeven_df["Breakeven Budget ($/ac)"]=breakeven_df["Revenue ($/ac)"]-breakeven_df["Fixed Inputs ($/ac)"]

    revenue_overall=target_yield*sell_price
    expenses_overall=base_exp
    profit_overall=revenue_overall-expenses_overall

    col_left,col_right=st.columns([1.2,1.8])

    with col_left:
        st.subheader("Profit Metrics Comparison")
        comparison=pd.DataFrame({
            "Metric":["Revenue ($/ac)","Expenses ($/ac)","Profit ($/ac)"],
            "Breakeven Budget":[revenue_overall,expenses_overall,profit_overall],
        })
        st.dataframe(
            comparison.style.format({
                "Breakeven Budget":"${:,.2f}"
            }).applymap(lambda v:"font-weight:700;",subset=["Breakeven Budget"]),
            use_container_width=True,hide_index=True,height=_h(len(comparison))
        )

    with col_right:
        st.subheader("Fixed Input Costs")
        if not expenses:
            keys=["chem","ins","insect","fert","seed","rent","mach","labor","col","fuel","int","truck"]
            labels=["Chemicals","Insurance","Insecticide/Fungicide","Fertilizer (Flat)","Seed (Flat)","Cash Rent",
                    "Machinery","Labor","Cost of Living","Extra Fuel","Extra Interest","Truck Fuel"]
            expenses={lbl:float(st.session_state.get(k,0.0)) for k,lbl in zip(keys,labels)}
        fixed_df=pd.DataFrame(list(expenses.items()),columns=["Expense","$/ac"])
        total=pd.DataFrame([{"Expense":"Total Fixed Costs","$/ac":fixed_df["$/ac"].sum()}])
        fixed_df=pd.concat([fixed_df,total],ignore_index=True)
        st.dataframe(
            fixed_df.style.format({"$/ac":"${:,.2f}"}).apply(
                lambda s:["font-weight:bold;" if v=="Total Fixed Costs" else "" for v in s],subset=["Expense"]
            ),
            use_container_width=True,hide_index=True,height=_h(len(fixed_df))
        )

# ===========================
# MAIN APP
# ===========================
apply_compact_theme()
_bootstrap_defaults()
render_uploaders()
render_fixed_inputs_and_strip()

# =========================================================
# MAP + OVERLAYS + HEATMAPS + SECTION 9 (Left/Right) — COMPACT & BULLETPROOF
# =========================================================

# ---------- tiny helpers (defined only if missing) ----------
if 'df_px_height' not in globals():
    def df_px_height(nrows: int, row_h: int = 28, header: int = 34, pad: int = 2) -> int:
        nrows = max(1, int(nrows))
        return int(header + nrows * row_h + pad)

if 'find_col' not in globals():
    def find_col(df, names):
        """Return first case-insensitive match from names or None."""
        lower = {c.lower(): c for c in df.columns}
        for n in names:
            if n in lower: return lower[n]
        return None

if 'make_base_map' not in globals():
    from branca.element import MacroElement, Template
    import folium
    def make_base_map():
        m = folium.Map(location=[39.5, -98.35], zoom_start=5, tiles=None, control_scale=True, prefer_canvas=True)
        # base tiles (guarded)
        try:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", name="Esri Imagery", control=False
            ).add_to(m)
        except Exception:
            pass
        try:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", name="Esri Labels", overlay=True, control=False
            ).add_to(m)
        except Exception:
            pass
        # enable scrollwheel only on click
        template = Template("""
            {% macro script(this, kwargs) %}
            var map = {{this._parent.get_name()}};
            map.scrollWheelZoom.disable();
            map.on('click', function(){ map.scrollWheelZoom.enable(); });
            map.on('mouseout', function(){ map.scrollWheelZoom.disable(); });
            {% endmacro %}
        """)
        macro = MacroElement(); macro._template = template
        m.get_root().add_child(macro)
        return m

if 'compute_bounds_for_heatmaps' not in globals():
    def compute_bounds_for_heatmaps():
        try:
            bnds = []
            z = st.session_state.get("zones_gdf")
            if z is not None and not getattr(z, "empty", True):
                tb = z.total_bounds; bnds.append([[tb[1], tb[0]], [tb[3], tb[2]]])
            s = st.session_state.get("seed_gdf")
            if s is not None and not getattr(s, "empty", True):
                tb = s.total_bounds; bnds.append([[tb[1], tb[0]], [tb[3], tb[2]]])
            for _, fg in st.session_state.get("fert_gdfs", {}).items():
                if fg is not None and not getattr(fg, "empty", True):
                    tb = fg.total_bounds; bnds.append([[tb[1], tb[0]], [tb[3], tb[2]]])
            y = st.session_state.get("yield_df")
            if isinstance(y, pd.DataFrame) and not y.empty:
                latc = find_col(y, ["latitude"]); lonc = find_col(y, ["longitude"])
                if latc and lonc:
                    bnds.append([[float(y[latc].min()), float(y[lonc].min())],
                                 [float(y[latc].max()), float(y[lonc].max())]])
            if bnds:
                south = min(b[0][0] for b in bnds)
                west  = min(b[0][1] for b in bnds)
                north = max(b[1][0] for b in bnds)
                east  = max(b[1][1] for b in bnds)
                return (south, west, north, east)
        except Exception:
            pass
        return (25.0, -125.0, 49.0, -66.0)  # fallback: CONUS

if 'add_gradient_legend' not in globals():
    from matplotlib import colors as mpl_colors
    import matplotlib.pyplot as plt
    def add_gradient_legend(m, name, vmin, vmax, cmap, index):
        top_offset = 20 + (index * 80)
        stops = [f"{mpl_colors.rgb2hex(cmap(i/100.0)[:3])} {i}%" for i in range(0, 101, 10)]
        gradient_css = ", ".join(stops)
        legend_html = f"""
        <div style="position:absolute; top:{top_offset}px; left:10px; z-index:9999;
                    font-family:sans-serif; font-size:12px; color:white;
                    background:rgba(0,0,0,.65); padding:6px 10px; border-radius:6px; width:180px;">
          <div style="font-weight:600; margin-bottom:4px;">{name}</div>
          <div style="height:14px; background:linear-gradient(90deg,{gradient_css});
                      border-radius:2px; margin-bottom:4px;"></div>
          <div style="display:flex; justify-content:space-between;">
            <span>{vmin:.1f}</span><span>{vmax:.1f}</span>
          </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

if 'add_prescription_overlay' not in globals():
    def add_prescription_overlay(m, gdf, name, cmap, index):
        """Color by rate column if present, else neutral fill."""
        if gdf is None or getattr(gdf, "empty", True): return
        g = gdf.copy()
        # guess a rate column
        rate_col = next((c for c in g.columns if ("tgt" in c.lower() or "rate" in c.lower())), None)
        vmin=vmax=None
        if rate_col is not None:
            vals = pd.to_numeric(g[rate_col], errors="coerce")
            good = vals.dropna()
            if not good.empty:
                vmin, vmax = float(good.min()), float(good.max())
                if vmin == vmax: vmax = vmin + 1.0

        def style_fn(feat):
            if rate_col is None or vmin is None:
                return {"stroke": False, "opacity": 0, "weight": 0, "fillColor": "#808080", "fillOpacity": 0.45}
            try:
                v = float(feat["properties"].get(rate_col))
                t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
                fill = mpl_colors.rgb2hex(cmap(t)[:3])
            except Exception:
                fill = "#808080"
            return {"stroke": False, "opacity": 0, "weight": 0, "fillColor": fill, "fillOpacity": 0.55}

        folium.GeoJson(g, name=name, style_function=style_fn,
                       tooltip=folium.GeoJsonTooltip(fields=[c for c in [rate_col] if c])).add_to(m)
        if vmin is not None:
            add_gradient_legend(m, f"{name} (rate)", vmin, vmax, cmap, index)

if 'add_heatmap_overlay' not in globals():
    import numpy as np
    from scipy.interpolate import griddata
    def add_heatmap_overlay(m, df, values, name, cmap, show_default, bounds):
        try:
            if df is None or df.empty: return None, None
            south, west, north, east = bounds
            latc = find_col(df, ["latitude"]) or "Latitude"
            lonc = find_col(df, ["longitude"]) or "Longitude"
            if latc not in df.columns or lonc not in df.columns: return None, None

            vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
            mask = df[latc].apply(np.isfinite) & df[lonc].apply(np.isfinite) & pd.Series(values).apply(np.isfinite)
            if mask.sum() < 3: return None, None

            pts_lon = df.loc[mask, lonc].astype(float).values
            pts_lat = df.loc[mask, latc].astype(float).values
            vals_ok = pd.to_numeric(pd.Series(values)[mask], errors="coerce").astype(float).values

            n = 220
            lon_lin = np.linspace(west, east, n); lat_lin = np.linspace(south, north, n)
            lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)
            grid_lin = griddata((pts_lon, pts_lat), vals_ok, (lon_grid, lat_grid), method="linear")
            grid_nn  = griddata((pts_lon, pts_lat), vals_ok, (lon_grid, lat_grid), method="nearest")
            grid = np.where(np.isnan(grid_lin), grid_nn, grid_lin)
            if grid is None or np.all(np.isnan(grid)): return None, None

            vmin = float(np.nanpercentile(vals_ok, 5)) if len(vals_ok) else 0.0
            vmax = float(np.nanpercentile(vals_ok, 95)) if len(vals_ok) else 1.0
            if vmin == vmax: vmax = vmin + 1.0

            rgba = cmap((grid - vmin) / (vmax - vmin))
            rgba = np.flipud(rgba); rgba = (rgba * 255).astype(np.uint8)

            folium.raster_layers.ImageOverlay(
                image=rgba, bounds=[[south, west], [north, east]],
                opacity=0.5, name=name, overlay=True, show=show_default
            ).add_to(m)
            return vmin, vmax
        except Exception:
            return None, None

# ---------- build base map ----------
m = make_base_map()

# ---------- Zones overlay ----------
zones_gdf = st.session_state.get("zones_gdf")
if zones_gdf is not None and not getattr(zones_gdf, "empty", True):
    try:
        zb = zones_gdf.total_bounds
        m.location = [(zb[1] + zb[3]) / 2, (zb[0] + zb[2]) / 2]
        m.zoom_start = 15

        palette = ["#FF0000","#FF8C00","#FFFF00","#32CD32","#006400",
                   "#1E90FF","#8A2BE2","#FFC0CB","#A52A2A","#00CED1"]
        unique_vals = list(dict.fromkeys(sorted(list(zones_gdf["Zone"].astype(str).unique()))))
        color_map = {z: palette[i % len(palette)] for i, z in enumerate(unique_vals)}

        folium.GeoJson(
            zones_gdf, name="Zones",
            style_function=lambda feature: {
                "fillColor": color_map.get(str(feature["properties"].get("Zone", "")), "#808080"),
                "color": "black", "weight": 1, "fillOpacity": 0.08,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["Zone", "Calculated Acres", "Override Acres"] if c in zones_gdf.columns]
            )
        ).add_to(m)

        legend_items = "".join(
            f"<div style='display:flex;align-items:center;margin:2px 0;'>"
            f"<div style='background:{color_map[z]};width:14px;height:14px;margin-right:6px;'></div>{z}</div>"
            for z in unique_vals
        )
        legend_html = f"""
        <div id="zone-legend" style="position:absolute; bottom:20px; right:20px; z-index:9999;
                     font-family:sans-serif; font-size:13px; color:white;
                     background-color: rgba(0,0,0,0.65); padding:6px 10px; border-radius:5px; width:160px;">
          <div style="font-weight:600; margin-bottom:4px; cursor:pointer;"
               onclick="var x = document.getElementById('zone-legend-items');
                        if (x.style.display === 'none') {{ x.style.display = 'block'; }}
                        else {{ x.style.display = 'none'; }}">
            Zone Colors ▼
          </div>
          <div id="zone-legend-items" style="display:block;">{legend_items}</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
    except Exception as e:
        st.warning(f"⚠️ Skipping zones overlay: {e}")

# ---------- Prescription overlays ----------
legend_ix = 0
seed_gdf = st.session_state.get("seed_gdf")
if seed_gdf is not None and not getattr(seed_gdf, "empty", True):
    add_prescription_overlay(m, seed_gdf, "Seed RX", plt.cm.Greens, legend_ix)
    legend_ix += 1

for k, fgdf in st.session_state.get("fert_gdfs", {}).items():
    if fgdf is not None and not fgdf.empty:
        add_prescription_overlay(m, fgdf, f"Fertilizer RX: {k}", plt.cm.Blues, legend_ix)
        legend_ix += 1

# ---------- Heatmaps (yield / profits) ----------
bounds = compute_bounds_for_heatmaps()
ydf = st.session_state.get("yield_df")
sell_price = float(st.session_state.get("sell_price", st.session_state.get("corn_price", 5.0)))

if not (isinstance(ydf, pd.DataFrame) and not ydf.empty and
        find_col(ydf, ["latitude"]) and find_col(ydf, ["longitude"])):
    # fallback: single cell at map center using Target Yield
    lat_center = (bounds[0] + bounds[2]) / 2.0
    lon_center = (bounds[1] + bounds[3]) / 2.0
    df_for_maps = pd.DataFrame({
        "Yield": [float(st.session_state.get("target_yield", 200.0))],
        "Latitude": [lat_center], "Longitude": [lon_center]
    })
else:
    df_for_maps = ydf.copy()
    # ensure consistent names
    latc = find_col(df_for_maps, ["latitude"])
    lonc = find_col(df_for_maps, ["longitude"])
    if latc and latc != "Latitude": df_for_maps.rename(columns={latc: "Latitude"}, inplace=True)
    if lonc and lonc != "Longitude": df_for_maps.rename(columns={lonc: "Longitude"}, inplace=True)

# Profit metrics for heatmaps (bulletproof)
try:
    df_for_maps = df_for_maps.copy()

    if "Yield" not in df_for_maps.columns:
        df_for_maps["Yield"] = 0.0
    df_for_maps["Yield"] = pd.to_numeric(df_for_maps["Yield"], errors="coerce").fillna(0.0)
    df_for_maps["Revenue_per_acre"] = df_for_maps["Yield"] * sell_price

    # base expenses fallback
    base_expenses_per_acre = float(st.session_state.get("base_expenses_per_acre", 0.0))
    if not base_expenses_per_acre:
        # try from expenses dict if available
        base_expenses_per_acre = float(sum(st.session_state.get("expenses_dict", {}).values())) if isinstance(
            st.session_state.get("expenses_dict", {}), dict) else 0.0
    st.session_state["base_expenses_per_acre"] = base_expenses_per_acre  # keep for Section 9

    # variable RX totals
    fert_var = 0.0
    for d in st.session_state.get("fert_layers_store", {}).values():
        if isinstance(d, pd.DataFrame) and not d.empty:
            fert_var += float(pd.to_numeric(d.get("CostPerAcre", 0), errors="coerce").fillna(0.0).sum())
    seed_var = 0.0
    for d in st.session_state.get("seed_layers_store", {}).values():
        if isinstance(d, pd.DataFrame) and not d.empty:
            seed_var += float(pd.to_numeric(d.get("CostPerAcre", 0), errors="coerce").fillna(0.0).sum())

    df_for_maps["NetProfit_Variable"] = df_for_maps["Revenue_per_acre"] - (base_expenses_per_acre + fert_var + seed_var)

    # fixed product costs ($/ac)
    fixed_costs = 0.0
    fx = st.session_state.get("fixed_products", pd.DataFrame())
    if isinstance(fx, pd.DataFrame) and not fx.empty:
        if "$/ac" in fx.columns:
            fixed_costs = float(pd.to_numeric(fx["$/ac"], errors="coerce").fillna(0.0).sum())
        else:
            rcol = next((c for c in ["Rate", "rate"] if c in fx.columns), None)
            pcol = next((c for c in ["CostPerUnit", "costperunit"] if c in fx.columns), None)
            if rcol and pcol:
                fixed_costs = float(
                    (pd.to_numeric(fx[rcol], errors="coerce").fillna(0.0) *
                     pd.to_numeric(fx[pcol], errors="coerce").fillna(0.0)).sum()
                )
    df_for_maps["NetProfit_Fixed"] = df_for_maps["Revenue_per_acre"] - (base_expenses_per_acre + fixed_costs)

except Exception:
    st.warning("⚠️ Could not compute profit metrics for heatmaps; using zeros.")
    df_for_maps["Revenue_per_acre"] = 0.0
    df_for_maps["NetProfit_Variable"] = 0.0
    df_for_maps["NetProfit_Fixed"] = 0.0

# add heatmap layers + legends
legend_i = legend_ix
ymin, ymax = add_heatmap_overlay(m, df_for_maps, df_for_maps["Yield"], "Yield (bu/ac)", plt.cm.RdYlGn, False, bounds)
if ymin is not None:
    add_gradient_legend(m, "Yield (bu/ac)", ymin, ymax, plt.cm.RdYlGn, legend_i); legend_i += 1

vmin, vmax = add_heatmap_overlay(m, df_for_maps, df_for_maps["NetProfit_Variable"],
                                 "Variable Rate Profit ($/ac)", plt.cm.RdYlGn, True, bounds)
if vmin is not None:
    add_gradient_legend(m, "Variable Rate Profit ($/ac)", vmin, vmax, plt.cm.RdYlGn, legend_i); legend_i += 1

fmin, fmax = add_heatmap_overlay(m, df_for_maps, df_for_maps["NetProfit_Fixed"],
                                 "Fixed Rate Profit ($/ac)", plt.cm.RdYlGn, False, bounds)
if fmin is not None:
    add_gradient_legend(m, "Fixed Rate Profit ($/ac)", fmin, fmax, plt.cm.RdYlGn, legend_i); legend_i += 1

# final map setup
try:
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
except Exception:
    pass
st_folium(m, use_container_width=True, height=600)

# =========================================================
# 9. PROFIT SUMMARY — 50% WIDTH CENTERED, NO SCROLLBARS
# =========================================================
def render_profit_summary():
    st.markdown("## Profit Summary", anchor=False)

    # Create a centered container that only uses 50% of viewport
    left_spacer, mid, right_spacer = st.columns([1, 2, 1], vertical_alignment="top")
    with mid:
        # --- make sure scrollbars never appear ---
        st.markdown("""
            <style>
            .stDataFrame {overflow:visible !important;}
            .stDataFrame table {margin-left:auto;margin-right:auto;}
            div[data-testid="stHorizontalBlock"]{justify-content:center !important;}
            </style>
        """, unsafe_allow_html=True)

        # ---- get data from session state ----
        expenses = st.session_state.get("expenses_dict", {})
        base_exp = float(st.session_state.get("base_expenses_per_acre", sum(expenses.values())))
        corn_yield = float(st.session_state.get("corn_yield", 200))
        corn_price = float(st.session_state.get("corn_price", 5))
        bean_yield = float(st.session_state.get("bean_yield", 60))
        bean_price = float(st.session_state.get("bean_price", 12))
        target_yield = float(st.session_state.get("target_yield", 200))
        sell_price = float(st.session_state.get("sell_price", corn_price))

        # ---- build dataframes ----
        breakeven_df = pd.DataFrame({
            "Crop": ["Corn", "Soybeans"],
            "Yield Goal (bu/ac)": [corn_yield, bean_yield],
            "Sell Price ($/bu)": [corn_price, bean_price],
        })
        breakeven_df["Revenue ($/ac)"] = breakeven_df["Yield Goal (bu/ac)"] * breakeven_df["Sell Price ($/bu)"]
        breakeven_df["Fixed Inputs ($/ac)"] = base_exp
        breakeven_df["Breakeven Budget ($/ac)"] = breakeven_df["Revenue ($/ac)"] - breakeven_df["Fixed Inputs ($/ac)"]

        fixed_df = pd.DataFrame(list(expenses.items()), columns=["Expense", "$/ac"])
        total_row = pd.DataFrame([{"Expense": "Total Fixed Costs", "$/ac": fixed_df["$/ac"].sum()}])
        fixed_df = pd.concat([fixed_df, total_row], ignore_index=True)

        # ---- two side-by-side tables in centered block ----
        col_left, col_right = st.columns([1, 1], gap="large")

        # ---- left table ----
        with col_left:
            st.subheader("Breakeven Budget (Corn vs Beans)")
            def _hl(val):
                try:
                    v = float(val)
                    if v > 0: return "color:#22c55e;font-weight:700;"
                    if v < 0: return "color:#ef4444;font-weight:700;"
                except: return ""
                return "font-weight:700;"
            st.dataframe(
                breakeven_df.style.applymap(_hl, subset=["Breakeven Budget ($/ac)"]).format({
                    "Yield Goal (bu/ac)": "{:,.0f}",
                    "Sell Price ($/bu)": "${:,.2f}",
                    "Revenue ($/ac)": "${:,.0f}",
                    "Fixed Inputs ($/ac)": "${:,.0f}",
                    "Breakeven Budget ($/ac)": "${:,.0f}",
                }),
                use_container_width=True, hide_index=True
            )

        # ---- right table ----
        with col_right:
            st.subheader("Fixed Input Costs")
            st.dataframe(
                fixed_df.style.format({"$/ac": "${:,.2f}"}).apply(
                    lambda s: ["font-weight:bold;" if v == "Total Fixed Costs" else "" for v in s],
                    subset=["Expense"]
                ),
                use_container_width=True, hide_index=True
            )

    # allow page to scroll again
    st.markdown("<style>html,body{overflow:auto!important;}</style>", unsafe_allow_html=True)


# ---------- render summary ----------
render_profit_summary()

