# =========================================================
# Farm Profit Mapping Tool V4 — COMPACT + BULLETPROOF
# =========================================================
import os
import zipfile
import tempfile
from typing import Optional

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

# Clear caches (avoid stale state during hot-reloads)
st.cache_data.clear()
st.cache_resource.clear()

# === SCROLL CONTROL (Scoped SAFE v2) ===
st.markdown("""
<style>
/* ------------------------------
   A. DataFrames (READ-ONLY tables)
   ------------------------------ */
/* Let dataframes expand with no internal scroll and no sticky header gap. */
[data-testid="stDataFrameContainer"],
[data-testid="stDataFrameScrollableContainer"],
[data-testid="stDataFrame"] {
    overflow: visible !important;
    height: auto !important;
    max-height: none !important;
    width: 100% !important;
    max-width: 100% !important;
}
[data-testid="stStickyTableHeader"] { position: static !important; }

/* ------------------------------
   B. DataEditors (EDITABLE grids)
   ------------------------------ */
/* Keep editor virtualization alive so cells actually render.
   We'll prevent scroll via explicit height, not CSS. */
[data-testid="stDataEditorContainer"],
[data-testid="stDataEditorGrid"] {
    overflow: visible !important;  /* allow full expansion */
    height: auto !important;
    max-height: none !important;
    width: 100% !important;
    max-width: 100% !important;
}

/* Make tables use the full width without forcing horizontal scroll */
[data-testid="stDataFrame"] table,
[data-testid="stDataEditorGrid"] table {
    min-width: 100% !important;
    width: 100% !important;
    table-layout: fixed !important;
    border-collapse: collapse !important;
}

/* Hide resize grips / scroll shadows */
[data-testid*="Resizer"], [class*="StyledScroll"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# === SAFE AUTO-EXPAND SCRIPT ===
st.markdown("""
<script>
function safeFixTables(){
  const dfs = parent.document.querySelectorAll('[data-testid="stDataFrameContainer"]');
  dfs.forEach(df=>{
    df.style.overflow='visible';
    df.style.height='auto';
    df.style.maxHeight='none';
  });
}
safeFixTables();
setTimeout(safeFixTables, 1000);
setTimeout(safeFixTables, 3000);
</script>
""", unsafe_allow_html=True)

# ===========================
# COMPACT THEME + LAYOUT
# ===========================
def apply_compact_theme():
    st.set_page_config(page_title="Farm ROI Tool V4", layout="wide")
    st.title("Farm Profit Mapping Tool V4")

# ===========================
# HELPERS
# ===========================
def auto_height(df: pd.DataFrame, row_h: int = 36, header: int = 44, pad: int = 16) -> int:
    """
    Robust height so tables/editors never show a vertical scrollbar.
    Slightly conservative to account for header wrapping on narrow screens.
    """
    n = max(1, len(df))
    return int(header + n * row_h + pad)

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
                st.error(f"Could not read {prescrip_type} map.")
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
        st.warning(f"Failed to read {file.name}: {e}")
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
                    f"Zones: {len(zones_gdf)}  |  Calc: {zones_gdf['Calculated Acres'].sum():,.2f} ac  |  "
                    f"Override: {zones_gdf['Override Acres'].sum():,.2f} ac"
                )
                st.session_state["zones_gdf"] = zones_gdf
            else:
                st.error("Could not read zone file.")
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
                        latc = find_col(df, ["latitude"])
                        lonc = find_col(df, ["longitude"])
                        if latc and latc != "Latitude":
                            df.rename(columns={latc: "Latitude"}, inplace=True)
                        if lonc and lonc != "Longitude":
                            df.rename(columns={lonc: "Longitude"}, inplace=True)

                        frames.append(df)
                        summary.append({"File": yf.name, "Rows": len(df)})
                except Exception as e:
                    st.warning(f"{yf.name}: {e}")
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
# UI: Fixed inputs + Variable/Flat/CornSoy strip
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
    st.session_state["expenses_dict"] = expenses
    st.session_state["base_expenses_per_acre"] = sum(expenses.values())

# =========================================================
# SECTION: Variable Rate, Flat Rate (2 Columns, no scroll)
# =========================================================
def render_input_sections():
    # --- Scoped CSS: keep things tidy & full-width without inner scroll ---
    st.markdown("""
    <style>
      [data-testid="stDataEditorGrid"],
      [data-testid="stDataFrameContainer"],
      [data-testid="stDataFrameScrollableContainer"],
      [data-testid="stDataFrame"] {
          overflow: visible !important;
          height: auto !important;
          max-height: none !important;
          width: 100% !important;
      }
      [data-testid="stDataEditorGrid"] table,
      [data-testid="stDataFrame"] table {
          width: 100% !important;
          min-width: 100% !important;
          table-layout: fixed !important;
      }
    </style>
    """, unsafe_allow_html=True)

    def _profit_color(v):
        if isinstance(v, (int, float)):
            if v > 0:
                return "color:limegreen;font-weight:bold;"
            if v < 0:
                return "color:#ff4d4d;font-weight:bold;"
        return ""

    # ===============================
    # PREP: Gather all detected inputs
    # ===============================
    fert_store = st.session_state.get("fert_layers_store", {})
    seed_store = st.session_state.get("seed_layers_store", {})

    fert_products, seed_products = [], []
    for _, df in fert_store.items():
        if isinstance(df, pd.DataFrame) and "product" in df.columns:
            fert_products.extend(list(df["product"].dropna().unique()))
    for _, df in seed_store.items():
        if isinstance(df, pd.DataFrame) and "product" in df.columns:
            seed_products.extend(list(df["product"].dropna().unique()))

    fert_products = sorted(set(fert_products))
    seed_products = sorted(set(seed_products))

    # Build editor rows from detected products
    all_variable_inputs = (
        [{"Type": "Fertilizer", "Product": p, "Units Applied": 0.0, "Price per Unit ($)": 0.0} for p in fert_products] +
        [{"Type": "Seed",       "Product": p, "Units Applied": 0.0, "Price per Unit ($)": 0.0} for p in seed_products]
    )
    if not all_variable_inputs:
        all_variable_inputs = [{
            "Type": "Fertilizer", "Product": "", "Units Applied": 0.0, "Price per Unit ($)": 0.0
        }]

    cols = st.columns(2, gap="small")

    # -------------------------------------------------
    # 1) VARIABLE RATE INPUTS
    # -------------------------------------------------
    with cols[0]:
        st.markdown("### Variable Rate Inputs")
        with st.expander("Open Variable Rate Inputs", expanded=False):
            st.caption("Enter price per unit and total units applied from RX maps or manually.")
            rx_df = pd.DataFrame(all_variable_inputs)

            # Fixed rows = no checkbox column
            edited = st.data_editor(
                rx_df,
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                key="var_inputs_editor_final",
                column_config={
                    "Type": st.column_config.TextColumn(disabled=True),
                    "Product": st.column_config.TextColumn(),
                    "Units Applied": st.column_config.NumberColumn(format="%.4f"),
                    "Price per Unit ($)": st.column_config.NumberColumn(format="%.2f"),
                },
                height=auto_height(rx_df)
            ).fillna(0.0)

            edited["Total Cost ($)"] = edited["Units Applied"] * edited["Price per Unit ($)"]

            st.dataframe(
                edited.style.format({
                    "Units Applied": "{:,.4f}",
                    "Price per Unit ($)": "${:,.2f}",
                    "Total Cost ($)": "${:,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
                height=auto_height(edited)
            )

            base_acres = float(st.session_state.get("base_acres", 1.0))
            st.session_state["variable_rate_inputs"] = edited
            st.session_state["variable_rate_cost_per_acre"] = (
                float(edited["Total Cost ($)"].sum()) / max(base_acres, 1.0)
            )

    # -------------------------------------------------
    # 2) FLAT RATE INPUTS
    # -------------------------------------------------
    with cols[1]:
        st.markdown("### Flat Rate Inputs")
        with st.expander("Open Flat Rate Inputs", expanded=False):
            st.caption("Uniform cost per acre for the entire field.")
            flat_products = sorted(set(fert_products + seed_products))
            flat_df = pd.DataFrame([{
                "Product": p, "Rate (units/ac)": 0.0, "Price per Unit ($)": 0.0
            } for p in flat_products]) if flat_products else pd.DataFrame(
                [{"Product": "", "Rate (units/ac)": 0.0, "Price per Unit ($)": 0.0}]
            )

            edited_flat = st.data_editor(
                flat_df,
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                key="flat_inputs_editor_final",
                column_config={
                    "Product": st.column_config.TextColumn(),
                    "Rate (units/ac)": st.column_config.NumberColumn(format="%.4f"),
                    "Price per Unit ($)": st.column_config.NumberColumn(format="%.2f"),
                },
                height=auto_height(flat_df)
            ).fillna(0.0)

            edited_flat["Cost per Acre ($/ac)"] = (
                edited_flat["Rate (units/ac)"] * edited_flat["Price per Unit ($)"]
            )

            st.dataframe(
                edited_flat.style.format({
                    "Rate (units/ac)": "{:,.4f}",
                    "Price per Unit ($)": "${:,.2f}",
                    "Cost per Acre ($/ac)": "${:,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
                height=auto_height(edited_flat)
            )

            # Persist in state for Section 9 math
            out_flat = edited_flat.copy()
            out_flat["$/ac"] = out_flat["Cost per Acre ($/ac)"]
            st.session_state["flat_products"] = edited_flat
            st.session_state["fixed_products"] = out_flat
            st.session_state["flat_rate_cost_per_acre"] = float(
                edited_flat["Cost per Acre ($/ac)"].sum()
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
        st.error(f"Failed to build base map: {e}")
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
        st.warning(f"Skipping heatmap {name}: {e}")
        return None, None

# ===========================
# MAIN APP
# ===========================
apply_compact_theme()
_bootstrap_defaults()
render_uploaders()
render_fixed_inputs_and_strip()
# Three collapsible input dropdowns ABOVE the map
render_input_sections()
st.markdown("---")
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
        st.warning(f"Skipping zones overlay: {e}")

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
    lat_center = (bounds[0] + bounds[2]) / 2.0
    lon_center = (bounds[1] + bounds[3]) / 2.0
    df_for_maps = pd.DataFrame({
        "Yield": [float(st.session_state.get("target_yield", 200.0))],
        "Latitude": [lat_center], "Longitude": [lon_center]
    })
else:
    df_for_maps = ydf.copy()
    latc = find_col(df_for_maps, ["latitude"])
    lonc = find_col(df_for_maps, ["longitude"])
    if latc and latc != "Latitude": df_for_maps.rename(columns={latc: "Latitude"}, inplace=True)
    if lonc and lonc != "Longitude": df_for_maps.rename(columns={lonc: "Longitude"}, inplace=True)

try:
    df_for_maps = df_for_maps.copy()

    if "Yield" not in df_for_maps.columns:
        df_for_maps["Yield"] = 0.0
    df_for_maps["Yield"] = pd.to_numeric(df_for_maps["Yield"], errors="coerce").fillna(0.0)
    df_for_maps["Revenue_per_acre"] = df_for_maps["Yield"] * sell_price

    base_expenses_per_acre = float(st.session_state.get("base_expenses_per_acre", 0.0))
    if not base_expenses_per_acre:
        base_expenses_per_acre = float(sum(st.session_state.get("expenses_dict", {}).values()))
    st.session_state["base_expenses_per_acre"] = base_expenses_per_acre

    fert_var = 0.0
    for d in st.session_state.get("fert_layers_store", {}).values():
        if isinstance(d, pd.DataFrame) and not d.empty:
            fert_var += float(pd.to_numeric(d.get("CostPerAcre", 0), errors="coerce").fillna(0.0).sum())
    seed_var = 0.0
    for d in st.session_state.get("seed_layers_store", {}).values():
        if isinstance(d, pd.DataFrame) and not d.empty:
            seed_var += float(pd.to_numeric(d.get("CostPerAcre", 0), errors="coerce").fillna(0.0).sum())

    df_for_maps["NetProfit_Variable"] = df_for_maps["Revenue_per_acre"] - (base_expenses_per_acre + fert_var + seed_var)

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
    st.warning("Could not compute profit metrics for heatmaps; using zeros.")
    df_for_maps["Revenue_per_acre"] = 0.0
    df_for_maps["NetProfit_Variable"] = 0.0
    df_for_maps["NetProfit_Fixed"] = 0.0

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

try:
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
except Exception:
    pass
st_folium(m, use_container_width=True, height=600)

# =========================================================
# 9. PROFIT SUMMARY — Profit Comparison + Corn/Soy + Fixed Inputs
# =========================================================
def render_profit_summary():
    st.header("Profit Summary")

    # ---------- Local helpers (unique; no duplicates) ----------
    def _money(x):
        try:
            return f"${x:,.2f}"
        except Exception:
            return x

    def _profit_color(v):
        if not isinstance(v, (int, float)):
            return ""
        if v > 0:
            return "color:limegreen;font-weight:bold;"
        if v < 0:
            return "color:#ff4d4d;font-weight:bold;"
        return "font-weight:bold;"

    def _df_height(df: pd.DataFrame, row_h: int = 34, header_h: int = 40, pad: int = 6, fudge: int = 0) -> int:
        """
        Pixel-exact height so st.dataframe never scrolls and never shows a partial blank row.
        Calibrated for Streamlit's dataframe row metrics (row ~34px, header ~40px).
        'fudge' lets us nudge individual tables (e.g., add 2–6px if needed).
        """
        try:
            n = len(df) if isinstance(df, pd.DataFrame) else 1
            # base linear height
            h = header_h + (max(1, n) * row_h) + pad
            # tiny tables need a touch more to avoid the "half line" artifact
            if n <= 2:
                h += 8
            # one more safety pixel
            return int(h + fudge)
        except Exception:
            return 200

    # ---------- Safe state pulls ----------
    expenses = st.session_state.get("expenses_dict", {}) or {}
    base_exp = float(st.session_state.get("base_expenses_per_acre", sum(expenses.values()) if expenses else 0.0))

    corn_yield = float(st.session_state.get("corn_yield", 200))
    corn_price = float(st.session_state.get("corn_price", 5))
    bean_yield = float(st.session_state.get("bean_yield", 60))
    bean_price = float(st.session_state.get("bean_price", 12))

    target_yield = float(st.session_state.get("target_yield", 200))
    sell_price = float(st.session_state.get("sell_price", corn_price))

    # ---------- Profit math (base / variable / fixed) ----------
    revenue_per_acre = target_yield * sell_price
    fixed_inputs = base_exp

    fert_costs_var = seed_costs_var = 0.0
    df_fert = st.session_state.get("fert_products")
    if isinstance(df_fert, pd.DataFrame) and "CostPerAcre" in df_fert.columns:
        fert_costs_var = pd.to_numeric(df_fert["CostPerAcre"], errors="coerce").fillna(0.0).sum()
    df_seed = st.session_state.get("seed_products")
    if isinstance(df_seed, pd.DataFrame) and "CostPerAcre" in df_seed.columns:
        seed_costs_var = pd.to_numeric(df_seed["CostPerAcre"], errors="coerce").fillna(0.0).sum()
    expenses_var = fixed_inputs + fert_costs_var + seed_costs_var
    profit_var = revenue_per_acre - expenses_var

    fert_costs_fix = seed_costs_fix = 0.0
    df_fix = st.session_state.get("fixed_products")
    if isinstance(df_fix, pd.DataFrame) and not df_fix.empty:
        if "Type" in df_fix.columns and "$/ac" in df_fix.columns:
            fert_costs_fix = pd.to_numeric(df_fix.loc[df_fix["Type"] == "Fertilizer", "$/ac"], errors="coerce").fillna(0.0).sum()
            seed_costs_fix = pd.to_numeric(df_fix.loc[df_fix["Type"] == "Seed", "$/ac"], errors="coerce").fillna(0.0).sum()
        elif "$/ac" in df_fix.columns:
            fert_costs_fix = seed_costs_fix = pd.to_numeric(df_fix["$/ac"], errors="coerce").fillna(0.0).sum()
    expenses_fix = fixed_inputs + fert_costs_fix + seed_costs_fix
    profit_fix = revenue_per_acre - expenses_fix

    # ---------- Tables ----------
    # 1) Profit comparison
    comparison = pd.DataFrame({
        "Metric": ["Revenue ($/ac)", "Expenses ($/ac)", "Profit ($/ac)"],
        "Breakeven Budget": [revenue_per_acre, fixed_inputs, revenue_per_acre - fixed_inputs],
        "Variable Rate":    [revenue_per_acre, expenses_var, profit_var],
        "Fixed Rate":       [revenue_per_acre, expenses_fix, profit_fix],
    })

    # 2) Corn vs Soy
    cornsoy = pd.DataFrame({
        "Crop": ["Corn", "Soybeans"],
        "Yield (bu/ac)": [corn_yield, bean_yield],
        "Sell Price ($/bu)": [corn_price, bean_price],
    })
    cornsoy["Revenue ($/ac)"] = cornsoy["Yield (bu/ac)"] * cornsoy["Sell Price ($/bu)"]
    cornsoy["Fixed Inputs ($/ac)"] = base_exp
    cornsoy["Profit ($/ac)"] = cornsoy["Revenue ($/ac)"] - cornsoy["Fixed Inputs ($/ac)"]

    # 3) Fixed inputs breakdown (right column)
    fixed_df = pd.DataFrame(list(expenses.items()), columns=["Expense", "$/ac"])
    if not fixed_df.empty:
        total_row = pd.DataFrame([{"Expense": "Total Fixed Costs", "$/ac": fixed_df["$/ac"].sum()}])
        fixed_df = pd.concat([fixed_df, total_row], ignore_index=True)

    # ---------- Layout (2 columns) ----------
    left, right = st.columns([2, 1], gap="large")

    # --- tighten the left-column charts (profit + corn/soy) ---
with left:
    st.subheader("Profit Comparison")
    styled_comp = (
        comparison.style
        .format(_money)
        .applymap(_profit_color, subset=["Breakeven Budget", "Variable Rate", "Fixed Rate"])
    )
    # Reduced row height slightly and trimmed padding
    st.dataframe(
        styled_comp,
        use_container_width=True,
        hide_index=True,
        height=_df_height(comparison, row_h=32, header_h=36, pad=-2, fudge=-6),
    )

    st.subheader("Corn vs Soybean Profitability")
    styled_cs = cornsoy.style.format(_money).applymap(_profit_color, subset=["Profit ($/ac)"])
    # Slightly smaller again to remove the visual “half row” buffer
    st.dataframe(
        styled_cs,
        use_container_width=True,
        hide_index=True,
        height=_df_height(cornsoy, row_h=31, header_h=35, pad=-2, fudge=-8),
    )

        # Compact horizontal formulas
        with st.expander("Show Calculation Formulas", expanded=False):
            st.markdown(
                """
                <div style="display:flex;flex-wrap:wrap;gap:6px;font-size:0.75rem;line-height:1.1rem;">
                  <div style="flex:1;min-width:180px;border:1px solid #444;border-radius:6px;padding:5px;background-color:#111;">
                    <b>Breakeven Budget</b><br>(Target Yield × Sell Price) − Fixed Inputs
                  </div>
                  <div style="flex:1;min-width:180px;border:1px solid #444;border-radius:6px;padding:5px;background-color:#111;">
                    <b>Variable Rate</b><br>(Target Yield × Sell Price) − (Fixed Inputs + Var Seed + Var Fert)
                  </div>
                  <div style="flex:1;min-width:180px;border:1px solid #444;border-radius:6px;padding:5px;background-color:#111;">
                    <b>Fixed Rate</b><br>(Target Yield × Sell Price) − (Fixed Inputs + Fixed Seed + Fixed Fert)
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("Corn vs Soybean Profitability")
        styled_cs = cornsoy.style.format(_money).applymap(_profit_color, subset=["Profit ($/ac)"])
        st.dataframe(
            styled_cs,
            use_container_width=True,
            hide_index=True,
            height=_df_height(cornsoy, fudge=1),  # tiny nudge to eliminate the partial-row hint
        )

    with right:
        st.subheader("Fixed Input Costs")
        if not fixed_df.empty:
            st.dataframe(
                fixed_df.style.format(_money),
                use_container_width=True,
                hide_index=True,
                height=_df_height(fixed_df, fudge=4),  # right table historically needed +4px
            )
        else:
            st.info("Enter your fixed inputs above to see totals here.")

# ---------- render ----------
render_profit_summary()

# --- FINAL SCROLL CLEANUP (runs after Streamlit's rerender) ---
st.markdown("""
<script>
function killScrollbars(){
  const els = window.parent.document.querySelectorAll(
    '[data-testid="stDataFrameContainer"],[data-testid="stDataEditorContainer"]'
  );
  els.forEach(el=>{
    el.style.overflow='visible';
    el.style.height='auto';
    el.style.maxHeight='none';
  });
}
// Run now and whenever Streamlit reflows
killScrollbars();
new MutationObserver(killScrollbars)
  .observe(window.parent.document.body,{childList:true,subtree:true});
</script>
""", unsafe_allow_html=True)

# =========================================================
# GLOBAL NO-SCROLL + WIDTH FIX — FINAL OVERRIDE
# =========================================================
st.markdown("""
<style>
/* kill all editor and dataframe scrolls */
[data-testid="stDataFrameContainer"],
[data-testid="stDataEditorGrid"],
[data-testid="stDataFrame"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="stDataEditorContainer"] {
    overflow: visible !important;
    height: auto !important;
    max-height: none !important;
    width: 100% !important;
    max-width: 100% !important;
}
/* force table body & header to render full height */
[data-testid="stDataEditorGrid"] table,
[data-testid="stDataFrame"] table {
    min-width: 100% !important;
}
/* remove Streamlit’s auto-scroll shadows */
[data-testid="stDataEditorResizer"],
[data-testid="stDataFrameResizer"] {
    display: none !important;
}
</style>

<script>
function fixLayoutAndScroll() {
    const outer = window.parent?.document?.querySelector('.block-container');
    if (outer) {
        outer.style.maxWidth = '85%';
        outer.style.margin = 'auto';
        outer.style.paddingTop = '0.5rem';
        outer.style.paddingBottom = '1rem';
    }
    // force every grid to expand fully
    document.querySelectorAll('[data-testid="stDataFrameContainer"],[data-testid="stDataEditorContainer"]').forEach(el=>{
        el.style.overflow='visible';
        el.style.height='auto';
        el.style.maxHeight='none';
    });
}
fixLayoutAndScroll();
setTimeout(fixLayoutAndScroll, 500);
setTimeout(fixLayoutAndScroll, 1500);
setTimeout(fixLayoutAndScroll, 4000);
</script>
""", unsafe_allow_html=True)
