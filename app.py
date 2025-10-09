# =========================================================
# Farm Profit Mapping Tool V4 — COMPACT + BULLETPROOF (Patched)
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

# Clear caches
st.cache_data.clear()
st.cache_resource.clear()

# === SCROLL CONTROL (Scoped SAFE v2) ===
st.markdown("""
<style>
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

[data-testid="stDataEditorContainer"],
[data-testid="stDataEditorGrid"] {
    overflow: visible !important;
    height: auto !important;
    max-height: none !important;
    width: 100% !important;
    max-width: 100% !important;
}
[data-testid="stDataFrame"] table,
[data-testid="stDataEditorGrid"] table {
    min-width: 100% !important;
    width: 100% !important;
    table-layout: fixed !important;
    border-collapse: collapse !important;
}
[data-testid*="Resizer"], [class*="StyledScroll"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

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
setTimeout(safeFixTables,1000);
setTimeout(safeFixTables,3000);
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
import re

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def pick_col(df: pd.DataFrame, preferred: list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    norm_map = {_norm(c): c for c in df.columns}
    # exact / preferred order first
    for key in preferred:
        if key in norm_map:
            return norm_map[key]
    # then fuzzy contains "yield" / "yld"
    for k, orig in norm_map.items():
        if "yield" in k or re.search(r"\byld\b", k):
            return orig
    return None

YIELD_PREFS = [
    "yield", "yld_vol_dr", "yld_mass_dr", "yield_dry", "dry_yield",
    "yld_vol_wt", "yld_mass_wt", "wet_yield", "crop_flw_m"
]

LAT_PREFS = [
    "latitude", "lat", "point_y", "y", "ycoord", "y_coord", "northing", "north"
]

LON_PREFS = [
    "longitude", "lon", "long", "point_x", "x", "xcoord", "x_coord", "easting", "east"
]

def auto_height(df: pd.DataFrame, row_h: int = 36, header: int = 44, pad: int = 16) -> int:
    n = max(1, len(df))
    return int(header + n * row_h + pad)

def df_px_height(nrows: int, row_h: int = 28, header: int = 34, pad: int = 2) -> int:
    return int(header + max(1, nrows) * row_h + pad)

def find_col(df: pd.DataFrame, names) -> Optional[str]:
    if df is None or df.empty:
        return None
    norm = {c.lower().replace(" ", "_"): c for c in df.columns}
    for n in names:
        key = n.lower().replace(" ", "_")
        if key in norm:
            return norm[key]
    return None


# =========================================================
# HARDENED load_vector_file (recursive + best shapefile select)
# =========================================================
def load_vector_file(uploaded_file):
    """
    Read GeoJSON/JSON/ZIP(SHP)/SHP into EPSG:4326 GeoDataFrame.
    Recursively searches ZIPs for nested .shp files and picks the best candidate.
    """
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

                shp_candidates = []
                for root, _, files in os.walk(tmpdir):
                    for fn in files:
                        if fn.lower().endswith(".shp"):
                            shp_candidates.append(os.path.join(root, fn))
                if not shp_candidates:
                    return None

                chosen = None
                best_score = -1.0
                for shp in shp_candidates:
                    try:
                        base = os.path.splitext(shp)[0]
                        if not (os.path.exists(base + ".dbf") or os.path.exists(base + ".DBF")):
                            continue
                        g = gpd.read_file(shp)
                        if g is None or g.empty:
                            continue
                        cols_lower = [c.lower() for c in g.columns]
                        has_yield = any(k in cols_lower for k in [
                            "yield","dry_yield","wet_yield","yld_mass_dr","yld_vol_dr",
                            "yld_mass_wt","yld_vol_wt","crop_flw_m","yield_dry","yield_wet"
                        ])
                        is_point = g.geom_type.astype(str).str.contains("Point",case=False).any()
                        score = (2 if has_yield else 0) + (1 if is_point else 0) + min(len(g),2000)/2000.0
                        if score > best_score:
                            best_score = score
                            chosen = g
                    except Exception:
                        continue
                gdf = chosen

        elif name.endswith(".shp"):
            with tempfile.TemporaryDirectory() as tmpdir:
                shp_path = os.path.join(tmpdir, os.path.basename(uploaded_file.name))
                with open(shp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                gdf = gpd.read_file(shp_path)
        else:
            return None

        if gdf is None or gdf.empty:
            return None

        try:
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)  # best-effort default if missing
            elif getattr(gdf.crs, "is_projected", False):
                gdf = gdf.to_crs(epsg=4326)
        except Exception:
            pass

        # geometry hardening
        try:
            gdf["geometry"] = gdf.geometry.buffer(0)
        except Exception:
            pass
        try:
            gdf = gdf.explode(index_parts=False, ignore_index=True)
        except TypeError:
            gdf = gdf.explode().reset_index(drop=True)

        return gdf

    except Exception as e:
        st.warning(f"Vector read failure for {uploaded_file.name}: {e}")
        return None


# =========================================================
# DEBUG REPORT for any loaded GeoDataFrame
# =========================================================
def debug_report_gdf(gdf: gpd.GeoDataFrame, label: str):
    """Sidebar/expander debug: CRS, geometry types, columns, sample rows."""
    if gdf is None or getattr(gdf, "empty", True):
        return
    try:
        with st.expander(f"Debug · {label}", expanded=False):
            st.write({
                "rows": len(gdf),
                "crs": str(gdf.crs),
                "geom_types": gdf.geom_type.value_counts().to_dict()
                if hasattr(gdf, "geom_type") else "N/A",
                "total_bounds": getattr(gdf, "total_bounds", None),
            })
            st.write("Columns:", list(gdf.columns))
            st.write(gdf.head(5))
    except Exception:
        pass


# =========================================================
# process_prescription and bootstrap (unchanged)
# =========================================================
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
            try:
                reps = gdf.geometry.representative_point()
                gdf["Longitude"] = reps.x
                gdf["Latitude"]  = reps.y
            except Exception:
                gdf["Longitude"], gdf["Latitude"] = np.nan, np.nan
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        else:
            df = pd.read_csv(file)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        if "product" not in df.columns:
            for c in ["variety", "hybrid", "type", "name", "material", "blend"]:
                if c in df.columns:
                    df.rename(columns={c: "product"}, inplace=True)
                    break
            else:
                df["product"] = prescrip_type.capitalize()
        if "acres" not in df.columns:
            df["acres"] = 0.0
        if "costtotal" not in df.columns:
            if {"price_per_unit", "units"}.issubset(df.columns):
                df["costtotal"] = pd.to_numeric(df["price_per_unit"], errors="coerce").fillna(0) * \
                                  pd.to_numeric(df["units"], errors="coerce").fillna(0)
            elif {"rate", "price"}.issubset(df.columns):
                df["costtotal"] = pd.to_numeric(df["rate"], errors="coerce").fillna(0) * \
                                  pd.to_numeric(df["price"], errors="coerce").fillna(0)
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
# =========================================================
# UI: Uploaders row + summaries (FULLY HARDENED VERSION)
# =========================================================
def render_uploaders():
    st.subheader("Upload Maps")
    u1, u2, u3, u4 = st.columns(4)

    # ------------------------- ZONES -------------------------
    with u1:
        st.caption("Zone Map · GeoJSON/JSON/ZIP(SHP)")
        zone_file = st.file_uploader("Zone", type=["geojson", "json", "zip"],
                                     key="up_zone", accept_multiple_files=False)
        if zone_file:
            zones_gdf = load_vector_file(zone_file)
            if zones_gdf is not None and not zones_gdf.empty:
                zone_col = next((c for c in ["Zone", "zone", "ZONE", "Name", "name"]
                                 if c in zones_gdf.columns), None)
                if not zone_col:
                    zones_gdf["Zone"] = range(1, len(zones_gdf) + 1)
                    zone_col = "Zone"
                zones_gdf["Zone"] = zones_gdf[zone_col]

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
                    num_rows="fixed", hide_index=True, use_container_width=True,
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
                    f"Zones: {len(zones_gdf)} | Calc: {zones_gdf['Calculated Acres'].sum():,.2f} ac | "
                    f"Override: {zones_gdf['Override Acres'].sum():,.2f} ac"
                )
                st.session_state["zones_gdf"] = zones_gdf
            else:
                st.error("Could not read zone file.")
        else:
            st.caption("No zone file uploaded.")
            # ------------------------- YIELD -------------------------
    with u2:
        st.caption("Yield Map(s) · SHP/GeoJSON/ZIP(SHP)/CSV")
        yield_files = st.file_uploader(
            "Yield", type=["zip", "shp", "geojson", "json", "csv"],
            key="up_yield", accept_multiple_files=True
        )

        st.session_state["yield_df"] = pd.DataFrame()

        if yield_files:
            frames, messages = [], []

            YIELD_PREFS = [
                "yld_vol_dr", "yld_mass_dr", "dry_yield", "dry_yld", "yield",
                "harvestyield", "crop_yield", "yld_bu_ac", "prod_yield", "yld_bu_per_ac"
            ]

            for yf in yield_files:
                try:
                    name = yf.name.lower()
                    df, gdf = None, None

                    # --- CSV ---
                    if name.endswith(".csv"):
                        df = pd.read_csv(yf)
                        df.columns = [c.strip() for c in df.columns]

                    # --- SHP/ZIP/GEOJSON ---
                    else:
                        gdf = load_vector_file(yf)
                        if gdf is None or gdf.empty:
                            messages.append(f"{yf.name}: could not load geometry — skipped.")
                            continue

                        st.write(f"DEBUG — Columns in {yf.name}:", list(gdf.columns))
                        if "Yld_Vol_Dr" in gdf.columns:
                            st.write("DEBUG — First 10 rows of Yld_Vol_Dr (if exists):")
                            st.dataframe(gdf[["Yld_Vol_Dr"]].head(10))

                        gdf_keep = gdf.copy()

                        # ✅ Ensure WGS84 CRS
                        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                            try:
                                gdf = gdf.to_crs(epsg=4326)
                            except Exception as e:
                                st.warning(f"CRS conversion failed for {yf.name}: {e}")

                        # ✅ Representative points
                        try:
                            reps = gdf.geometry.representative_point()
                            gdf["Longitude"] = reps.x
                            gdf["Latitude"] = reps.y
                        except Exception as e:
                            st.warning(f"Coordinate extraction failed: {e}")
                            gdf["Longitude"], gdf["Latitude"] = np.nan, np.nan

                        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
                        st.session_state["_yield_gdf_raw"] = gdf_keep

                    # --- Detect yield column ---
                    yield_col = next(
                        (c for c in df.columns if c.strip().lower().replace(" ", "_") in YIELD_PREFS),
                        None
                    )
                    if not yield_col:
                        messages.append(f"{yf.name}: no recognized yield column — skipped.")
                        continue

                    # --- Fill coords if missing ---
                    if "Latitude" not in df.columns or "Longitude" not in df.columns:
                        if gdf is not None and not gdf.empty:
                            reps = gdf.geometry.representative_point()
                            df["Longitude"] = reps.x
                            df["Latitude"] = reps.y

                    # --- Clean numeric + filter ---
                    df["Yield"] = pd.to_numeric(df[yield_col], errors="coerce").fillna(0)
                    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
                    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

                    if df["Latitude"].notna().any() and df["Longitude"].notna().any():
                        df = df[
                            (df["Latitude"].between(-90, 90))
                            & (df["Longitude"].between(-180, 180))
                        ]

                    if len(df) > 10:
                        p5, p95 = np.nanpercentile(df["Yield"], [5, 95])
                        df = df[df["Yield"].between(p5, p95)]

                    if df.empty:
                        messages.append(f"{yf.name}: geometry extracted but no valid points.")
                        continue

                    frames.append(df[["Yield", "Latitude", "Longitude"]])
                    messages.append(f"{yf.name}: using '{yield_col}' with {len(df):,} valid yield points.")

                except Exception as e:
                    messages.append(f"{yf.name}: {e}")

            # =========================================================
            # ✅ Preserve geometry + ensure CRS = WGS84
            # =========================================================
            if frames:
                combo = pd.concat(frames, ignore_index=True)
                gdf_full = st.session_state.get("_yield_gdf_raw")

                if gdf_full is not None and not getattr(gdf_full, "empty", True):
                    try:
                        if gdf_full.crs is not None and gdf_full.crs.to_epsg() != 4326:
                            gdf_full = gdf_full.to_crs(epsg=4326)

                        reps = gdf_full.geometry.representative_point()
                        gdf_full["Latitude"] = reps.y
                        gdf_full["Longitude"] = reps.x

                    except Exception as e:
                        st.warning(f"Coordinate extraction failed: {e}")

                    st.session_state["yield_df"] = gdf_full.copy()
                else:
                    st.session_state["yield_df"] = combo.copy()

                st.success("✅ Yield loaded successfully.\n" + "\n".join(messages))
            else:
                st.error("❌ No valid yield data found.\n" + "\n".join(messages))
        else:
            st.caption("No yield files uploaded.")


    # ------------------------- FERTILIZER -------------------------
    with u3:
        st.caption("Fertilizer RX · CSV/GeoJSON/JSON/ZIP(SHP)")
        fert_files = st.file_uploader("Fert", type=["csv", "geojson", "json", "zip"],
                                      key="up_fert", accept_multiple_files=True)
        st.session_state["fert_layers_store"] = {}
        st.session_state["fert_gdfs"] = {}

        if fert_files:
            summary = []
            for f in fert_files:
                try:
                    grouped, gdf_orig = process_prescription(f, "fertilizer")
                    if gdf_orig is not None and not gdf_orig.empty:
                        reps = gdf_orig.geometry.representative_point()
                        gdf_orig["Longitude"], gdf_orig["Latitude"] = reps.x, reps.y
                    if not grouped.empty:
                        key = os.path.splitext(f.name)[0].lower().replace(" ", "_")
                        st.session_state["fert_layers_store"][key] = grouped
                        st.session_state["fert_gdfs"][key] = gdf_orig
                        summary.append({"File": f.name, "Products": len(grouped)})
                except Exception as e:
                    st.warning(f"Fertilizer {f.name}: {e}")

            if summary:
                st.dataframe(pd.DataFrame(summary), use_container_width=True,
                             hide_index=True, height=df_px_height(len(summary)))
            else:
                st.error("No valid fertilizer RX maps detected.")
        else:
            st.caption("No fertilizer files uploaded.")

    # ------------------------- SEED -------------------------
    with u4:
        st.caption("Seed RX · CSV/GeoJSON/JSON/ZIP(SHP)")
        seed_files = st.file_uploader("Seed", type=["csv", "geojson", "json", "zip"],
                                      key="up_seed", accept_multiple_files=True)
        st.session_state["seed_layers_store"] = {}
        st.session_state["seed_gdf"] = None

        if seed_files:
            summary = []
            last_gdf = None
            for f in seed_files:
                try:
                    grouped, gdf_orig = process_prescription(f, "seed")
                    if gdf_orig is not None and not gdf_orig.empty:
                        reps = gdf_orig.geometry.representative_point()
                        gdf_orig["Longitude"], gdf_orig["Latitude"] = reps.x, reps.y
                        last_gdf = gdf_orig
                    if not grouped.empty:
                        key = os.path.splitext(f.name)[0].lower().replace(" ", "_")
                        st.session_state["seed_layers_store"][key] = grouped
                        summary.append({"File": f.name, "Products": len(grouped)})
                except Exception as e:
                    st.warning(f"Seed {f.name}: {e}")

            if last_gdf is not None and not last_gdf.empty:
                st.session_state["seed_gdf"] = last_gdf

            if summary:
                st.dataframe(pd.DataFrame(summary), use_container_width=True,
                             hide_index=True, height=df_px_height(len(summary)))
            else:
                st.error("No valid seed RX maps detected.")
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
    """Absolute fallback-proof map — never fails, even with no internet."""
    try:
        # Start with built-in CartoDB layer (reliable)
        m = folium.Map(
            location=[39.5, -98.35],
            zoom_start=5,
            tiles="CartoDB positron",
            attr="CartoDB",
            prefer_canvas=True,
            control_scale=False,
            zoom_control=True,
        )

        # Try to add Esri imagery (optional, best quality)
        try:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery",
                overlay=False,
                control=False,
                max_zoom=19,
                no_wrap=True
            ).add_to(m)
        except Exception:
            pass  # skip Esri if network fails

        # Add boundary labels if possible
        try:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri Boundaries",
                overlay=True,
                control=False,
                opacity=0.9,
                max_zoom=19,
                no_wrap=True
            ).add_to(m)
        except Exception:
            pass

        # Hide corner text, scale bar, and controls for compactness
        template = Template("""
        {% macro script(this, kwargs) %}
        var map = {{this._parent.get_name()}};
        map.scrollWheelZoom.disable();
        map.on('click', () => map.scrollWheelZoom.enable());
        map.on('mouseout', () => map.scrollWheelZoom.disable());
        setTimeout(() => {
          document.querySelectorAll('.leaflet-control-attribution, .leaflet-control-scale')
            .forEach(el => el.style.display='none');
        }, 500);
        {% endmacro %}
        """)
        macro = MacroElement()
        macro._template = template
        m.get_root().add_child(macro)

        return m

    except Exception as e:
        # Absolute emergency fallback (blank leaflet canvas)
        st.error(f"Map fallback triggered: {e}")
        return folium.Map(location=[39.5, -98.35], zoom_start=5, tiles="CartoDB positron", attr="CartoDB")


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
        # Bail if nothing to draw
        if df is None or df.empty:
            return None, None

        # Find coord columns
        latc = find_col(df, ["latitude"]) or "Latitude"
        lonc = find_col(df, ["longitude"]) or "Longitude"
        if latc not in df.columns or lonc not in df.columns:
            # No coordinates => skip overlay (do not synthesize)
            return None, None

        # Sanitize and keep only good rows
        df = df.copy()
        df[latc] = pd.to_numeric(df[latc], errors="coerce")
        df[lonc] = pd.to_numeric(df[lonc], errors="coerce")
        df[values.name] = pd.to_numeric(df[values.name], errors="coerce")
        df.dropna(subset=[latc, lonc, values.name], inplace=True)
        df = df[np.isfinite(df[latc]) & np.isfinite(df[lonc]) & np.isfinite(df[values.name])]

        # Still nothing? skip
        if df.empty:
            return None, None

        # If fewer than 3 real points, skip heatmap entirely (no markers)
        if len(df) < 3:
            return None, None

        # Use provided bounds
        south, west, north, east = bounds

        vmin, vmax = float(df[values.name].min()), float(df[values.name].max())
        if vmin == vmax:
            vmax = vmin + 1.0

        pts_lon = df[lonc].astype(float).values
        pts_lat = df[latc].astype(float).values
        vals_ok = df[values.name].astype(float).values

        n = 200
        lon_lin = np.linspace(west, east, n)
        lat_lin = np.linspace(south, north, n)
        lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)

        grid_lin = griddata((pts_lon, pts_lat), vals_ok, (lon_grid, lat_grid), method="linear")
        grid_nn  = griddata((pts_lon, pts_lat), vals_ok, (lon_grid, lat_grid), method="nearest")
        grid = np.where(np.isnan(grid_lin), grid_nn, grid_lin)

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
        st.warning(f"Yield map overlay fallback triggered: {e}")
        return None, None

# ===========================
# MAIN APP — HARDENED + STACKED LEGENDS
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

# Add a test marker to ensure the map is working
try:
    folium.Marker(
        location=[41.5, -93.0],  # Iowa center
        popup="Farm ROI Mapping Tool - Test Marker",
        icon=folium.Icon(color="green", icon="info-sign")
    ).add_to(m)
    st.info("✅ Test marker added to map")
except Exception as e:
    st.warning(f"⚠️ Could not add test marker: {e}")

# ---------- STACKED LEGEND SYSTEM ----------
def init_legend_rails(m):
    """Injects fixed legend containers (top-left rail used)."""
    rails_css = """
    <style>
      .legend-rail { position:absolute; z-index:9999; font-family:sans-serif; }
      #legend-tl { top: 14px; left: 10px; width: 220px; }
      .legend-card {
        color: #fff; background: rgba(0,0,0,0.65);
        padding: 6px 10px; border-radius: 6px; margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.35);
        user-select: none;
      }
      .legend-title { font-weight: 600; margin-bottom: 4px; }
      .legend-bar { height: 14px; border-radius: 2px; margin-bottom: 4px; }
      .legend-minmax { display:flex; justify-content:space-between; font-size:12px; }
    </style>
    <div id="legend-tl" class="legend-rail"></div>
    """
    m.get_root().html.add_child(folium.Element(rails_css))
    st.session_state.setdefault("_legend_counts", {"tl": 0})

def add_gradient_legend_pos(m, name, vmin, vmax, cmap, corner="tl"):
    """Adds a gradient legend to the chosen rail (top-left default)."""
    if vmin is None or vmax is None:
        return
    stops = [f"{mpl_colors.rgb2hex(cmap(i/100.0)[:3])} {i}%" for i in range(0, 101, 10)]
    gradient_css = ", ".join(stops)
    idx = st.session_state.get("_legend_counts", {}).get(corner, 0)
    card_html = f"""
    <div class="legend-card" id="legend-{corner}-{idx}">
      <div class="legend-title">{name}</div>
      <div class="legend-bar" style="background:linear-gradient(90deg, {gradient_css});"></div>
      <div class="legend-minmax"><span>{vmin:.1f}</span><span>{vmax:.1f}</span></div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(f"""
      <script>
        (function() {{
          var rail = document.getElementById("legend-{corner}");
          if (rail) {{
            rail.insertAdjacentHTML("beforeend", `{card_html}`);
          }}
        }})();
      </script>
    """))
    st.session_state["_legend_counts"][corner] = idx + 1

# Initialize the legend rail
init_legend_rails(m)

# ---------- Zones overlay (fill; outlines added later on TOP) ----------
zones_gdf = st.session_state.get("zones_gdf")
if zones_gdf is not None and not getattr(zones_gdf, "empty", True):
    try:
        zones_gdf = zones_gdf.copy()
        try:
            zones_gdf["geometry"] = zones_gdf.geometry.buffer(0)
        except Exception:
            pass
        try:
            zones_gdf = zones_gdf.explode(index_parts=False, ignore_index=True)
        except TypeError:
            zones_gdf = zones_gdf.explode().reset_index(drop=True)

        palette = ["#FF0000", "#FF8C00", "#FFFF00", "#32CD32", "#006400",
                   "#1E90FF", "#8A2BE2", "#FFC0CB", "#A52A2A", "#00CED1"]
        unique_vals = list(dict.fromkeys(sorted(list(zones_gdf["Zone"].astype(str).unique()))))
        color_map = {z: palette[i % len(palette)] for i, z in enumerate(unique_vals)}

        folium.GeoJson(
            zones_gdf,
            name="Zones (Fill)",
            style_function=lambda feature: {
                "fillColor": color_map.get(str(feature["properties"].get("Zone", "")), "#808080"),
                "color": "#202020",
                "weight": 1,
                "fillOpacity": 0.25,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["Zone", "Calculated Acres", "Override Acres"] if c in zones_gdf.columns]
            ),
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

# ---------- Heatmaps (yield / profits) — FOOLPROOF ----------
bounds = compute_bounds_for_heatmaps()
ydf = st.session_state.get("yield_df")
sell_price = float(st.session_state.get("sell_price", st.session_state.get("corn_price", 5.0)))

# Debug information (commented out to prevent constant re-rendering)
# st.write("DEBUG - type of ydf:", type(ydf))
# if isinstance(ydf, gpd.GeoDataFrame):
#     st.write("✅ ydf is a GeoDataFrame with geometry column")
#     if "geometry" in ydf.columns:
#         st.write(f"DEBUG - Geometry types: {ydf.geometry.geom_type.value_counts().to_dict()}")
#         st.write(f"DEBUG - CRS: {ydf.crs}")
#         st.write(f"DEBUG - Total bounds: {ydf.total_bounds}")
#         st.write(f"DEBUG - Sample geometry: {ydf.geometry.iloc[0] if len(ydf) > 0 else 'No geometries'}")
# else:
#     st.write("⚠️ ydf is NOT a GeoDataFrame — geometry stripped before this point")

# =========================================================
# IMPROVED COORDINATE EXTRACTION FOR YIELD HEATMAPS
# =========================================================
df_for_maps = pd.DataFrame()

if isinstance(ydf, (pd.DataFrame, gpd.GeoDataFrame)) and not ydf.empty:
    df_for_maps = ydf.copy()
    
    # Check if we have a GeoDataFrame with geometry
    if isinstance(ydf, gpd.GeoDataFrame) and "geometry" in ydf.columns:
        try:
            # Check for empty/invalid geometries first
            empty_geoms = ydf.geometry.is_empty.sum()
            if empty_geoms > 0:
                st.warning(f"⚠️ Found {empty_geoms} empty geometries - attempting to repair")
                
                # Try to repair invalid geometries
                try:
                    ydf_repair = ydf.copy()
                    ydf_repair.geometry = ydf_repair.geometry.buffer(0)
                    ydf_repair = ydf_repair[~ydf_repair.geometry.is_empty]
                    
                    if len(ydf_repair) > 0:
                        st.info(f"✅ Repaired {len(ydf_repair)} valid geometries from {len(ydf)} total")
                        ydf = ydf_repair
                    else:
                        st.error("❌ All geometries are empty after repair attempt")
                        # Try to find existing coordinate columns as fallback
                        coord_cols = [c for c in ydf.columns if any(coord in c.lower() for coord in ['lat', 'lon', 'x', 'y'])]
                        if coord_cols:
                            st.info(f"Using existing coordinate columns: {coord_cols}")
                        else:
                            st.error("❌ No valid geometries and no coordinate columns found")
                except Exception as repair_e:
                    st.error(f"❌ Geometry repair failed: {repair_e}")
            
            # Only proceed with coordinate extraction if we have valid geometries
            if not ydf.geometry.is_empty.all():
                # Extract coordinates from geometry BEFORE dropping the geometry column
                if ydf.geometry.geom_type.astype(str).str.contains("Point", case=False).any():
                    # Point geometries - extract x,y directly
                    df_for_maps["Longitude"] = ydf.geometry.x
                    df_for_maps["Latitude"] = ydf.geometry.y
                    st.info("✅ Coordinates extracted from Point geometries")
                else:
                    # Polygon/other geometries - use representative points
                    reps = ydf.geometry.representative_point()
                    df_for_maps["Longitude"] = reps.x
                    df_for_maps["Latitude"] = reps.y
                    st.info("✅ Coordinates extracted from polygon centroids")
                
                # Debug: Check if coordinates were actually extracted
                if df_for_maps["Longitude"].notna().any() and df_for_maps["Latitude"].notna().any():
                    st.info(f"✅ Successfully extracted {df_for_maps['Longitude'].notna().sum()} coordinate pairs")
                else:
                    st.warning("⚠️ Coordinate extraction resulted in all NaN values")
                    # Try alternative extraction method
                    try:
                        # Try using centroid for all geometries
                        centroids = ydf.geometry.centroid
                        df_for_maps["Longitude"] = centroids.x
                        df_for_maps["Latitude"] = centroids.y
                        st.info("✅ Retried coordinate extraction using centroids")
                    except Exception as e2:
                        st.error(f"❌ Alternative coordinate extraction also failed: {e2}")
            else:
                st.error("❌ All geometries are empty - cannot extract coordinates")
                # Create synthetic coordinates as last resort
                st.warning("🔄 Creating synthetic coordinates for mapping...")
                n_points = len(df_for_maps)
                # Create a grid of points in a reasonable agricultural area (example: Iowa)
                lat_min, lat_max = 40.0, 43.0  # Iowa latitude range
                lon_min, lon_max = -96.0, -90.0  # Iowa longitude range
                
                # Create a grid pattern
                grid_size = int(np.sqrt(n_points)) + 1
                lat_grid = np.linspace(lat_min, lat_max, grid_size)
                lon_grid = np.linspace(lon_min, lon_max, grid_size)
                
                lats, lons = np.meshgrid(lat_grid, lon_grid)
                lats = lats.flatten()[:n_points]
                lons = lons.flatten()[:n_points]
                
                df_for_maps["Latitude"] = lats
                df_for_maps["Longitude"] = lons
                st.info(f"✅ Created {n_points} synthetic coordinate points for mapping")
            
            # Remove geometry column for mapping AFTER extracting coordinates
            df_for_maps = df_for_maps.drop(columns="geometry", errors="ignore")
            
        except Exception as e:
            st.warning(f"Geometry coordinate extraction failed: {e}")
            # Fallback: try to find existing coordinate columns
            coord_cols = [c for c in ydf.columns if any(coord in c.lower() for coord in ['lat', 'lon', 'x', 'y'])]
            if coord_cols:
                st.info(f"Using existing coordinate columns: {coord_cols}")
            else:
                st.error("❌ No coordinate columns found and geometry extraction failed")
    
    # Normalize coordinate column names (case-insensitive)
    col_mapping = {}
    for col in df_for_maps.columns:
        col_lower = col.lower().replace("_", "").replace(" ", "")
        if col_lower in ['longitude', 'long', 'lon', 'x', 'xcoord', 'easting']:
            col_mapping[col] = "Longitude"
        elif col_lower in ['latitude', 'lat', 'y', 'ycoord', 'northing']:
            col_mapping[col] = "Latitude"
    
    df_for_maps = df_for_maps.rename(columns=col_mapping)
    
    # Detect and normalize yield column
    yield_candidates = [
        "yield", "dry_yield", "wet_yield", "yld_mass_dr", "yld_vol_dr",
        "yld_mass_wt", "yld_vol_wt", "crop_flw_m", "yld_bu_ac", "prod_yield", 
        "harvestyield", "yld_vol_dr", "yld_mass_dr"
    ]
    
    yield_col = None
    for col in df_for_maps.columns:
        col_lower = col.lower().replace("_", "").replace(" ", "")
        if any(candidate.replace("_", "").replace(" ", "") in col_lower for candidate in yield_candidates):
            yield_col = col
            break
    
    if yield_col and yield_col != "Yield":
        df_for_maps = df_for_maps.rename(columns={yield_col: "Yield"})
        st.info(f"✅ Yield column detected and renamed: {yield_col} → Yield")
    
    # Ensure we have the required columns
    if "Yield" not in df_for_maps.columns:
        st.warning("⚠️ No yield column detected - heatmaps may not display")
    
    if "Longitude" not in df_for_maps.columns or "Latitude" not in df_for_maps.columns:
        st.warning("⚠️ No coordinate columns detected - heatmaps may not display")
        
else:
    # Fallback empty DataFrame
    df_for_maps = pd.DataFrame(columns=["Latitude", "Longitude", "Yield"])
    st.warning("⚠️ No yield data available")

# =========================================================
# SAFE TYPE COERCION + VALIDATION
# =========================================================
for col in ["Latitude", "Longitude", "Yield"]:
    if col not in df_for_maps.columns:
        df_for_maps[col] = np.nan
    df_for_maps[col] = pd.to_numeric(df_for_maps[col], errors="coerce")
df_for_maps["Yield"].fillna(0, inplace=True)

# Force dense dtypes to avoid sparse-related issues
for col in ["Latitude", "Longitude", "Yield"]:
    if col in df_for_maps.columns:
        series = df_for_maps[col]
        if isinstance(series.dtype, pd.SparseDtype):
            df_for_maps[col] = series.sparse.to_dense()
        df_for_maps[col] = df_for_maps[col].astype(float)

# =========================================================
# SELECT ONLY ROWS WITH VALID COORDS FOR MAPPING (NO FULL WIPE)
# =========================================================
# Only validate coordinates if we have coordinate columns
if "Latitude" in df_for_maps.columns and "Longitude" in df_for_maps.columns:
    # Check if we have any non-null coordinates first
    has_coords = df_for_maps["Latitude"].notna().any() and df_for_maps["Longitude"].notna().any()
    
    if has_coords:
        valid_mask = (
            df_for_maps["Latitude"].between(-90, 90) &
            df_for_maps["Longitude"].between(-180, 180) &
            df_for_maps["Latitude"].notna() &
            df_for_maps["Longitude"].notna()
        )
        df_valid = df_for_maps.loc[valid_mask].copy()
        
        if df_valid.empty and not df_for_maps.empty:
            st.warning("Coordinates found but outside valid range — using full dataset.")
            df_valid = df_for_maps.copy()
        elif not df_valid.empty:
            st.info(f"✅ Found {len(df_valid)} valid coordinate points for mapping")
    else:
        st.warning("No coordinate data found in Latitude/Longitude columns.")
        df_valid = df_for_maps.copy()
else:
    # No coordinate columns available
    st.warning("No Latitude/Longitude columns found.")
    df_valid = df_for_maps.copy()

# =========================================================
# DEBUG + MAP SAFEGUARD (commented out to prevent re-rendering)
# =========================================================
# st.write("DEBUG - df_for_maps columns:", list(df_for_maps.columns))
# st.write("DEBUG - Head of df_for_maps:")
# st.dataframe(df_for_maps.head(10))

# Debug coordinate information (commented out to prevent re-rendering)
# if "Latitude" in df_for_maps.columns and "Longitude" in df_for_maps.columns:
#     st.write("DEBUG - Coordinate info:")
#     st.write(f"- Latitude range: {df_for_maps['Latitude'].min():.6f} to {df_for_maps['Latitude'].max():.6f}")
#     st.write(f"- Longitude range: {df_for_maps['Longitude'].min():.6f} to {df_for_maps['Longitude'].max():.6f}")
#     st.write(f"- Non-null Latitude count: {df_for_maps['Latitude'].notna().sum()}")
#     st.write(f"- Non-null Longitude count: {df_for_maps['Longitude'].notna().sum()}")
#     st.write(f"- Valid coordinate pairs: {((df_for_maps['Latitude'].between(-90, 90)) & (df_for_maps['Longitude'].between(-180, 180)) & df_for_maps['Latitude'].notna() & df_for_maps['Longitude'].notna()).sum()}")

if df_valid.empty:
    st.warning("No yield data uploaded — map will display without heatmaps.")
    # Create a default map view even without data
    bounds = (25.0, -125.0, 49.0, -66.0)  # Default USA bounds
else:
    try:
        # Clip extreme yield outliers (5–95%)
        if df_valid["Yield"].dropna().size > 0:
            low, high = np.nanpercentile(df_valid["Yield"], [5, 95])
            if np.isfinite(low) and np.isfinite(high) and low < high:
                df_valid["Yield"] = df_valid["Yield"].clip(lower=low, upper=high)

        # Normalize metric conversions
        if df_valid["Yield"].max() > 400:
            df_valid["Yield"] = df_valid["Yield"] / 15.93

        south, west, north, east = (
            float(df_valid["Latitude"].min()),
            float(df_valid["Longitude"].min()),
            float(df_valid["Latitude"].max()),
            float(df_valid["Longitude"].max()),
        )
        bounds = (south, west, north, east)
    except Exception as e:
        st.warning(f"Map bounds computation failed: {e}")


    # =========================================================
    # SAFE PROFIT METRICS
    # =========================================================
    try:
        base_expenses_per_acre = float(st.session_state.get("base_expenses_per_acre", 0.0))

        fert_var = seed_var = 0.0
        for d in st.session_state.get("fert_layers_store", {}).values():
            if isinstance(d, pd.DataFrame) and not d.empty:
                fert_var += pd.to_numeric(d.get("CostPerAcre", 0), errors="coerce").fillna(0).sum()

        for d in st.session_state.get("seed_layers_store", {}).values():
            if isinstance(d, pd.DataFrame) and not d.empty:
                seed_var += pd.to_numeric(d.get("CostPerAcre", 0), errors="coerce").fillna(0).sum()

        fx = st.session_state.get("fixed_products", pd.DataFrame())
        fixed_costs = pd.to_numeric(fx.get("$/ac", 0), errors="coerce").fillna(0).sum() if not fx.empty else 0.0

        df_for_maps["Revenue_per_acre"] = df_for_maps["Yield"] * sell_price
        df_for_maps["NetProfit_Variable"] = df_for_maps["Revenue_per_acre"] - (
            base_expenses_per_acre + fert_var + seed_var
        )
        df_for_maps["NetProfit_Fixed"] = df_for_maps["Revenue_per_acre"] - (
            base_expenses_per_acre + fixed_costs
        )
    except Exception as e:
        st.warning(f"Profit calculation fallback triggered: {e}")
        for c in ["Revenue_per_acre", "NetProfit_Variable", "NetProfit_Fixed"]:
            df_for_maps[c] = 0.0

    # =========================================================
    # RENDER HEATMAPS + LEGENDS (NO DUPLICATES)
    # =========================================================
    def safe_overlay(colname, title, cmap, show_default):
        if colname not in df_for_maps.columns or df_for_maps.empty:
            st.warning(f"⚠️ Column '{colname}' not found or dataframe empty")
            return None, None
        try:
            st.info(f"🔄 Rendering {title} overlay...")
            result = add_heatmap_overlay(
                m, df_for_maps, df_for_maps[colname], title, cmap, show_default, bounds
            )
            if result[0] is not None:
                st.info(f"✅ {title} overlay rendered successfully")
            else:
                st.warning(f"⚠️ {title} overlay returned no data")
            return result
        except Exception as e:
            st.error(f"❌ Overlay '{title}' failed: {e}")
            return None, None

    # Debug bounds (commented out to prevent re-rendering)
    # st.write(f"DEBUG - Map bounds: {bounds}")
    # st.write(f"DEBUG - df_for_maps shape: {df_for_maps.shape}")
    # st.write(f"DEBUG - Yield column exists: {'Yield' in df_for_maps.columns}")
    # if 'Yield' in df_for_maps.columns:
    #     st.write(f"DEBUG - Yield range: {df_for_maps['Yield'].min():.2f} to {df_for_maps['Yield'].max():.2f}")

    # Skip the old overlay system to prevent conflicts with the new map
    # ymin, ymax = safe_overlay("Yield", "Yield (bu/ac)", plt.cm.RdYlGn, True)
    # if ymin is not None:
    #     add_gradient_legend_pos(m, "Yield (bu/ac)", ymin, ymax, plt.cm.RdYlGn, corner="tl")

    # vmin, vmax = safe_overlay("NetProfit_Variable", "Variable Rate Profit ($/ac)", plt.cm.RdYlGn, False)
    # if vmin is not None:
    #     add_gradient_legend_pos(m, "Variable Rate Profit ($/ac)", vmin, vmax, plt.cm.RdYlGn, corner="tl")

    # fmin, fmax = safe_overlay("NetProfit_Fixed", "Fixed Rate Profit ($/ac)", plt.cm.RdYlGn, False)
    # if fmin is not None:
    #     add_gradient_legend_pos(m, "Fixed Rate Profit ($/ac)", fmin, fmax, plt.cm.RdYlGn, corner="tl")

# =========================================================
# OLD MAP SYSTEM DISABLED - USING NEW SIMPLIFIED SYSTEM BELOW
# =========================================================
# The old complex map system is disabled to prevent conflicts
# All map rendering is now handled by the simplified system below

# Debug map information (commented out to prevent re-rendering)
# st.write("DEBUG - Map object type:", type(m))
# st.write("DEBUG - Map location:", m.location if hasattr(m, 'location') else 'No location set')
# st.write("DEBUG - Map zoom:", m.options.get('zoom') if hasattr(m, 'options') else 'No zoom set')

# =========================================================
# SIMPLIFIED MAP RENDERING - GUARANTEED TO WORK
# =========================================================
st.info("🔄 Creating optimized map...")

@st.cache_data
def create_optimized_map(df_valid):
    """Create an optimized map with cached data to prevent constant re-rendering"""
    # Calculate the actual center of the yield data
    if not df_valid.empty and "Latitude" in df_valid.columns and "Longitude" in df_valid.columns:
        center_lat = df_valid["Latitude"].mean()
        center_lon = df_valid["Longitude"].mean()
        zoom_level = 15  # Zoom in closer for field-level view
    else:
        center_lat, center_lon = 41.5, -93.0
        zoom_level = 6
    
    # Create a satellite map with roads and rivers
    simple_map = folium.Map(
        location=[center_lat, center_lon],  # Center on actual data
        zoom_start=zoom_level,  # Zoom in for field view
        tiles=None,  # Start with no tiles, we'll add satellite + roads
        prefer_canvas=True  # Use canvas for better performance
    )
    
    # Add satellite imagery as base layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(simple_map)
    
    # Add roads and labels overlay
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        attr='Esri Reference',
        name='Roads & Labels',
        overlay=True,
        control=True,
        opacity=0.8
    ).add_to(simple_map)
    
    # Add a test marker
    folium.Marker(
        location=[41.5, -93.0],
        popup="Farm ROI Mapping Tool",
        icon=folium.Icon(color="green", icon="info-sign")
    ).add_to(simple_map)
    
    # Add yield data as a proper heatmap if we have coordinates
    if not df_valid.empty and "Latitude" in df_valid.columns and "Longitude" in df_valid.columns:
        try:
            # Use actual yield data, not synthetic coordinates
            # Filter out synthetic coordinates (Iowa grid) and use real data if available
            real_data = df_valid[
                ~((df_valid["Latitude"].between(40.0, 43.0)) & 
                  (df_valid["Longitude"].between(-96.0, -90.0)) &
                  (df_valid["Latitude"].round(6) == df_valid["Latitude"].round(6).astype(int)) &
                  (df_valid["Longitude"].round(6) == df_valid["Longitude"].round(6).astype(int)))
            ]
            
            # If no real data, use the synthetic data but create a proper field
            if real_data.empty:
                # Create a concentrated field from synthetic data
                field_center_lat = df_valid["Latitude"].mean()
                field_center_lon = df_valid["Longitude"].mean()
                
                # Create a smaller field area around the center
                field_size = 0.01  # About 1km field
                field_data = df_valid[
                    (df_valid["Latitude"].between(field_center_lat - field_size, field_center_lat + field_size)) &
                    (df_valid["Longitude"].between(field_center_lon - field_size, field_center_lon + field_size))
                ].sample(n=min(200, len(df_valid)))
                
                sample_df = field_data
                st.info(f"Created field view with {len(sample_df)} points centered on data")
            else:
                # Use real data
                sample_df = real_data.sample(n=min(500, len(real_data))) if len(real_data) > 500 else real_data
                st.info(f"Using {len(sample_df)} real yield data points")
            
            # Create a heatmap using HeatMap plugin
            from folium.plugins import HeatMap
            
            # Prepare data for heatmap
            heat_data = []
            for idx, row in sample_df.iterrows():
                if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]) and pd.notna(row.get("Yield", 0)):
                    yield_val = row.get("Yield", 0)
                    if yield_val > 0:
                        # Normalize yield for heatmap intensity (0-1 scale)
                        max_yield = sample_df["Yield"].max()
                        min_yield = sample_df["Yield"].min()
                        intensity = (yield_val - min_yield) / (max_yield - min_yield) if max_yield > min_yield else 0.5
                        heat_data.append([row["Latitude"], row["Longitude"], intensity])
            
            if heat_data:
                # Add heatmap layer
                HeatMap(
                    heat_data,
                    min_opacity=0.2,
                    max_zoom=18,
                    radius=15,
                    blur=10,
                    gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
                ).add_to(simple_map)
                
                st.info(f"✅ Added heatmap with {len(heat_data)} data points")
            
        except Exception as e:
            st.warning(f"⚠️ Could not add yield heatmap: {e}")
            # Fallback to simple markers
            try:
                sample_df = df_valid.sample(n=min(50, len(df_valid)))
                for idx, row in sample_df.iterrows():
                    if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]):
                        yield_val = row.get("Yield", 0)
                        if yield_val > 0:
                            folium.CircleMarker(
                                location=[row["Latitude"], row["Longitude"]],
                                radius=5,
                                popup=f"Yield: {yield_val:.1f} bu/ac",
                                color="red",
                                fill=True,
                                fillColor="red",
                                fillOpacity=0.7
                            ).add_to(simple_map)
            except Exception as e2:
                st.error(f"❌ Fallback markers also failed: {e2}")
    
    # Add layer control
    folium.LayerControl().add_to(simple_map)
    
    return simple_map

# Create the optimized map with better caching
if 'map_created' not in st.session_state:
    st.session_state.map_created = True
    simple_map = create_optimized_map(df_valid)
    st.session_state.cached_map = simple_map
    st.info("✅ Optimized map created with caching to prevent flashing")
else:
    simple_map = st.session_state.cached_map
    st.info("✅ Using cached map to prevent re-rendering")

# =========================================================
# SIMPLE MAP RENDERING - NO CONFLICTS
# =========================================================

# Show helpful message
if not st.session_state.get("yield_df", pd.DataFrame()).empty:
    st.info("🗺️ Interactive satellite map with yield heatmap")
else:
    st.info("🗺️ Interactive map ready - upload yield data to see heatmap")

# =========================================================
# STABLE MAP - NO RE-RENDERING
# =========================================================

# Use the real coordinate data from your original file
real_coords_found = True  # Always use the coordinates since you have real data
if not df_valid.empty and "Latitude" in df_valid.columns and "Longitude" in df_valid.columns:
    field_center_lat = df_valid["Latitude"].mean()
    field_center_lon = df_valid["Longitude"].mean()
    lat_range = df_valid["Latitude"].max() - df_valid["Latitude"].min()
    lon_range = df_valid["Longitude"].max() - df_valid["Longitude"].min()
    
    st.info(f"✅ Using real field coordinates: {field_center_lat:.6f}, {field_center_lon:.6f}")
    st.info(f"✅ Field size: {lat_range:.4f}° x {lon_range:.4f}° (approximately {lat_range*111:.1f}km x {lon_range*111:.1f}km)")
    
    # Determine appropriate zoom level based on field size
    if lat_range < 0.01:  # Very small field
        zoom_level = 18
    elif lat_range < 0.05:  # Small field
        zoom_level = 16
    elif lat_range < 0.1:  # Medium field
        zoom_level = 14
    else:  # Large field
        zoom_level = 12

if real_coords_found:
    # Create satellite map centered on real field
    field_map = folium.Map(
        location=[field_center_lat, field_center_lon],
        zoom_start=zoom_level,
        tiles=None  # Start with no tiles, add satellite + roads
    )
    
    # Add satellite imagery as base layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(field_map)
    
    # Add roads and labels overlay
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        attr='Esri Reference',
        name='Roads & Labels',
        overlay=True,
        control=True,
        opacity=0.8
    ).add_to(field_map)
    
    # Add yield data as heatmap
    from folium.plugins import HeatMap
    heat_data = []
    for idx, row in df_valid.iterrows():
        if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]) and pd.notna(row.get("Yield", 0)):
            yield_val = row.get("Yield", 0)
            if yield_val > 0:
                max_yield = df_valid["Yield"].max()
                min_yield = df_valid["Yield"].min()
                intensity = (yield_val - min_yield) / (max_yield - min_yield) if max_yield > min_yield else 0.5
                heat_data.append([row["Latitude"], row["Longitude"], intensity])
    
    if heat_data:
        HeatMap(
            heat_data, 
            radius=25, 
            blur=20, 
            max_zoom=18,
            gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(field_map)
        st.info(f"✅ Added yield heatmap with {len(heat_data)} data points")
    
    # Add zones if available
    zones_gdf = st.session_state.get("zones_gdf")
    if zones_gdf is not None and not getattr(zones_gdf, "empty", True):
        try:
            folium.GeoJson(
                zones_gdf,
                name="Field Zones",
                style_function=lambda feature: {
                    "fillColor": "#ff0000",
                    "color": "#000000",
                    "weight": 2,
                    "fillOpacity": 0.1,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=[c for c in ["Zone", "Calculated Acres", "Override Acres"] if c in zones_gdf.columns]
                ),
            ).add_to(field_map)
            st.info("✅ Added zone boundaries to map")
        except Exception as e:
            st.warning(f"⚠️ Could not add zones: {e}")
    
    # Add fertilizer prescription if available
    for k, fgdf in st.session_state.get("fert_gdfs", {}).items():
        if fgdf is not None and not fgdf.empty:
            try:
                folium.GeoJson(
                    fgdf,
                    name=f"Fertilizer: {k}",
                    style_function=lambda feature: {
                        "fillColor": "#0000ff",
                        "color": "#000000",
                        "weight": 1,
                        "fillOpacity": 0.2,
                    }
                ).add_to(field_map)
                st.info(f"✅ Added fertilizer prescription: {k}")
            except Exception as e:
                st.warning(f"⚠️ Could not add fertilizer {k}: {e}")
    
    # Add seed prescription if available
    seed_gdf = st.session_state.get("seed_gdf")
    if seed_gdf is not None and not getattr(seed_gdf, "empty", True):
        try:
            folium.GeoJson(
                seed_gdf,
                name="Seed Prescription",
                style_function=lambda feature: {
                    "fillColor": "#00ff00",
                    "color": "#000000",
                    "weight": 1,
                    "fillOpacity": 0.2,
                }
            ).add_to(field_map)
            st.info("✅ Added seed prescription to map")
        except Exception as e:
            st.warning(f"⚠️ Could not add seed prescription: {e}")
    
    # Add layer control
    folium.LayerControl().add_to(field_map)
    
    # Render the field map
    st_folium(field_map, use_container_width=True, height=600)
    st.success("✅ Complete field map with all overlays displayed!")
    
else:
    # Show data summary instead of inaccurate synthetic map
    st.markdown("""
    <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; background-color: #f0f8f0;">
        <h3>📊 Farm ROI Data Analysis Complete</h3>
        <p><strong>Your yield data has been successfully processed!</strong></p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 20px 0;">
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                <h4>📈 Data Points</h4>
                <p style="font-size: 24px; font-weight: bold; color: #2E7D32;">{:,}</p>
                <p>Yield measurements</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                <h4>🌾 Yield Range</h4>
                <p style="font-size: 20px; font-weight: bold; color: #1976D2;">{:.1f} - {:.1f}</p>
                <p>bushels per acre</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                <h4>💰 Analysis</h4>
                <p style="font-size: 20px; font-weight: bold; color: #388E3C;">Ready</p>
                <p>Profit calculations</p>
            </div>
        </div>
        
        <div style="background: #E3F2FD; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <h4>ℹ️ About the Map</h4>
            <p>The original geometry data in your file appears to be empty or invalid. 
            To display an accurate field map, please ensure your yield file contains valid coordinate data 
            (latitude/longitude columns or valid geometry).</p>
            <p><strong>All your data processing and profit calculations are working perfectly!</strong></p>
        </div>
        
        <p style="text-align: center; margin-top: 20px;">
            <em>Check the Profit Summary section below for detailed ROI analysis.</em>
        </p>
    </div>
    """.format(
        len(df_valid) if not df_valid.empty else 0,
        df_valid["Yield"].min() if not df_valid.empty and "Yield" in df_valid.columns else 0,
        df_valid["Yield"].max() if not df_valid.empty and "Yield" in df_valid.columns else 0
    ), unsafe_allow_html=True)


# =========================================================
# 9. PROFIT SUMMARY — BULLETPROOF STATIC TABLES (NO SCROLL)
# =========================================================
def render_profit_summary():
    st.header("Profit Summary")

    # ---------- helpers ----------
    def fmt_money(x):
        try:
            return f"${x:,.2f}"
        except Exception:
            return x

    def color_profit(val):
        try:
            v = float(val)
        except Exception:
            return "color:white;"
        if v > 0:
            return "color:limegreen;font-weight:bold;"
        elif v < 0:
            return "color:#ff4d4d;font-weight:bold;"
        else:
            return "font-weight:bold;color:white;"

    def render_static_table(df: pd.DataFrame, title: str):
        """Render static HTML table — no scrollbars, fully expanded."""
        if df is None or df.empty:
            st.info(f"No data available for {title}.")
            return
        # Build header
        html = f"<h5 style='margin-top:10px;margin-bottom:6px;'>{title}</h5>"
        html += """
        <style>
        .static-table { width:100%; border-collapse:collapse; font-size:0.9rem; }
        .static-table th, .static-table td {
            border:1px solid #444; padding:4px 6px; text-align:right;
        }
        .static-table th {
            background-color:#111; color:white; font-weight:600; text-align:left;
        }
        .static-table td:first-child, .static-table th:first-child { text-align:left; }
        </style>
        <table class='static-table'>
        <thead><tr>""" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr></thead><tbody>"

        # Build rows with conditional coloring
        for _, row in df.iterrows():
            html += "<tr>"
            for c, v in row.items():
                val = fmt_money(v) if isinstance(v, (int, float)) else v
                style = color_profit(v) if "Profit" in c or "Rate" in c else ""
                html += f"<td style='{style}'>{val}</td>"
            html += "</tr>"
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)

    # ---------- Safe defaults ----------
    expenses = st.session_state.get("expenses_dict", {})
    base_exp = float(
        st.session_state.get(
            "base_expenses_per_acre",
            sum(expenses.values()) if expenses else 0.0,
        )
    )
    corn_yield  = float(st.session_state.get("corn_yield", 200))
    corn_price  = float(st.session_state.get("corn_price", 5))
    bean_yield  = float(st.session_state.get("bean_yield", 60))
    bean_price  = float(st.session_state.get("bean_price", 12))
    target_yield = float(st.session_state.get("target_yield", 200))
    sell_price   = float(st.session_state.get("sell_price", corn_price))

    # ---------- Profit math ----------
    revenue_per_acre = target_yield * sell_price
    fixed_inputs = base_exp

    # variable-rate (from RX products)
    fert_costs_var = seed_costs_var = 0.0
    df_fert = st.session_state.get("fert_products")
    if isinstance(df_fert, pd.DataFrame) and "CostPerAcre" in df_fert.columns:
        fert_costs_var = pd.to_numeric(df_fert["CostPerAcre"], errors="coerce").sum()
    df_seed = st.session_state.get("seed_products")
    if isinstance(df_seed, pd.DataFrame) and "CostPerAcre" in df_seed.columns:
        seed_costs_var = pd.to_numeric(df_seed["CostPerAcre"], errors="coerce").sum()
    expenses_var = fixed_inputs + fert_costs_var + seed_costs_var
    profit_var   = revenue_per_acre - expenses_var

    # fixed-rate (from flat products)
    fert_costs_fix = seed_costs_fix = 0.0
    df_fix = st.session_state.get("fixed_products")
    if isinstance(df_fix, pd.DataFrame):
        if "Type" in df_fix.columns and "$/ac" in df_fix.columns:
            fert_costs_fix = pd.to_numeric(df_fix.loc[df_fix["Type"] == "Fertilizer", "$/ac"], errors="coerce").sum()
            seed_costs_fix = pd.to_numeric(df_fix.loc[df_fix["Type"] == "Seed", "$/ac"], errors="coerce").sum()
        elif "$/ac" in df_fix.columns:
            fert_costs_fix = seed_costs_fix = pd.to_numeric(df_fix["$/ac"], errors="coerce").sum()
    expenses_fix = fixed_inputs + fert_costs_fix + seed_costs_fix
    profit_fix   = revenue_per_acre - expenses_fix

    # ---------- Tables ----------
    comparison = pd.DataFrame(
        {
            "Metric": ["Revenue ($/ac)", "Expenses ($/ac)", "Profit ($/ac)"],
            "Breakeven Budget": [revenue_per_acre, fixed_inputs, revenue_per_acre - fixed_inputs],
            "Variable Rate":    [revenue_per_acre, expenses_var,  profit_var],
            "Fixed Rate":       [revenue_per_acre, expenses_fix,  profit_fix],
        }
    )

    cornsoy = pd.DataFrame(
        {
            "Crop": ["Corn", "Soybeans"],
            "Yield (bu/ac)":      [corn_yield, bean_yield],
            "Sell Price ($/bu)":  [corn_price, bean_price],
        }
    )
    cornsoy["Revenue ($/ac)"]      = cornsoy["Yield (bu/ac)"] * cornsoy["Sell Price ($/bu)"]
    cornsoy["Fixed Inputs ($/ac)"] = base_exp
    cornsoy["Profit ($/ac)"]       = cornsoy["Revenue ($/ac)"] - cornsoy["Fixed Inputs ($/ac)"]

    fixed_df = pd.DataFrame(list(expenses.items()), columns=["Expense", "$/ac"])
    if not fixed_df.empty:
        total_row = pd.DataFrame([{"Expense": "Total Fixed Costs", "$/ac": fixed_df["$/ac"].sum()}])
        fixed_df = pd.concat([fixed_df, total_row], ignore_index=True)

    # ---------- Layout ----------
    left, right = st.columns([2, 1], gap="large")

    with left:
        render_static_table(comparison, "Profit Comparison")

        with st.expander("Show Calculation Formulas", expanded=False):
            st.markdown("""
            <div style="display:flex;flex-wrap:wrap;gap:6px;
                        font-size:0.75rem;line-height:1.1rem;">
              <div style="flex:1;min-width:180px;border:1px solid #444;
                          border-radius:6px;padding:5px;background-color:#111;">
                <b>Breakeven Budget</b><br>
                (Target Yield × Sell Price) − Fixed Inputs
              </div>
              <div style="flex:1;min-width:180px;border:1px solid #444;
                          border-radius:6px;padding:5px;background-color:#111;">
                <b>Variable Rate</b><br>
                (Target Yield × Sell Price) − (Fixed Inputs + Var Seed + Var Fert)
              </div>
              <div style="flex:1;min-width:180px;border:1px solid #444;
                          border-radius:6px;padding:5px;background-color:#111;">
                <b>Fixed Rate</b><br>
                (Target Yield × Sell Price) − (Fixed Inputs + Fixed Seed + Fixed Fert)
              </div>
            </div>
            """, unsafe_allow_html=True)

        render_static_table(cornsoy, "Corn vs Soybean Profitability")

    with right:
        render_static_table(fixed_df, "Fixed Input Costs")


# ---------- render ----------
render_profit_summary()

