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
# UI: Uploaders row + summaries  (Yield patched with debug + robust coords)
# =========================================================
def render_uploaders():
    st.subheader("Upload Maps")
    u1, u2, u3, u4 = st.columns(4)

    # --- Zones (same as before) ---
    with u1:
        st.caption("Zone Map · GeoJSON/JSON/ZIP(SHP)")
        zone_file = st.file_uploader("Zone", type=["geojson", "json", "zip"],
                                     key="up_zone", accept_multiple_files=False)
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
# --- Yield (FINAL — Bulletproof, No Dummy Points) ---
with u2:
    st.caption("Yield Map(s) · CSV/GeoJSON/JSON/ZIP(SHP)")
    yield_files = st.file_uploader("Yield", type=["csv","geojson","json","zip"],
                                   key="up_yield", accept_multiple_files=True)
    st.session_state["yield_df"] = pd.DataFrame()

    if yield_files:
        frames = []
        yield_source_info = []

        for yf in yield_files:
            try:
                name = yf.name.lower()

                # === CSV path ===
                if name.endswith(".csv"):
                    df = pd.read_csv(yf)
                    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]

                    lat_col = find_col(df, ["latitude","lat","y","point_y","northing"])
                    lon_col = find_col(df, ["longitude","lon","x","point_x","easting"])
                    ycol = find_col(df, [
                        "yield","yld_vol_dr","yld_mass_dr","yield_dry","dry_yield","wet_yield",
                        "yld_vol_wt","yld_mass_wt","crop_flw_m"
                    ])

                    if lat_col and lon_col:
                        df.rename(columns={lat_col:"Latitude", lon_col:"Longitude"}, inplace=True)
                    else:
                        st.warning(f"{yf.name}: no coordinate columns detected — skipped.")
                        continue

                    if not ycol:
                        st.warning(f"{yf.name}: no yield column found — skipped.")
                        continue
                    if ycol != "yield":
                        df.rename(columns={ycol:"Yield"}, inplace=True)

                    df = df[["Latitude","Longitude","Yield"]].dropna(subset=["Latitude","Longitude","Yield"])
                    if df.empty:
                        st.warning(f"{yf.name}: all rows invalid — skipped.")
                        continue

                    frames.append(df)
                    yield_source_info.append(f"{yf.name} — using field '{ycol}'")

                # === GeoJSON / SHP / ZIP ===
                else:
                    gdf = load_vector_file(yf)
                    if gdf is None or gdf.empty:
                        st.warning(f"{yf.name}: empty or unreadable geometry.")
                        continue

                    if "geometry" in gdf.columns or hasattr(gdf, "geometry"):
                        try:
                            if gdf.geom_type.astype(str).str.contains("Point", case=False).any():
                                gdf["Longitude"] = gdf.geometry.x
                                gdf["Latitude"] = gdf.geometry.y
                            else:
                                reps = gdf.geometry.representative_point()
                                gdf["Longitude"] = reps.x
                                gdf["Latitude"] = reps.y
                        except Exception as e:
                            st.warning(f"{yf.name}: geometry coord extraction failed ({e})")
                            continue

                    ycol = find_col(gdf, [
                        "yield","yld_vol_dr","yld_mass_dr","yield_dry","dry_yield","wet_yield",
                        "yld_vol_wt","yld_mass_wt","crop_flw_m"
                    ])
                    if not ycol:
                        st.warning(f"{yf.name}: no yield column found — skipped.")
                        continue
                    if ycol != "yield":
                        gdf.rename(columns={ycol:"Yield"}, inplace=True)

                    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
                    if "Latitude" not in df.columns or "Longitude" not in df.columns:
                        st.warning(f"{yf.name}: no coordinates after extraction — skipped.")
                        continue

                    df = df[["Latitude","Longitude","Yield"]].dropna(subset=["Latitude","Longitude","Yield"])
                    if df.empty:
                        st.warning(f"{yf.name}: all rows invalid — skipped.")
                        continue

                    frames.append(df)
                    yield_source_info.append(f"{yf.name} — using field '{ycol}'")

            except Exception as e:
                st.warning(f"Yield file {yf.name}: {e}")

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            st.session_state["yield_df"] = combined
            st.success(f"✅ Yield data loaded successfully.\n" + "\n".join(yield_source_info))
        else:
            st.error("❌ No valid yield data found — nothing to map.")
    else:
        st.caption("No yield files uploaded.")

    
    # --- Fertilizer (Hardened) ---
    with u3:
        st.caption("Fertilizer RX · CSV/GeoJSON/JSON/ZIP(SHP)")
        fert_files = st.file_uploader("Fert", type=["csv", "geojson", "json", "zip"],
                                      key="up_fert", accept_multiple_files=True)
        st.session_state["fert_layers_store"] = {}
        st.session_state["fert_gdfs"] = {}

        if fert_files:
            summ = []
            for f in fert_files:
                try:
                    grouped, gdf_orig = process_prescription(f, "fertilizer")

                    # Geometry hardening + centroids for overlays
                    if gdf_orig is not None and not gdf_orig.empty:
                        gdf_orig = gdf_orig.copy()
                        try:
                            gdf_orig["geometry"] = gdf_orig.geometry.buffer(0)
                        except Exception:
                            pass
                        try:
                            gdf_orig = gdf_orig.explode(index_parts=False, ignore_index=True)
                        except TypeError:
                            gdf_orig = gdf_orig.explode().reset_index(drop=True)
                        reps = gdf_orig.geometry.representative_point()
                        gdf_orig["Longitude"] = reps.x
                        gdf_orig["Latitude"]  = reps.y

                    if not grouped.empty:
                        key = os.path.splitext(f.name)[0].lower().replace(" ", "_")
                        st.session_state["fert_layers_store"][key] = grouped
                        if gdf_orig is not None and not gdf_orig.empty:
                            st.session_state["fert_gdfs"][key] = gdf_orig
                        summ.append({"File": f.name, "Products": len(grouped)})

                except Exception as e:
                    st.warning(f"Fertilizer file {f.name}: {e}")

            if summ:
                st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True,
                             height=df_px_height(len(summ)))
            else:
                st.error("No valid fertilizer RX maps detected — check geometry or attributes.")
        else:
            st.caption("No fertilizer files uploaded.")

    # --- Seed (Hardened) ---
    with u4:
        st.caption("Seed RX · CSV/GeoJSON/JSON/ZIP(SHP)")
        seed_files = st.file_uploader("Seed", type=["csv", "geojson", "json", "zip"],
                                      key="up_seed", accept_multiple_files=True)
        st.session_state["seed_layers_store"] = {}
        st.session_state["seed_gdf"] = None

        if seed_files:
            summ = []
            last_seed_gdf = None
            for f in seed_files:
                try:
                    grouped, gdf_orig = process_prescription(f, "seed")

                    # Geometry hardening + centroids for overlays
                    if gdf_orig is not None and not gdf_orig.empty:
                        gdf_orig = gdf_orig.copy()
                        try:
                            gdf_orig["geometry"] = gdf_orig.geometry.buffer(0)
                        except Exception:
                            pass
                        try:
                            gdf_orig = gdf_orig.explode(index_parts=False, ignore_index=True)
                        except TypeError:
                            gdf_orig = gdf_orig.explode().reset_index(drop=True)
                        reps = gdf_orig.geometry.representative_point()
                        gdf_orig["Longitude"] = reps.x
                        gdf_orig["Latitude"]  = reps.y
                        last_seed_gdf = gdf_orig

                    if not grouped.empty:
                        key = os.path.splitext(f.name)[0].lower().replace(" ", "_")
                        st.session_state["seed_layers_store"][key] = grouped
                        summ.append({"File": f.name, "Products": len(grouped)})

                except Exception as e:
                    st.warning(f"Seed file {f.name}: {e}")

            if last_seed_gdf is not None and not last_seed_gdf.empty:
                st.session_state["seed_gdf"] = last_seed_gdf

            if summ:
                st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True,
                             height=df_px_height(len(summ)))
            else:
                st.error("No valid seed RX maps detected — check geometry or attributes.")
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
        if df is None or df.empty:
            return None, None

        # --- coordinate detection or synthesize if missing ---
        latc = find_col(df, ["latitude"]) or "Latitude"
        lonc = find_col(df, ["longitude"]) or "Longitude"
        if latc not in df.columns or lonc not in df.columns:
            # fallback to approximate center of USA or zones
            zg = st.session_state.get("zones_gdf")
            if zg is not None and not getattr(zg, "empty", True):
                tb = zg.total_bounds
                df["Latitude"] = [(tb[1]+tb[3])/2.0]
                df["Longitude"] = [(tb[0]+tb[2])/2.0]
            else:
                df["Latitude"] = [39.5]
                df["Longitude"] = [-98.35]
            latc, lonc = "Latitude", "Longitude"

        # --- sanitize coordinates and values ---
        df = df.copy()
        df[latc] = pd.to_numeric(df[latc], errors="coerce")
        df[lonc] = pd.to_numeric(df[lonc], errors="coerce")
        df[values.name] = pd.to_numeric(df[values.name], errors="coerce")
        df.dropna(subset=[latc, lonc, values.name], inplace=True)

        if df.empty:
            # final fallback single dummy point
            df = pd.DataFrame({latc:[39.5], lonc:[-98.35], values.name:[values.mean() if hasattr(values,'mean') else 0.0]})

        south, west, north, east = bounds
        vmin, vmax = float(df[values.name].min()), float(df[values.name].max())
        if vmin == vmax: vmax = vmin + 1.0

        # --- if <3 points, draw discrete points ---
        if len(df) < 3:
            for _, r in df.iterrows():
                val = r[values.name]
                folium.CircleMarker(
                    location=[r[latc], r[lonc]],
                    radius=5,
                    color="#ffffff80",
                    fill=True,
                    fill_color=mpl_colors.rgb2hex(cmap((val - vmin)/(vmax - vmin + 1e-9))[:3]),
                    fill_opacity=0.9,
                    popup=f"{name}: {val:.1f}"
                ).add_to(m)
            return vmin, vmax

        # --- normal dense heatmap path ---
        pts_lon = df[lonc].astype(float).values
        pts_lat = df[latc].astype(float).values
        vals_ok = df[values.name].astype(float).values
        n = 200
        lon_lin = np.linspace(west, east, n)
        lat_lin = np.linspace(south, north, n)
        lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)
        grid_lin = griddata((pts_lon, pts_lat), vals_ok, (lon_grid, lat_grid), method="linear")
        grid_nn = griddata((pts_lon, pts_lat), vals_ok, (lon_grid, lat_grid), method="nearest")
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

# ---------- Heatmap overlays + stacked legends ----------
ymin, ymax = add_heatmap_overlay(m, df_for_maps, df_for_maps["Yield"], "Yield (bu/ac)", plt.cm.RdYlGn, False, bounds)
if ymin is not None:
    add_gradient_legend_pos(m, "Yield (bu/ac)", ymin, ymax, plt.cm.RdYlGn, corner="tl")

vmin, vmax = add_heatmap_overlay(m, df_for_maps, df_for_maps["NetProfit_Variable"],
                                 "Variable Rate Profit ($/ac)", plt.cm.RdYlGn, True, bounds)
if vmin is not None:
    add_gradient_legend_pos(m, "Variable Rate Profit ($/ac)", vmin, vmax, plt.cm.RdYlGn, corner="tl")

fmin, fmax = add_heatmap_overlay(m, df_for_maps, df_for_maps["NetProfit_Fixed"],
                                 "Fixed Rate Profit ($/ac)", plt.cm.RdYlGn, False, bounds)
if fmin is not None:
    add_gradient_legend_pos(m, "Fixed Rate Profit ($/ac)", fmin, fmax, plt.cm.RdYlGn, corner="tl")

# ---------- FORCE-ON-TOP zone outlines + FINAL GLOBAL fit_bounds ----------
try:
    if zones_gdf is not None and not getattr(zones_gdf, "empty", True):
        folium.GeoJson(
            zones_gdf,
            name="Zone Outlines (Top)",
            style_function=lambda feature: {
                "fillOpacity": 0,
                "color": "#000000",
                "weight": 3,
                "opacity": 1.0,
            },
            tooltip=None
        ).add_to(m)

    south, west, north, east = compute_bounds_for_heatmaps()
    if zones_gdf is not None and not getattr(zones_gdf, "empty", True):
        zb = zones_gdf.total_bounds
        south = min(south, zb[1]); west = min(west, zb[0])
        north = max(north, zb[3]);  east = max(east, zb[2])

    m.fit_bounds([[south, west], [north, east]])
except Exception as e:
    st.warning(f"Auto-zoom/top-outline fallback failed: {e}")

# NOTE: removed LayerControl to prevent selectors

st_folium(m, use_container_width=True, height=600)


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
