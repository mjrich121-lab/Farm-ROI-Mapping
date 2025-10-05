# =========================================================
# Farm Profit Mapping Tool V4  — BULLETPROOF & ULTRA-COMPACT
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
import math
from branca.element import MacroElement, Template
from matplotlib import colors as mpl_colors

# ---------------------------------------------------------
# Page + Title
# ---------------------------------------------------------
st.set_page_config(page_title="Farm ROI Tool V4", layout="wide")
st.title("Farm Profit Mapping Tool V4")

# ---------------------------------------------------------
# Compact CSS (tight gutters, short widgets, no dead space)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Tight gutters + tighter vertical rhythm */
    div[data-testid="column"]{padding-left:.15rem !important;padding-right:.15rem !important;}
    section[data-testid="stVerticalBlock"]>div{padding-top:.15rem !important;padding-bottom:.15rem !important;}

    /* Headers tiny */
    h2,h3{margin:.25rem 0 .15rem 0 !important;font-size:1rem !important;}
    .block-container{padding-top:1rem !important;}

    /* Expander compact */
    div[data-testid="stExpander"] details summary{padding:.2rem .4rem !important;font-size:.85rem !important;}
    div[data-testid="stExpander"] details>div{padding:.25rem .4rem !important;}

    /* Number inputs compact + narrow */
    div[data-testid="stNumberInput"] label{font-size:.72rem !important;margin-bottom:.05rem !important;}
    div[data-testid="stNumberInput"] div[role="spinbutton"]{
        min-height:22px !important;height:22px !important;padding:0 4px !important;font-size:.8rem !important;}
    div[data-testid="stNumberInput"] button{padding:0 !important;min-width:16px !important;}

    /* For the one-line 12 fixed inputs row only: make boxes 96px */
    .fi-row div[data-testid="stNumberInput"]{width:96px !important;max-width:96px !important;}

    /* DataFrames & DataEditors no wasted padding */
    div[data-testid="stDataFrame"] table, div[data-testid="stDataEditor"] table{font-size:.8rem !important;}
    div[data-testid="stDataFrame"] th, div[data-testid="stDataFrame"] td,
    div[data-testid="stDataEditor"] th, div[data-testid="stDataEditor"] td{
        padding:1px 4px !important;line-height:1.05rem !important;}

    /* Captions tiny */
    div[data-testid="stCaptionContainer"]{margin:.1rem 0 !important;font-size:.75rem !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Safe helpers
# ---------------------------------------------------------
def _to_float(x, default=0.0):
    try:
        if x is None: return float(default)
        if isinstance(x, (int, float)) and math.isfinite(x): return float(x)
        return float(str(x).strip())
    except Exception:
        return float(default)

def df_px_height(nrows: int, row_h: int = 26, header: int = 30, pad: int = 2) -> int:
    """Exact pixel height so tables/editors render with NO internal scroll."""
    return int(header + nrows * row_h + pad)

def _ss_init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

# Compact number input with unique key namespace (prevents collisions)
def _mini_num(label: str, key: str, default: float = 0.0, step: float = 1.0, ns="fi__"):
    st.caption(label)
    safe_key = f"{ns}{key}"
    init = _to_float(st.session_state.get(safe_key, st.session_state.get(key, default)), default)
    return st.number_input(label, min_value=0.0, value=init, step=step,
                           label_visibility="collapsed", key=safe_key)

# ---------------------------------------------------------
# File loaders (bulletproof)
# ---------------------------------------------------------
def load_vector_file(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith((".geojson", ".json")):
            gdf = gpd.read_file(uploaded_file)
        elif uploaded_file.name.lower().endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zpath = os.path.join(tmpdir, "in.zip")
                with open(zpath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(tmpdir)
                shp_path = None
                for fn in os.listdir(tmpdir):
                    if fn.lower().endswith(".shp"):
                        shp_path = os.path.join(tmpdir, fn); break
                if not shp_path: return None
                gdf = gpd.read_file(shp_path)
        elif uploaded_file.name.lower().endswith(".shp"):
            with tempfile.TemporaryDirectory() as tmpdir:
                shp_path = os.path.join(tmpdir, uploaded_file.name)
                with open(shp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                gdf = gpd.read_file(shp_path)
        else:
            return None

        if gdf is None or gdf.empty: return None
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
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
            gdf.columns = [c.strip().lower().replace(" ","_") for c in gdf.columns]
            if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
            else: gdf = gdf.to_crs(epsg=4326)
            gdf["Longitude"] = gdf.geometry.representative_point().x
            gdf["Latitude"]  = gdf.geometry.representative_point().y
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        else:
            df = pd.read_csv(file)
            df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
    except Exception as e:
        st.warning(f"⚠️ Failed to read {file.name}: {e}")
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])

    if "product" not in df.columns:
        for c in ["variety","hybrid","type","name","material"]:
            if c in df.columns:
                df.rename(columns={c:"product"}, inplace=True); break
        else:
            df["product"] = prescrip_type.capitalize()
    if "acres" not in df.columns: df["acres"] = 0.0

    # Compact options
    with st.expander(f"⚙️ {prescrip_type.capitalize()} Map Options — {file.name}", expanded=False):
        override = st.number_input(
            f"{prescrip_type}_override_{file.name}", min_value=0.0, value=0.0, step=0.1,
            label_visibility="visible", help="Override acres per polygon (optional)"
        )
        if override > 0: df["acres"] = override

    if "costtotal" not in df.columns:
        if {"price_per_unit","units"}.issubset(df.columns):
            df["costtotal"] = df["price_per_unit"] * df["units"]
        elif {"rate","price"}.issubset(df.columns):
            df["costtotal"] = df["rate"] * df["price"]
        else:
            df["costtotal"] = 0.0

    if df.empty:
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])

    grouped = df.groupby("product", as_index=False).agg(
        Acres=("acres","sum"), CostTotal=("costtotal","sum")
    )
    grouped["CostPerAcre"] = grouped.apply(lambda r: r["CostTotal"]/r["Acres"] if r["Acres"]>0 else 0.0, axis=1)
    return grouped

# =========================================================
# 2–3. FILE UPLOADS (Two rows, four tiles) — compact
# =========================================================
st.header("Upload Maps")
c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

# ---------- ZONES ----------
with c1:
    st.subheader("Zone Map")
    zone_file = st.file_uploader("Upload Zone Map", type=["geojson","json","zip","shp"], key="zone_file",
                                 accept_multiple_files=False, label_visibility="collapsed")
    st.caption("GeoJSON / JSON / zipped SHP")
    if zone_file:
        gdf = load_vector_file(zone_file)
        if gdf is not None and not gdf.empty:
            st.caption(f"✅ Loaded {len(gdf)} polygons")

            # zone column
            zone_col = None
            for cand in ["Zone","zone","ZONE","Name","name"]:
                if cand in gdf.columns: zone_col = cand; break
            if zone_col is None:
                gdf["ZoneIndex"] = range(1, len(gdf)+1)
                zone_col = "ZoneIndex"
            gdf["Zone"] = gdf[zone_col]

            # acres (equal-area)
            _ea = gdf.copy()
            if _ea.crs is None: _ea.set_crs(epsg=4326, inplace=True)
            if _ea.crs.is_geographic: _ea = _ea.to_crs(epsg=5070)
            gdf["Calculated Acres"] = (_ea.geometry.area * 0.000247105).astype(float)
            gdf["Override Acres"]   = gdf["Calculated Acres"].astype(float)

            disp = gdf[["Zone","Calculated Acres","Override Acres"]].copy()
            edited = st.data_editor(
                disp, num_rows="fixed", hide_index=True, use_container_width=True,
                column_config={
                    "Zone": st.column_config.TextColumn(disabled=True),
                    "Calculated Acres": st.column_config.NumberColumn(format="%.2f", disabled=True),
                    "Override Acres": st.column_config.NumberColumn(format="%.2f"),
                },
                key="zone_acres_editor",
                height=df_px_height(len(disp), row_h=24, header=28, pad=2)
            )
            edited["Override Acres"] = pd.to_numeric(edited["Override Acres"], errors="coerce") \
                                        .fillna(edited["Calculated Acres"])
            gdf["Override Acres"] = edited["Override Acres"].astype(float).values
            st.caption(f"Total Acres — Calc: {gdf['Calculated Acres'].sum():,.2f} | Override: {gdf['Override Acres'].sum():,.2f}")
            st.session_state["zones_gdf"] = gdf
        else:
            st.error("❌ Could not load zone map.")
    else:
        st.caption("No zone file")

# ---------- YIELD ----------
with c2:
    st.subheader("Yield Map(s)")
    yield_files = st.file_uploader("Upload Yield Map(s)", type=["csv","geojson","json","zip","shp"],
                                   key="yield", accept_multiple_files=True, label_visibility="collapsed")
    st.caption("CSV / GeoJSON / JSON / zipped SHP")
    st.session_state.setdefault("yield_df", pd.DataFrame())
    if yield_files:
        frames = []
        total_rows = 0
        for yf in yield_files:
            try:
                if yf.name.lower().endswith(".csv"):
                    df = pd.read_csv(yf)
                else:
                    yg = load_vector_file(yf)
                    df = pd.DataFrame(yg.drop(columns="geometry", errors="ignore")) if yg is not None else pd.DataFrame()
                if not df.empty:
                    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
                    ycols = [c for c in df.columns if any(k in c for k in
                             ["yld_vol_dr","yld_mass_dr","yield_dry","dry_yield","yield","wet_yield"])]
                    if ycols: df.rename(columns={ycols[0]:"Yield"}, inplace=True)
                    else: df["Yield"] = 0.0
                    frames.append(df); total_rows += len(df)
            except Exception as e:
                st.warning(f"⚠️ {yf.name}: {e}")
        st.session_state["yield_df"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        st.caption(f"Loaded: {len(yield_files)} file(s), rows={total_rows:,}")
    else:
        st.caption("No yield files")

# ---------- FERT ----------
with c3:
    st.subheader("Fertilizer RX")
    fert_files = st.file_uploader("Upload Fert Map(s)", type=["csv","geojson","json","zip","shp"],
                                  key="fert", accept_multiple_files=True, label_visibility="collapsed")
    st.caption("CSV / GeoJSON / JSON / zipped SHP")
    st.session_state["fert_layers_store"] = {}
    if fert_files:
        summ = []
        for f in fert_files:
            grouped = process_prescription(f, "fertilizer")
            if not grouped.empty:
                k = os.path.splitext(f.name)[0].lower().replace(" ","_")
                st.session_state["fert_layers_store"][k] = grouped
                summ.append({"File": f.name, "Products": len(grouped)})
        if summ:
            st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True,
                         height=df_px_height(len(summ)))
    else:
        st.caption("No fert files")

# ---------- SEED ----------
with c4:
    st.subheader("Seed RX")
    seed_files = st.file_uploader("Upload Seed Map(s)", type=["csv","geojson","json","zip","shp"],
                                  key="seed", accept_multiple_files=True, label_visibility="collapsed")
    st.caption("CSV / GeoJSON / JSON / zipped SHP")
    st.session_state["seed_layers_store"] = {}
    if seed_files:
        summ = []
        for f in seed_files:
            grouped = process_prescription(f, "seed")
            if not grouped.empty:
                k = os.path.splitext(f.name)[0].lower().replace(" ","_")
                st.session_state["seed_layers_store"][k] = grouped
                summ.append({"File": f.name, "Products": len(grouped)})
        if summ:
            st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True,
                         height=df_px_height(len(summ)))
    else:
        st.caption("No seed files")

# =========================================================
# 4. COMPACT CONTROLS ABOVE MAP
# =========================================================

# ---------- Fixed Inputs ($/ac) — 12 on ONE row ----------
st.caption("Fixed Inputs ($/ac)")
fi_cols = st.columns(12, gap="small")
with st.container():
    with st.container():
        with st.container():
            pass
# put a scoping class on the row for CSS width control
st.write('<div class="fi-row"></div>', unsafe_allow_html=True)
with fi_cols[0]:  chemicals      = _mini_num("Chem ($/ac)",        "chem",   0.0, 1.0)
with fi_cols[1]:  insurance      = _mini_num("Insur ($/ac)",       "ins",    0.0, 1.0)
with fi_cols[2]:  insecticide    = _mini_num("Insect/Fung ($/ac)", "insect", 0.0, 1.0)
with fi_cols[3]:  fertilizer     = _mini_num("Fert Flat ($/ac)",   "fert",   0.0, 1.0)
with fi_cols[4]:  seed           = _mini_num("Seed Flat ($/ac)",   "seed",   0.0, 1.0)
with fi_cols[5]:  cash_rent      = _mini_num("Cash Rent ($/ac)",   "rent",   0.0, 1.0)
with fi_cols[6]:  machinery      = _mini_num("Mach ($/ac)",        "mach",   0.0, 1.0)
with fi_cols[7]:  labor          = _mini_num("Labor ($/ac)",       "labor",  0.0, 1.0)
with fi_cols[8]:  coliving       = _mini_num("Living ($/ac)",      "col",    0.0, 1.0)
with fi_cols[9]:  extra_fuel     = _mini_num("Fuel ($/ac)",        "fuel",   0.0, 1.0)
with fi_cols[10]: extra_interest = _mini_num("Interest ($/ac)",    "int",    0.0, 1.0)
with fi_cols[11]: truck_fuel     = _mini_num("Truck Fuel ($/ac)",  "truck",  0.0, 1.0)

expenses = {
    "Chemicals": chemicals, "Insurance": insurance, "Insecticide/Fungicide": insecticide,
    "Fertilizer (Flat)": fertilizer, "Seed (Flat)": seed, "Cash Rent": cash_rent,
    "Machinery": machinery, "Labor": labor, "Cost of Living": coliving,
    "Extra Fuel": extra_fuel, "Extra Interest": extra_interest, "Truck Fuel": truck_fuel
}
base_expenses_per_acre = float(sum(expenses.values()))

# ---------- Corn/Soy assumptions in ONE compact row ----------
ass_cols = st.columns(4, gap="small")
with ass_cols[0]:
    st.caption("Corn Yield (bu/ac)")
    _ss_init("corn_yield", 200.0)
    st.session_state["corn_yield"] = st.number_input("corn_yld", min_value=0.0,
                                                     value=_to_float(st.session_state["corn_yield"],200.0),
                                                     step=1.0, label_visibility="collapsed")
with ass_cols[1]:
    st.caption("Corn Price ($/bu)")
    _ss_init("corn_price", 5.0)
    st.session_state["corn_price"] = st.number_input("corn_px", min_value=0.0,
                                                     value=_to_float(st.session_state["corn_price"],5.0),
                                                     step=0.05, label_visibility="collapsed")
with ass_cols[2]:
    st.caption("Soy Yield (bu/ac)")
    _ss_init("bean_yield", 60.0)
    st.session_state["bean_yield"] = st.number_input("bean_yld", min_value=0.0,
                                                     value=_to_float(st.session_state["bean_yield"],60.0),
                                                     step=1.0, label_visibility="collapsed")
with ass_cols[3]:
    st.caption("Soy Price ($/bu)")
    _ss_init("bean_price", 12.0)
    st.session_state["bean_price"] = st.number_input("bean_px", min_value=0.0,
                                                     value=_to_float(st.session_state["bean_price"],12.0),
                                                     step=0.05, label_visibility="collapsed")

# Tiny preview (2 rows, no scroll)
preview_df = pd.DataFrame({
    "Crop": ["Corn","Soybeans"],
    "Yield": [st.session_state["corn_yield"], st.session_state["bean_yield"]],
    "Price": [st.session_state["corn_price"], st.session_state["bean_price"]],
    "Revenue": [
        st.session_state["corn_yield"]*st.session_state["corn_price"],
        st.session_state["bean_yield"]*st.session_state["bean_price"]
    ],
    "Fixed": [base_expenses_per_acre, base_expenses_per_acre]
})
preview_df["Breakeven"] = preview_df["Revenue"] - preview_df["Fixed"]
st.dataframe(
    preview_df.style.format({
        "Yield":"{:.0f}", "Price":"${:.2f}",
        "Revenue":"${:,.0f}","Fixed":"${:,.0f}","Breakeven":"${:,.0f}"
    }),
    use_container_width=True, hide_index=True,
    height=df_px_height(2, row_h=24, header=26, pad=2)
)

# Optional: fixed/variable product tables (kept compact)
fx_col, vr_col = st.columns(2, gap="small")
with fx_col:
    with st.expander("Fixed Rate Inputs", expanded=False):
        if "fixed_products" not in st.session_state or st.session_state["fixed_products"].empty:
            st.session_state["fixed_products"] = pd.DataFrame(
                {"Type":["Seed","Fertilizer"], "Product":["",""], "Rate":[0.0,0.0],
                 "CostPerUnit":[0.0,0.0], "$/ac":[0.0,0.0]}
            )
        fx = st.data_editor(st.session_state["fixed_products"], num_rows="dynamic",
                            use_container_width=True, key="fixed_editor")
        st.session_state["fixed_products"] = fx.copy().reset_index(drop=True)

with vr_col:
    with st.expander("Variable Rate Inputs", expanded=False):
        fert_df = st.session_state.get("fert_products", pd.DataFrame())
        seed_df = st.session_state.get("seed_products", pd.DataFrame())
        if not fert_df.empty:
            st.caption("Fertilizer (VR)")
            st.dataframe(fert_df, use_container_width=True, hide_index=True,
                         height=df_px_height(len(fert_df)))
        if not seed_df.empty:
            st.caption("Seed (VR)")
            st.dataframe(seed_df, use_container_width=True, hide_index=True,
                         height=df_px_height(len(seed_df)))
        if fert_df.empty and seed_df.empty:
            st.caption("— No VR inputs —")

# =========================================================
# 5. BASE MAP (bulletproof)
# =========================================================
def make_base_map():
    try:
        m = folium.Map(location=[39.5, -98.35], zoom_start=5, min_zoom=2, tiles=None,
                       scrollWheelZoom=False, prefer_canvas=True)
        try:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", overlay=False, control=True, name="Esri Imagery"
            ).add_to(m)
        except Exception: pass
        try:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", overlay=True, control=True, name="Esri Boundaries"
            ).add_to(m)
        except Exception: pass

        # Enable scrollwheel only on click
        try:
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
        except Exception: pass

        try: folium.LayerControl(collapsed=False, position="topright").add_to(m)
        except Exception: pass
        return m
    except Exception:
        return folium.Map(location=[39.5, -98.35], zoom_start=4)

m = make_base_map()
st.session_state["layer_control_added"] = False

# =========================================================
# 6. ZONES OVERLAY (safe)
# =========================================================
def add_zones_overlay(m):
    zones_gdf = st.session_state.get("zones_gdf")
    if zones_gdf is None or zones_gdf.empty: return m
    try:
        zones_gdf = zones_gdf.to_crs(epsg=4326)
        if "Zone" not in zones_gdf.columns:
            zones_gdf["Zone"] = range(1, len(zones_gdf)+1)

        zb = zones_gdf.total_bounds
        m.location = [(zb[1]+zb[3])/2, (zb[0]+zb[2])/2]; m.zoom_start = 15

        palette = ["#FF0000","#FF8C00","#FFFF00","#32CD32","#006400",
                   "#1E90FF","#8A2BE2","#FFC0CB","#A52A2A","#00CED1"]
        uniq = list(dict.fromkeys(sorted(list(zones_gdf["Zone"].astype(str).unique()))))
        color_map = {z: palette[i % len(palette)] for i,z in enumerate(uniq)}

        folium.GeoJson(
            zones_gdf,
            name="Zones",
            style_function=lambda f: {"fillColor": color_map.get(str(f["properties"].get("Zone","")),"#808080"),
                                      "color":"black","weight":1,"fillOpacity":0.08},
            tooltip=folium.GeoJsonTooltip(fields=[c for c in ["Zone","Calculated Acres","Override Acres"]
                                                  if c in zones_gdf.columns])
        ).add_to(m)

        items = "".join([f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                         f"<div style='background:{color_map[z]};width:12px;height:12px;margin-right:6px;'></div>{z}</div>"
                         for z in uniq])
        legend = f"""
        <div style="position:absolute; bottom:14px; right:14px; z-index:9999;
                    font-family:sans-serif; font-size:12px; color:white;
                    background:rgba(0,0,0,.65); padding:6px 10px; border-radius:5px; width:150px;">
          <div style="font-weight:600;margin-bottom:4px;cursor:pointer;"
               onclick="var x=document.getElementById('zl'); x.style.display=(x.style.display==='none'?'block':'none');">
            Zone Colors ▼
          </div>
          <div id="zl" style="display:block;">{items}</div>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend))
    except Exception as e:
        st.warning(f"⚠️ Zones overlay skipped: {e}")
    return m

add_zones_overlay(m)

# =========================================================
# 7A. Bounds helper
# =========================================================
def compute_bounds_for_heatmaps():
    try:
        bnds = []
        for key in ["zones_gdf"]:
            g = st.session_state.get(key)
            if g is not None and not g.empty:
                tb = g.total_bounds
                if tb is not None and len(tb)==4 and not any(pd.isna(tb)):
                    bnds.append([[tb[1], tb[0]], [tb[3], tb[2]]])
        ydf = st.session_state.get("yield_df")
        if ydf is not None and not ydf.empty and {"latitude","longitude"}.issubset({c.lower() for c in ydf.columns}):
            y2 = ydf.rename(columns={c:c.lower() for c in ydf.columns})
            bnds.append([[y2["latitude"].min(), y2["longitude"].min()],
                         [y2["latitude"].max(), y2["longitude"].max()]])
        if bnds:
            south=min(b[0][0] for b in bnds); west=min(b[0][1] for b in bnds)
            north=max(b[1][0] for b in bnds); east=max(b[1][1] for b in bnds)
            return south, west, north, east
    except Exception: pass
    return 25.0, -125.0, 49.0, -66.0

def safe_fit_bounds(m, bounds):
    try: m.fit_bounds([[bounds[0],bounds[1]],[bounds[2],bounds[3]]])
    except Exception: pass

# =========================================================
# 7B. Legends + RX overlays (optional hook)
# =========================================================
def add_gradient_legend(name, vmin, vmax, cmap, index):
    top_offset = 20 + index*72
    stops = [f"{mpl_colors.rgb2hex(cmap(i/100.0)[:3])} {i}%" for i in range(0,101,10)]
    gradient_css = ", ".join(stops)
    html = f"""
    <div style="position:absolute;top:{top_offset}px;left:10px;z-index:9999;
                font-family:sans-serif;font-size:12px;color:white;
                background:rgba(0,0,0,.65);padding:6px 10px;border-radius:5px;width:180px;">
      <div style="font-weight:600;margin-bottom:4px;">{name}</div>
      <div style="height:12px;background:linear-gradient(90deg,{gradient_css});border-radius:2px;margin-bottom:4px;"></div>
      <div style="display:flex;justify-content:space-between;">
        <span>{vmin:.1f}</span><span>{vmax:.1f}</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))

# =========================================================
# 7C. Yield + Profit heatmaps (crash-proof)
# =========================================================
def add_heatmap_overlay(df, values, name, cmap, show_default, bounds):
    try:
        if df is None or df.empty: return None, None
        south, west, north, east = bounds
        vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        if vals.empty: return None, None
        mask = (df[["Latitude","Longitude"]].applymap(np.isfinite).all(axis=1)) & vals.notna()
        if mask.sum() < 3: return None, None

        pts_lon = df.loc[mask,"Longitude"].astype(float).values
        pts_lat = df.loc[mask,"Latitude"].astype(float).values
        vals_ok = vals.loc[mask].astype(float).values

        n = 200
        lon_lin = np.linspace(west, east, n); lat_lin = np.linspace(south, north, n)
        lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)

        grid_lin = griddata((pts_lon,pts_lat), vals_ok, (lon_grid,lat_grid), method="linear")
        grid_nn  = griddata((pts_lon,pts_lat), vals_ok, (lon_grid,lat_grid), method="nearest")
        grid = np.where(np.isnan(grid_lin), grid_nn, grid_lin)
        if grid is None or np.all(np.isnan(grid)): return None, None

        vmin = float(np.nanpercentile(vals_ok, 5)) if len(vals_ok)>0 else 0.0
        vmax = float(np.nanpercentile(vals_ok,95)) if len(vals_ok)>0 else 1.0
        if vmin==vmax: vmax = vmin + 1.0

        rgba = cmap((grid - vmin) / (vmax - vmin)); rgba = np.flipud(rgba); rgba = (rgba*255).astype(np.uint8)

        folium.raster_layers.ImageOverlay(
            image=rgba, bounds=[[south,west],[north,east]], opacity=0.5,
            name=name, overlay=True, show=show_default
        ).add_to(m)
        return vmin, vmax
    except Exception as e:
        st.warning(f"⚠️ Skipping heatmap {name}: {e}")
        return None, None

bounds = st.session_state.get("map_bounds", compute_bounds_for_heatmaps())
ydf = st.session_state.get("yield_df", pd.DataFrame())
df = None
if (not ydf.empty) and {"latitude","longitude","yield"}.issubset({c.lower() for c in ydf.columns}):
    y2 = ydf.rename(columns={c:c.lower() for c in ydf.columns})
    df = y2.rename(columns={"yield":"Yield","latitude":"Latitude","longitude":"Longitude"})[["Latitude","Longitude","Yield"]].copy()

if df is None or df.empty:
    # fallback single point at map center, driven by Target Yield input
    lat_center = (bounds[0]+bounds[2])/2.0; lon_center = (bounds[1]+bounds[3])/2.0
    tgt = st.number_input("Set Target Yield (bu/ac)", min_value=0.0, value=200.0, step=1.0)
    df = pd.DataFrame({"Yield":[tgt], "Latitude":[lat_center], "Longitude":[lon_center]})

# sell price for profits: default to corn price unless overridden
sell_price = _to_float(st.session_state.get("sell_price", st.session_state.get("corn_price", 5.0)), 5.0)

# VR costs (safe)
def _safe_sum(df_like, col):
    try:
        if df_like is None or df_like.empty: return 0.0
        return float(pd.to_numeric(df_like.get(col, 0), errors="coerce").fillna(0).sum())
    except Exception: return 0.0

fert_var = _safe_sum(st.session_state.get("fert_products", pd.DataFrame()), "CostPerAcre")
seed_var = _safe_sum(st.session_state.get("seed_products", pd.DataFrame()), "CostPerAcre")

try:
    df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce").fillna(0.0)
    df["Revenue_per_acre"] = df["Yield"] * sell_price
    df["NetProfit_per_acre_variable"] = df["Revenue_per_acre"] - (base_expenses_per_acre + fert_var + seed_var)

    fixed_costs = 0.0
    if "fixed_products" in st.session_state and not st.session_state["fixed_products"].empty:
        fx = st.session_state["fixed_products"].copy()
        if "$/ac" not in fx.columns:
            fx["$/ac"] = fx.apply(lambda r: _to_float(r.get("Rate",0))*_to_float(r.get("CostPerUnit",0)), axis=1)
        fixed_costs = _to_float(fx["$/ac"].sum(), 0.0)
    df["NetProfit_per_acre_fixed"] = df["Revenue_per_acre"] - (base_expenses_per_acre + fixed_costs)
except Exception:
    df["Revenue_per_acre"] = 0.0
    df["NetProfit_per_acre_variable"] = 0.0
    df["NetProfit_per_acre_fixed"] = 0.0

if "legend_index" not in st.session_state: st.session_state["legend_index"] = 0

ymin,ymax = add_heatmap_overlay(df, df["Yield"].values, "Yield (bu/ac)", plt.cm.RdYlGn, False, bounds)
if ymin is not None:
    add_gradient_legend("Yield (bu/ac)", ymin, ymax, plt.cm.RdYlGn, st.session_state["legend_index"])
    st.session_state["legend_index"] += 1

vmin,vmax = add_heatmap_overlay(df, df["NetProfit_per_acre_variable"].values, "Variable Rate Profit ($/ac)", plt.cm.RdYlGn, True, bounds)
if vmin is not None:
    add_gradient_legend("Variable Rate Profit ($/ac)", vmin, vmax, plt.cm.RdYlGn, st.session_state["legend_index"])
    st.session_state["legend_index"] += 1

fmin,fmax = add_heatmap_overlay(df, df["NetProfit_per_acre_fixed"].values, "Fixed Rate Profit ($/ac)", plt.cm.RdYlGn, False, bounds)
if fmin is not None:
    add_gradient_legend("Fixed Rate Profit ($/ac)", fmin, fmax, plt.cm.RdYlGn, st.session_state["legend_index"])
    st.session_state["legend_index"] += 1

safe_fit_bounds(m, compute_bounds_for_heatmaps())

# =========================================================
# 8. DISPLAY MAP (height tight)
# =========================================================
st_folium(m, use_container_width=True, height=560)

# =========================================================
# 9. PROFIT SUMMARY (no scroll)
# =========================================================
st.header("Profit Summary")

# ensure keys exist
for k, val in [("fert_products", pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])),
               ("seed_products", pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])),
               ("zones_gdf", None),
               ("yield_df", None)]:
    if k not in st.session_state: st.session_state[k] = val

revenue_per_acre = 0.0; net_profit_per_acre = 0.0
expenses_per_acre = base_expenses_per_acre

if st.session_state["yield_df"] is not None and not st.session_state["yield_df"].empty:
    _df = st.session_state["yield_df"]
    if "Revenue_per_acre" in _df.columns: revenue_per_acre = _to_float(_df["Revenue_per_acre"].mean(), 0.0)
    if "NetProfit_per_acre" in _df.columns: net_profit_per_acre = _to_float(_df["NetProfit_per_acre"].mean(), 0.0)

col_left, col_right = st.columns([2,2], gap="small")

with col_left:
    st.subheader("Breakeven Budget Tool (Corn vs Beans)")
    cy = _to_float(st.session_state.get("corn_yield", 200.0), 200.0)
    cp = _to_float(st.session_state.get("corn_price", 5.0), 5.0)
    sy = _to_float(st.session_state.get("bean_yield", 60.0), 60.0)
    sp = _to_float(st.session_state.get("bean_price", 12.0), 12.0)

    corn_rev = cy*cp; bean_rev = sy*sp
    corn_budget = corn_rev - expenses_per_acre
    bean_budget = bean_rev - expenses_per_acre

    breakeven_df = pd.DataFrame({
        "Crop":["Corn","Soybeans"],
        "Yield Goal (bu/ac)":[cy, sy],
        "Sell Price ($/bu)":[cp, sp],
        "Revenue ($/ac)":[corn_rev, bean_rev],
        "Fixed Inputs ($/ac)":[expenses_per_acre, expenses_per_acre],
        "Breakeven Budget ($/ac)":[corn_budget, bean_budget]
    })

    def _hl_budget(v):
        if isinstance(v,(int,float)):
            if v>0: return "color:#22c55e;font-weight:700;"
            if v<0: return "color:#ef4444;font-weight:700;"
        return "font-weight:700;"

    st.dataframe(
        breakeven_df.style.applymap(_hl_budget, subset=["Breakeven Budget ($/ac)"]).format({
            "Yield Goal (bu/ac)":"{:,.0f}","Sell Price ($/bu)":"${:,.2f}",
            "Revenue ($/ac)":"${:,.0f}","Fixed Inputs ($/ac)":"${:,.0f}",
            "Breakeven Budget ($/ac)":"${:,.0f}"
        }),
        use_container_width=True, hide_index=True,
        height=df_px_height(2, row_h=24, header=28, pad=2)
    )

    st.subheader("Profit Metrics Comparison")
    fert_costs = _to_float(st.session_state["fert_products"]["CostPerAcre"].sum(),0.0) \
                 if not st.session_state["fert_products"].empty else 0.0
    seed_costs = _to_float(st.session_state["seed_products"]["CostPerAcre"].sum(),0.0) \
                 if not st.session_state["seed_products"].empty else 0.0
    y = st.session_state.get("yield_df")
    revenue_var = _to_float(y["Revenue_per_acre"].mean(),0.0) if (y is not None and not y.empty and "Revenue_per_acre" in y.columns) else 0.0
    expenses_var = expenses_per_acre + fert_costs + seed_costs
    var_profit = revenue_var - expenses_var

    fixed_profit = 0.0; revenue_fixed = revenue_var; expenses_fixed = expenses_per_acre
    if "fixed_products" in st.session_state and not st.session_state["fixed_products"].empty:
        _fx = st.session_state["fixed_products"].copy()
        if "$/ac" not in _fx.columns:
            _fx["$/ac"] = _fx.apply(lambda r: _to_float(r.get("Rate",0))*_to_float(r.get("CostPerUnit",0)), axis=1)
        fert_fixed_costs = _to_float(_fx[_fx.get("Type","")== "Fertilizer"]["$/ac"].sum(), 0.0)
        seed_fixed_costs = _to_float(_fx[_fx.get("Type","")== "Seed"]["$/ac"].sum(), 0.0)
        expenses_fixed = expenses_per_acre + fert_fixed_costs + seed_fixed_costs
        fixed_profit = revenue_fixed - expenses_fixed

    comparison = pd.DataFrame({
        "Metric":["Revenue ($/ac)","Expenses ($/ac)","Profit ($/ac)"],
        "Breakeven Budget":[round(revenue_per_acre,2), round(expenses_per_acre,2), round(net_profit_per_acre,2)],
        "Variable Rate":[round(revenue_var,2), round(expenses_var,2), round(var_profit,2)],
        "Fixed Rate":[round(revenue_fixed,2), round(expenses_fixed,2), round(fixed_profit,2)]
    })

    def _hl_profit(v):
        if isinstance(v,(int,float)):
            if v>0: return "color:#22c55e;font-weight:700;"
            if v<0: return "color:#ef4444;font-weight:700;"
        return "font-weight:700;"

    st.dataframe(
        comparison.style.applymap(_hl_profit, subset=["Breakeven Budget","Variable Rate","Fixed Rate"]).format({
            "Breakeven Budget":"${:,.2f}","Variable Rate":"${:,.2f}","Fixed Rate":"${:,.2f}"
        }),
        use_container_width=True, hide_index=True,
        height=df_px_height(3, row_h=24, header=28, pad=2)
    )

with col_right:
    st.subheader("Fixed Input Costs")
    fixed_df = pd.DataFrame(list(expenses.items()), columns=["Expense","$/ac"])
    total_fixed = pd.DataFrame([{"Expense":"Total Fixed Costs","$/ac":fixed_df["$/ac"].sum()}])
    fixed_df = pd.concat([fixed_df, total_fixed], ignore_index=True)
    styled = fixed_df.style.format({"$/ac":"${:,.2f}"}).apply(
        lambda s: ["font-weight:700;" if v=="Total Fixed Costs" else "" for v in s], subset=["Expense"]
    ).apply(
        lambda s: ["font-weight:700;" if i==len(fixed_df)-1 else "" for i in range(len(s))], subset=["$/ac"]
    )
    st.dataframe(styled, use_container_width=True, hide_index=True,
                 height=df_px_height(len(fixed_df), row_h=24, header=28, pad=2))
