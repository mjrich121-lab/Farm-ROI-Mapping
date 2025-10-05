# =========================================================
# Farm Profit Mapping Tool V4 — Ultra-Compact (Minimal Options)
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
import tempfile
from branca.element import MacroElement, Template
from matplotlib import colors as mpl_colors

# -------------------- App shell --------------------
st.set_page_config(page_title="Farm ROI Tool V4", layout="wide")
st.title("Farm Profit Mapping Tool V4")

# =========================================================
# GLOBAL COMPACT UI STYLING
# =========================================================
st.markdown(
    """
    <style>
    /* Tight gutters + tighter vertical rhythm */
    div[data-testid="column"] { padding-left:.15rem !important; padding-right:.15rem !important; }
    section[data-testid="stVerticalBlock"] > div { padding-top:.15rem !important; padding-bottom:.15rem !important; }

    h1 { margin:.25rem 0 .25rem 0 !important; font-size:1.2rem !important; }
    h2, h3 { margin:.2rem 0 .15rem 0 !important; font-size:1rem !important; }

    /* Very small expanders */
    div[data-testid="stExpander"] details summary { padding:.2rem .4rem !important; font-size:.8rem !important; }
    div[data-testid="stExpander"] details > div { padding:.2rem .4rem !important; }

    /* Number inputs compact */
    div[data-testid="stNumberInput"] label { font-size:.72rem !important; margin-bottom:.05rem !important; }
    div[data-testid="stNumberInput"] div[role="spinbutton"]{
        min-height:22px !important; height:22px !important; padding:0 4px !important; font-size:.78rem !important;
    }
    div[data-testid="stNumberInput"] button{ padding:0 !important; min-width:16px !important; }
    div[data-testid="stNumberInput"]{ width:132px !important; max-width:132px !important; }

    /* Uploaders */
    div[data-testid="stFileUploaderDropzone"]{ padding:.2rem !important; min-height:32px !important; }
    div[data-testid="stFileUploaderDropzone"] p{ font-size:.65rem !important; margin:0 !important; }

    /* DataFrames / DataEditors */
    div[data-testid="stDataFrame"] table { font-size:.75rem !important; }
    div[data-testid="stDataFrame"] th, div[data-testid="stDataFrame"] td { padding:1px 4px !important; line-height:1rem !important; }
    div[data-testid="stDataEditor"] table { font-size:.75rem !important; }
    div[data-testid="stDataEditor"] th, div[data-testid="stDataEditor"] td { padding:1px 4px !important; line-height:1rem !important; }

    /* Tiny captions */
    div[data-testid="stCaptionContainer"]{ margin:.1rem 0 !important; font-size:.7rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HELPERS
# =========================================================
def df_px_height(nrows: int, row_h: int = 28, header: int = 34, pad: int = 2) -> int:
    """Exact pixel height so tables/editors show with NO internal scroll."""
    return int(header + nrows * row_h + pad)

def load_vector_file(uploaded_file):
    """Read GeoJSON/JSON/ZIP(SHP) into EPSG:4326 GeoDataFrame."""
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
                for f_name in os.listdir(tmpdir):
                    if f_name.lower().endswith(".shp"):
                        shp_path = os.path.join(tmpdir, f_name)
                        break
                if not shp_path: return None
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
    """Normalize prescription tables -> product, Acres, CostTotal, CostPerAcre"""
    if file is None:
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])
    try:
        if file.name.lower().endswith((".geojson",".json",".zip",".shp")):
            gdf = load_vector_file(file)
            if gdf is None or gdf.empty:
                st.error(f"❌ Could not read {prescrip_type} map.")
                return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])
            gdf.columns = [c.strip().lower().replace(" ", "_") for c in gdf.columns]
            if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
            else: gdf = gdf.to_crs(epsg=4326)
            gdf["Longitude"] = gdf.geometry.representative_point().x
            gdf["Latitude"]  = gdf.geometry.representative_point().y
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        else:
            df = pd.read_csv(file)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    except Exception as e:
        st.warning(f"⚠️ Failed to read {file.name}: {e}")
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])

    if "product" not in df.columns:
        for c in ["variety","hybrid","type","name","material"]:
            if c in df.columns:
                df.rename(columns={c: "product"}, inplace=True)
                break
        else:
            df["product"] = prescrip_type.capitalize()
    if "acres" not in df.columns:
        df["acres"] = 0.0

    # Compact options
    with st.expander(f"⚙️ {prescrip_type.capitalize()} Map Options — {file.name}", expanded=False):
        override = st.number_input(
            "Override Acres Per Polygon", min_value=0.0, value=0.0, step=0.1,
            key=f"{prescrip_type}_{file.name}_override"
        )
        if override > 0: df["acres"] = override

    if "costtotal" not in df.columns:
        if {"price_per_unit","units"}.issubset(df.columns):
            df["costtotal"] = df["price_per_unit"] * df["units"]
        elif {"rate","price"}.issubset(df.columns):
            df["costtotal"] = df["rate"] * df["price"]
        else:
            df["costtotal"] = 0

    if df.empty:
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])

    grouped = df.groupby("product", as_index=False).agg(
        Acres=("acres","sum"), CostTotal=("costtotal","sum")
    )
    grouped["CostPerAcre"] = grouped.apply(
        lambda r: r["CostTotal"]/r["Acres"] if r["Acres"]>0 else 0, axis=1
    )
    return grouped

def get_var_costs():
    """Sum CostPerAcre for VR seed/fert from either *_products or *_layers_store."""
    fert_cost = 0.0; seed_cost = 0.0
    # direct products tables
    if "fert_products" in st.session_state and not st.session_state["fert_products"].empty:
        fert_cost += float(st.session_state["fert_products"]["CostPerAcre"].sum())
    if "seed_products" in st.session_state and not st.session_state["seed_products"].empty:
        seed_cost += float(st.session_state["seed_products"]["CostPerAcre"].sum())
    # fall back to layer stores (dict of grouped frames)
    if "fert_layers_store" in st.session_state:
        for df in st.session_state["fert_layers_store"].values():
            if not df.empty: fert_cost += float(df["CostPerAcre"].sum())
    if "seed_layers_store" in st.session_state:
        for df in st.session_state["seed_layers_store"].values():
            if not df.empty: seed_cost += float(df["CostPerAcre"].sum())
    return fert_cost, seed_cost

# =========================================================
# 2–3. FILE UPLOADS — COMPACT 4-UP
# =========================================================
st.markdown("### Upload Maps")
c1, c2, c3, c4 = st.columns(4)

# ZONE MAP
with c1:
    zone_file = st.file_uploader("Zone", type=["geojson","json","zip"], key="zone_file", accept_multiple_files=False)
    if zone_file:
        gdf = load_vector_file(zone_file)
        if gdf is not None and not gdf.empty:
            st.caption(f"✅ Zones: {len(gdf)} polys")
            # detect/create Zone col
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

            # editor (no scroll)
            disp = gdf[["Zone","Calculated Acres","Override Acres"]].copy()
            edited = st.data_editor(
                disp, num_rows="fixed", hide_index=True, use_container_width=True,
                column_config={
                    "Zone": st.column_config.TextColumn(disabled=True),
                    "Calculated Acres": st.column_config.NumberColumn(format="%.2f", disabled=True),
                    "Override Acres": st.column_config.NumberColumn(format="%.2f"),
                },
                key="zone_acres_editor",
                height=df_px_height(len(disp), row_h=26, header=30, pad=2)
            )
            edited["Override Acres"] = pd.to_numeric(edited["Override Acres"], errors="coerce").fillna(edited["Calculated Acres"])
            gdf["Override Acres"] = edited["Override Acres"].astype(float).values

            st.caption(f"Total Acres — Calc: {gdf['Calculated Acres'].sum():,.2f} | Override: {gdf['Override Acres'].sum():,.2f}")
            st.session_state["zones_gdf"] = gdf
        else:
            st.error("❌ Could not load zone map.")
    else:
        st.caption("No zone file")

# YIELD MAPS
with c2:
    yield_files = st.file_uploader("Yield", type=["csv","geojson","json","zip"], key="yield", accept_multiple_files=True)
    st.session_state.setdefault("yield_df", pd.DataFrame())
    if yield_files:
        rows = 0
        frames = []
        for yf in yield_files:
            try:
                if yf.name.lower().endswith(".csv"):
                    df = pd.read_csv(yf)
                else:
                    yg = load_vector_file(yf)
                    df = pd.DataFrame(yg.drop(columns="geometry", errors="ignore")) if yg is not None else pd.DataFrame()
                if not df.empty:
                    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
                    ycols = [c for c in df.columns if any(k in c for k in ["yld_vol_dr","yld_mass_dr","yield_dry","dry_yield","yield","wet_yield"])]
                    if ycols:
                        df.rename(columns={ycols[0]:"Yield"}, inplace=True)
                    else:
                        df["Yield"] = 0.0
                    frames.append(df)
                    rows += len(df)
            except Exception as e:
                st.warning(f"⚠️ {yf.name}: {e}")
        st.session_state["yield_df"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        st.caption(f"Loaded: {len(yield_files)} file(s), rows={rows:,}")
    else:
        st.caption("No yield files")

# FERT MAPS
with c3:
    fert_files = st.file_uploader("Fert", type=["csv","geojson","json","zip"], key="fert", accept_multiple_files=True)
    st.session_state["fert_layers_store"] = {}
    if fert_files:
        summ = []
        for f in fert_files:
            grouped = process_prescription(f, "fertilizer")
            if not grouped.empty:
                key = os.path.splitext(f.name)[0].lower().replace(" ","_")
                st.session_state["fert_layers_store"][key] = grouped
                summ.append({"File": f.name, "Products": len(grouped)})
        if summ:
            st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True,
                         height=df_px_height(len(summ)))
    else:
        st.caption("No fert files")

# SEED MAPS
with c4:
    seed_files = st.file_uploader("Seed", type=["csv","geojson","json","zip"], key="seed", accept_multiple_files=True)
    st.session_state["seed_layers_store"] = {}
    if seed_files:
        summ = []
        for f in seed_files:
            grouped = process_prescription(f, "seed")
            if not grouped.empty:
                key = os.path.splitext(f.name)[0].lower().replace(" ","_")
                st.session_state["seed_layers_store"][key] = grouped
                summ.append({"File": f.name, "Products": len(grouped)})
        if summ:
            st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True,
                         height=df_px_height(len(summ)))
    else:
        st.caption("No seed files")

# =========================================================
# 4. ULTRA-COMPACT COSTS + ASSUMPTIONS
# =========================================================
st.markdown("### Costs & Assumptions")
# ------------------------------
# Fixed Inputs ($/ac) — 12 in ONE row (ultra-compact)
# ------------------------------
# If your helper is named _mini_num, this alias makes the same code work.
num = (mini_num if "mini_num" in globals() else _mini_num)

# Make the number boxes narrower so 12 fit across
st.markdown(
    """
    <style>
      /* override any earlier width for this specific row */
      div[data-testid="fixed-row"] div[data-testid="stNumberInput"]{
          width:100px !important; max-width:100px !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.caption("Fixed Inputs ($/ac)")
fixed_row = st.columns(12, gap="small")
with fixed_row[0]:  chemicals      = num("Chem ($/ac)",        "chem",   0.0, 1.0)
with fixed_row[1]:  insurance      = num("Insur ($/ac)",       "ins",    0.0, 1.0)
with fixed_row[2]:  insecticide    = num("Insect/Fung ($/ac)", "insect", 0.0, 1.0)
with fixed_row[3]:  fertilizer     = num("Fert Flat ($/ac)",   "fert",   0.0, 1.0)
with fixed_row[4]:  seed           = num("Seed Flat ($/ac)",   "seed",   0.0, 1.0)
with fixed_row[5]:  cash_rent      = num("Cash Rent ($/ac)",   "rent",   0.0, 1.0)
with fixed_row[6]:  machinery      = num("Mach ($/ac)",        "mach",   0.0, 1.0)
with fixed_row[7]:  labor          = num("Labor ($/ac)",       "labor",  0.0, 1.0)
with fixed_row[8]:  coliving       = num("Living ($/ac)",      "col",    0.0, 1.0)
with fixed_row[9]:  extra_fuel     = num("Fuel ($/ac)",        "fuel",   0.0, 1.0)
with fixed_row[10]: extra_interest = num("Interest ($/ac)",    "int",    0.0, 1.0)
with fixed_row[11]: truck_fuel     = num("Truck Fuel ($/ac)",  "truck",  0.0, 1.0)

# Pack into the same dict/total the rest of your app expects
expenses = {
    "Chemicals": chemicals,
    "Insurance": insurance,
    "Insecticide/Fungicide": insecticide,
    "Fertilizer (Flat)": fertilizer,
    "Seed (Flat)": seed,
    "Cash Rent": cash_rent,
    "Machinery": machinery,
    "Labor": labor,
    "Cost of Living": coliving,
    "Extra Fuel": extra_fuel,
    "Extra Interest": extra_interest,
    "Truck Fuel": truck_fuel,
}
base_expenses_per_acre = float(sum(expenses.values()))


# One-row assumptions: Sell Price + (Target Yield only if no map)
ass_cols = st.columns([1,1,2])
with ass_cols[0]:
    st.caption("Sell Price ($/bu)")
    st.session_state["sell_price"] = st.number_input(
        "sell_price", min_value=0.0, value=float(st.session_state.get("sell_price", 5.00)),
        step=0.05, label_visibility="collapsed"
    )

with ass_cols[1]:
    has_yield = (not st.session_state.get("yield_df", pd.DataFrame()).empty) \
                and {"latitude","longitude","yield"}.issubset(
                    set(c.lower() for c in st.session_state["yield_df"].columns)
                )
    if not has_yield:
        st.caption("Target Yield (bu/ac)")
        st.session_state["target_yield"] = st.number_input(
            "target_yield", min_value=0.0, value=float(st.session_state.get("target_yield", 200.0)),
            step=1.0, label_visibility="collapsed"
        )
    else:
        st.caption("Target Yield (bu/ac)")
        st.markdown("<div style='opacity:.6;'>Using uploaded yield map</div>", unsafe_allow_html=True)

with ass_cols[2]:
    # Tiny preview (no scroll)
    tgt = float(st.session_state.get("target_yield", 200.0))
    px  = float(st.session_state.get("sell_price", 5.0))
    breakeven = tgt * px - base_expenses_per_acre
    prev_df = pd.DataFrame([{"Yield":tgt,"Price":px,"Revenue":tgt*px,"Fixed":base_expenses_per_acre,"Breakeven":breakeven}])
    def _hl(val):
        if isinstance(val,(int,float)):
            if val>0: return "color:#22c55e; font-weight:700;"
            if val<0: return "color:#ef4444; font-weight:700;"
        return "font-weight:700;"
    st.dataframe(
        prev_df.style.applymap(_hl, subset=["Breakeven"]).format({
            "Yield":"{:.0f}","Price":"${:.2f}","Revenue":"${:,.0f}","Fixed":"${:,.0f}","Breakeven":"${:,.0f}"
        }),
        use_container_width=True, hide_index=True, height=df_px_height(1, row_h=26, header=30, pad=2)
    )

# Fixed & Variable inputs expanders
fx_col, vr_col = st.columns(2)
with fx_col:
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
# 5. BASE MAP (satellite + reference + toggles)
# =========================================================
def make_base_map():
    try:
        m = folium.Map(
            location=[39.5, -98.35], zoom_start=5, min_zoom=2,
            tiles=None, scrollWheelZoom=False, prefer_canvas=True
        )
        # Esri Imagery + reference
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri", overlay=False, control=True, name="Esri World Imagery"
        ).add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
            attr="Esri", overlay=True, control=True, name="Reference Labels"
        ).add_to(m)

        # Enable scrollwheel only on click
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
    except Exception as e:
        st.error(f"❌ Failed to build base map: {e}")
        return folium.Map(location=[39.5,-98.35], zoom_start=4)

m = make_base_map()

# =========================================================
# 6. ZONES OVERLAY + LEGEND
# =========================================================
def add_zones_overlay(m):
    zones_gdf = st.session_state.get("zones_gdf")
    if zones_gdf is None or zones_gdf.empty: return m
    try:
        zones_gdf = zones_gdf.to_crs(epsg=4326)
        if "Zone" not in zones_gdf.columns:
            zones_gdf["Zone"] = range(1, len(zones_gdf)+1)

        # center/zoom
        zb = zones_gdf.total_bounds
        m.location = [(zb[1]+zb[3])/2, (zb[0]+zb[2])/2]
        m.zoom_start = 15

        palette = ["#FF0000","#FF8C00","#FFFF00","#32CD32","#006400",
                   "#1E90FF","#8A2BE2","#FFC0CB","#A52A2A","#00CED1"]
        uniq = list(dict.fromkeys(sorted(zones_gdf["Zone"].astype(str).unique())))
        cmap = {z: palette[i % len(palette)] for i, z in enumerate(uniq)}

        folium.GeoJson(
            zones_gdf,
            name="Zones",
            style_function=lambda f: {
                "fillColor": cmap.get(str(f["properties"].get("Zone","")), "#808080"),
                "color": "black", "weight": 1, "fillOpacity": 0.08,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["Zone","Calculated Acres","Override Acres"] if c in zones_gdf.columns]
            )
        ).add_to(m)

        items = "".join(
            f"<div style='display:flex;align-items:center;margin:2px 0;'>"
            f"<div style='background:{cmap[z]};width:14px;height:14px;margin-right:6px;'></div>{z}</div>"
            for z in uniq
        )
        legend_html = f"""
        <div style="position:absolute; bottom:20px; right:20px; z-index:9999;
                    background:rgba(0,0,0,.65); color:white; font:13px sans-serif;
                    padding:6px 10px; border-radius:5px; width:160px;">
          <div style="font-weight:600;margin-bottom:4px;cursor:pointer;"
               onclick="var x=document.getElementById('zl');x.style.display=(x.style.display==='none')?'block':'none';">
            Zone Colors ▼
          </div>
          <div id="zl" style="display:block;">{items}</div>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))
    except Exception as e:
        st.warning(f"⚠️ Skipping zones overlay: {e}")
    return m

add_zones_overlay(m)

# =========================================================
# 7. HEATMAPS (Yield, Variable Profit, Fixed Profit)
# =========================================================
def compute_bounds_for_heatmaps():
    try:
        bnds = []
        for key in ["zones_gdf", "seed_gdf"]:
            g = st.session_state.get(key)
            if g is not None and not g.empty:
                tb = g.total_bounds
                if tb is not None and len(tb) == 4 and not any(pd.isna(tb)):
                    bnds.append([[tb[1], tb[0]], [tb[3], tb[2]]])
        # include fert layers if present
        for _, fg in st.session_state.get("fert_gdfs", {}).items():
            if fg is not None and not fg.empty:
                tb = fg.total_bounds
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
    return 25.0, -125.0, 49.0, -66.0

def add_heatmap_overlay(df, values, name, cmap, show_default, bounds):
    """Interpolate scattered points -> raster overlay."""
    try:
        if df is None or df.empty: return None, None
        vals = pd.to_numeric(pd.Series(values), errors="coerce")
        mask = vals.notna() & df["Latitude"].apply(np.isfinite) & df["Longitude"].apply(np.isfinite)
        if mask.sum() < 3: return None, None

        south, west, north, east = bounds
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

        vmin = float(np.nanpercentile(vals_ok, 5))
        vmax = float(np.nanpercentile(vals_ok, 95))
        if vmin == vmax: vmax = vmin + 1.0

        rgba = cmap((grid - vmin) / (vmax - vmin))
        rgba = np.flipud(rgba)  # folium expects origin at top-left
        rgba = (rgba * 255).astype(np.uint8)

        folium.raster_layers.ImageOverlay(
            image=rgba, bounds=[[south, west], [north, east]],
            opacity=0.5, name=name, overlay=True, show=show_default
        ).add_to(m)
        return vmin, vmax
    except Exception as e:
        st.warning(f"⚠️ Skipping heatmap {name}: {e}")
        return None, None

# Build DF for overlays
bounds = compute_bounds_for_heatmaps()
ydf = st.session_state.get("yield_df", pd.DataFrame()).copy()

if not ydf.empty:
    # try to standardize columns
    cols = {c.lower(): c for c in ydf.columns}
    if "latitude" in cols and "longitude" in cols:
        ydf.rename(columns={cols["latitude"]:"Latitude", cols["longitude"]:"Longitude"}, inplace=True)
else:
    lat_center = (bounds[0] + bounds[2]) / 2.0
    lon_center = (bounds[1] + bounds[3]) / 2.0
    target_yield = float(st.session_state.get("target_yield", 200.0))
    ydf = pd.DataFrame({"Yield":[target_yield], "Latitude":[lat_center], "Longitude":[lon_center]})

# Revenue & profits
sell_price = float(st.session_state.get("sell_price", 5.0))
try:
    ydf["Revenue_per_acre"] = pd.to_numeric(ydf["Yield"], errors="coerce").fillna(0.0) * sell_price
    fert_var, seed_var = get_var_costs()
    ydf["NetProfit_per_acre_variable"] = ydf["Revenue_per_acre"] - (base_expenses_per_acre + fert_var + seed_var)

    fixed_costs = 0.0
    if "fixed_products" in st.session_state and not st.session_state["fixed_products"].empty:
        fx = st.session_state["fixed_products"].copy()
        fx["$/ac"] = fx.apply(lambda r: (r.get("Rate",0) or 0)*(r.get("CostPerUnit",0) or 0), axis=1)
        fixed_costs = float(fx["$/ac"].sum())
    ydf["NetProfit_per_acre_fixed"] = ydf["Revenue_per_acre"] - (base_expenses_per_acre + fixed_costs)
except Exception:
    st.warning("⚠️ Could not compute profit metrics, using defaults.")
    ydf["Revenue_per_acre"] = 0.0
    ydf["NetProfit_per_acre_variable"] = 0.0
    ydf["NetProfit_per_acre_fixed"] = 0.0

# Heatmap overlays + legends
st.session_state.setdefault("legend_index", 0)

y_min, y_max = add_heatmap_overlay(ydf, ydf["Yield"], "Yield (bu/ac)", plt.cm.RdYlGn, False, bounds)
if y_min is not None:
    # legend
    def add_grad_legend(name, vmin, vmax, cmap, index):
        top = 20 + index*80
        stops = ", ".join(f"{mpl_colors.rgb2hex(cmap(i/100.0)[:3])} {i}%" for i in range(0,101,10))
        html = f"""
        <div style="position:absolute; top:{top}px; left:10px; z-index:9999;
                    background:rgba(0,0,0,.65); color:white; font:12px sans-serif;
                    padding:6px 10px; border-radius:5px; width:180px;">
          <div style="font-weight:600; margin-bottom:4px;">{name}</div>
          <div style="height:14px; background:linear-gradient(90deg, {stops}); border-radius:2px; margin-bottom:4px;"></div>
          <div style="display:flex;justify-content:space-between;"><span>{vmin:.1f}</span><span>{vmax:.1f}</span></div>
        </div>"""
        m.get_root().html.add_child(folium.Element(html))
    add_grad_legend("Yield (bu/ac)", y_min, y_max, plt.cm.RdYlGn, st.session_state["legend_index"])
    st.session_state["legend_index"] += 1

v_min, v_max = add_heatmap_overlay(ydf, ydf["NetProfit_per_acre_variable"], "Variable Rate Profit ($/ac)", plt.cm.RdYlGn, True, bounds)
if v_min is not None:
    add_grad_legend("Variable Rate Profit ($/ac)", v_min, v_max, plt.cm.RdYlGn, st.session_state["legend_index"])
    st.session_state["legend_index"] += 1

f_min, f_max = add_heatmap_overlay(ydf, ydf["NetProfit_per_acre_fixed"], "Fixed Rate Profit ($/ac)", plt.cm.RdYlGn, False, bounds)
if f_min is not None:
    add_grad_legend("Fixed Rate Profit ($/ac)", f_min, f_max, plt.cm.RdYlGn, st.session_state["legend_index"])
    st.session_state["legend_index"] += 1

# Fit bounds
def safe_fit_bounds(m, bounds):
    try:
        south, west, north, east = bounds
        m.fit_bounds([[south, west], [north, east]])
    except Exception:
        pass
safe_fit_bounds(m, bounds)

# Layer control
try:
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
except Exception:
    pass

# Display map
st_folium(m, use_container_width=True, height=550)

# =========================================================
# 9. PROFIT SUMMARY (Minimal)
# =========================================================
st.header("Profit Summary")

# Derived metrics
rev_avg = float(ydf["Revenue_per_acre"].mean()) if "Revenue_per_acre" in ydf.columns else 0.0
fert_var, seed_var = get_var_costs()
var_exp = base_expenses_per_acre + fert_var + seed_var
var_profit = rev_avg - var_exp

fixed_costs = 0.0
if "fixed_products" in st.session_state and not st.session_state["fixed_products"].empty:
    fx = st.session_state["fixed_products"].copy()
    fx["$/ac"] = fx.apply(lambda r: (r.get("Rate",0) or 0)*(r.get("CostPerUnit",0) or 0), axis=1)
    fixed_costs = float(fx["$/ac"].sum())
fix_exp = base_expenses_per_acre + fixed_costs
fix_profit = rev_avg - fix_exp

tgt = float(st.session_state.get("target_yield", 200.0))
px  = float(st.session_state.get("sell_price", 5.0))
be_rev = tgt * px
be_exp = base_expenses_per_acre
be_profit = be_rev - be_exp

summary = pd.DataFrame({
    "Metric": ["Revenue ($/ac)", "Expenses ($/ac)", "Profit ($/ac)"],
    "Breakeven (Target)": [be_rev, be_exp, be_profit],
    "Variable Rate": [rev_avg, var_exp, var_profit],
    "Fixed Rate": [rev_avg, fix_exp, fix_profit]
})

def _hl_summary(val):
    if isinstance(val,(int,float)):
        if val>0: return "color:#22c55e; font-weight:700;"
        if val<0: return "color:#ef4444; font-weight:700;"
    return "font-weight:700;"

st.dataframe(
    summary.style.applymap(_hl_summary, subset=["Breakeven (Target)","Variable Rate","Fixed Rate"]).format({
        "Breakeven (Target)":"${:,.2f}",
        "Variable Rate":"${:,.2f}",
        "Fixed Rate":"${:,.2f}",
    }),
    use_container_width=True,
    hide_index=True,
    height=df_px_height(len(summary))
)

# Fixed input costs table (no scroll)
fixed_df = pd.DataFrame(list(expenses.items()), columns=["Expense","$/ac"])
fixed_df = pd.concat([fixed_df, pd.DataFrame([{"Expense":"Total Fixed Costs","$/ac":fixed_df["$/ac"].sum()}])], ignore_index=True)
st.dataframe(
    fixed_df.style.format({"$/ac":"${:,.2f}"}).apply(
        lambda s: ["font-weight:700;" if v=="Total Fixed Costs" else "" for v in s], subset=["Expense"]
    ).apply(
        lambda s: ["font-weight:700;" if i==len(fixed_df)-1 else "" for i in range(len(s))], subset=["$/ac"]
    ),
    use_container_width=True,
    hide_index=True,
    height=df_px_height(len(fixed_df))
)


