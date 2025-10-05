# =========================================================
# Farm Profit Mapping Tool V4 — WORKING + COMPACT TABLES
# =========================================================
import os
import zipfile
import tempfile
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

# ---------------------------------------------------------
# Page + compact styling (tables & inputs only)
# ---------------------------------------------------------
st.set_page_config(page_title="Farm ROI Tool V4", layout="wide")
st.title("Farm Profit Mapping Tool V4")

st.markdown("""
<style>
/* Tighter gutters between columns */
div[data-testid="column"]{
  padding-left:.22rem !important;
  padding-right:.22rem !important;
}
/* Reduce vertical whitespace between blocks */
section[data-testid="stVerticalBlock"] > div{
  padding-top:.22rem !important;
  padding-bottom:.22rem !important;
}
/* Smaller headers, but readable */
h1{ margin:.35rem 0 .3rem 0 !important; font-size:1.22rem !important; }
h2,h3{ margin:.28rem 0 .2rem 0 !important; font-size:1.0rem !important; }

/* Compact number inputs */
div[data-testid="stNumberInput"] label{
  font-size:.78rem !important;
  margin-bottom:.12rem !important;
}
div[data-testid="stNumberInput"] div[role="spinbutton"]{
  min-height:28px !important;
  height:28px !important;
  padding:0 6px !important;
  font-size:.86rem !important;
}
div[data-testid="stNumberInput"] button{
  padding:0 !important; min-width:22px !important;
}

/* Make DataFrame / DataEditor rows tighter */
div[data-testid="stDataFrame"] table,
div[data-testid="stDataEditor"] table{
  font-size:.86rem !important;
}
div[data-testid="stDataFrame"] th, div[data-testid="stDataFrame"] td,
div[data-testid="stDataEditor"] th, div[data-testid="stDataEditor"] td{
  padding:2px 6px !important;     /* <- compact cells */
  line-height:1.15rem !important; /* <- tighter rows */
}

/* Tiny captions */
div[data-testid="stCaptionContainer"]{
  margin:.16rem 0 !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* pull page content up a bit */
.block-container { padding-top:.35rem !important; }

/* uploader: smaller footprint */
div[data-testid="stFileUploader"] { margin-top:.15rem !important; }
div[data-testid="stFileUploaderDropzone"]{
  padding:.25rem !important;           /* tighter box */
  min-height:42px !important;
}
div[data-testid="stFileUploaderDropzone"] p{
  margin:0 !important; font-size:.78rem !important;
}
</style>
""", unsafe_allow_html=True)
zone_file = st.file_uploader(
    "Zone", type=["geojson","json","zip"], key="up_zone",
    accept_multiple_files=False, label_visibility="collapsed"
)
yield_files = st.file_uploader(
    "Yield", type=["csv","geojson","json","zip"], key="up_yield",
    accept_multiple_files=True, label_visibility="collapsed"
)
fert_files = st.file_uploader(
    "Fert", type=["csv","geojson","json","zip"], key="up_fert",
    accept_multiple_files=True, label_visibility="collapsed"
)
seed_files = st.file_uploader(
    "Seed", type=["csv","geojson","json","zip"], key="up_seed",
    accept_multiple_files=True, label_visibility="collapsed"
)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def df_px_height(nrows: int, row_h: int = 28, header: int = 34, pad: int = 2) -> int:
    """Exact height so tables don't scroll internally."""
    nrows = max(1, int(nrows))
    return int(header + nrows * row_h + pad)

def load_vector_file(uploaded_file):
    """Read .geojson/.json/.zip(SHP)/.shp into EPSG:4326 GeoDataFrame."""
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
    If CSV (no geometry), original_gdf_or_None will be None.
    """
    if file is None:
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"]), None
    try:
        name = file.name.lower()
        gdf_orig = None
        if name.endswith((".geojson",".json",".zip",".shp")):
            gdf = load_vector_file(file)
            if gdf is None or gdf.empty:
                st.error(f"❌ Could not read {prescrip_type} map.")
                return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"]), None
            gdf.columns = [c.strip().lower().replace(" ", "_") for c in gdf.columns]
            # Keep the original gdf for overlay:
            gdf_orig = gdf.copy()
            # Also make a flat table for costs:
            gdf["Longitude"] = gdf.geometry.representative_point().x
            gdf["Latitude"]  = gdf.geometry.representative_point().y
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        else:
            df = pd.read_csv(file)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        if "product" not in df.columns:
            for c in ["variety","hybrid","type","name","material"]:
                if c in df.columns:
                    df.rename(columns={c:"product"}, inplace=True)
                    break
            else:
                df["product"] = prescrip_type.capitalize()
        if "acres" not in df.columns:
            df["acres"] = 0.0

        # Estimate total cost if missing
        if "costtotal" not in df.columns:
            if {"price_per_unit","units"}.issubset(df.columns):
                df["costtotal"] = df["price_per_unit"] * df["units"]
            elif {"rate","price"}.issubset(df.columns):
                df["costtotal"] = df["rate"] * df["price"]
            else:
                df["costtotal"] = 0.0

        if df.empty:
            return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"]), gdf_orig

        grouped = df.groupby("product", as_index=False).agg(
            Acres=("acres","sum"),
            CostTotal=("costtotal","sum")
        )
        grouped["CostPerAcre"] = grouped.apply(
            lambda r: r["CostTotal"]/r["Acres"] if r["Acres"]>0 else 0, axis=1
        )
        return grouped, gdf_orig
    except Exception as e:
        st.warning(f"⚠️ Failed to read {file.name}: {e}")
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"]), None

def find_col(df, names):
    """Return first matching column name (case-insensitive), else None."""
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n in cols: return cols[n]
    return None

# ---------------------------------------------------------
# Uploaders (one tight row) + summaries per category
# ---------------------------------------------------------
st.subheader("Upload Maps")
u1,u2,u3,u4 = st.columns(4)

# --- Zones ---
with u1:
    st.caption("Zone Map · GeoJSON/JSON/ZIP(SHP)")
    zone_file = st.file_uploader("Zone", type=["geojson","json","zip"], key="up_zone", accept_multiple_files=False)
    if zone_file:
        zones_gdf = load_vector_file(zone_file)
        if zones_gdf is not None and not zones_gdf.empty:
            # detect/create Zone col
            zone_col = None
            for cand in ["Zone","zone","ZONE","Name","name"]:
                if cand in zones_gdf.columns:
                    zone_col = cand; break
            if zone_col is None:
                zones_gdf["ZoneIndex"] = range(1, len(zones_gdf)+1)
                zone_col = "ZoneIndex"
            zones_gdf["Zone"] = zones_gdf[zone_col]

            # acres on equal-area
            g2 = zones_gdf.copy()
            if g2.crs is None: g2.set_crs(epsg=4326, inplace=True)
            if g2.crs.is_geographic: g2 = g2.to_crs(epsg=5070)
            zones_gdf["Calculated Acres"] = (g2.geometry.area * 0.000247105).astype(float)
            zones_gdf["Override Acres"]   = zones_gdf["Calculated Acres"].astype(float)

            # editor (no scroll)
            disp = zones_gdf[["Zone","Calculated Acres","Override Acres"]].copy()
            edited = st.data_editor(
                disp, num_rows="fixed", hide_index=True, use_container_width=True,
                column_config={
                    "Zone": st.column_config.TextColumn(disabled=True),
                    "Calculated Acres": st.column_config.NumberColumn(format="%.2f", disabled=True),
                    "Override Acres": st.column_config.NumberColumn(format="%.2f"),
                },
                height=df_px_height(len(disp))
            )
            edited["Override Acres"] = pd.to_numeric(edited["Override Acres"], errors="coerce") \
                                         .fillna(edited["Calculated Acres"])
            zones_gdf["Override Acres"] = edited["Override Acres"].astype(float).values

            st.caption(f"✅ Zones: {len(zones_gdf)}  |  Total Calc: {zones_gdf['Calculated Acres'].sum():,.2f} ac  |  "
                       f"Override: {zones_gdf['Override Acres'].sum():,.2f} ac")
            st.session_state["zones_gdf"] = zones_gdf
        else:
            st.error("❌ Could not read zone file.")
    else:
        st.caption("No zone file uploaded.")

# --- Yield ---
with u2:
    st.caption("Yield Map(s) · CSV/GeoJSON/JSON/ZIP(SHP)")
    yield_files = st.file_uploader("Yield", type=["csv","geojson","json","zip"], key="up_yield", accept_multiple_files=True)
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
                    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
                    # normalize to "Yield"
                    ycol = find_col(df, ["yield","yld_vol_dr","yld_mass_dr","yield_dry","dry_yield","wet_yield"])
                    if ycol:
                        if ycol != "Yield":
                            df.rename(columns={ycol:"Yield"}, inplace=True)
                    else:
                        df["Yield"] = 0.0
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

# --- Fert ---
with u3:
    st.caption("Fertilizer RX · CSV/GeoJSON/JSON/ZIP(SHP)")
    fert_files = st.file_uploader("Fert", type=["csv","geojson","json","zip"], key="up_fert", accept_multiple_files=True)
    st.session_state["fert_layers_store"] = {}
    st.session_state["fert_gdfs"] = {}
    if fert_files:
        summ = []
        for f in fert_files:
            grouped, gdf_orig = process_prescription(f, "fertilizer")
            if not grouped.empty:
                key = os.path.splitext(f.name)[0].lower().replace(" ","_")
                st.session_state["fert_layers_store"][key] = grouped
                if gdf_orig is not None and not gdf_orig.empty:
                    st.session_state["fert_gdfs"][key] = gdf_orig
                summ.append({"File": f.name, "Products": len(grouped)})
        if summ:
            st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True,
                         height=df_px_height(len(summ)))
    else:
        st.caption("No fertilizer files uploaded.")

# --- Seed ---
with u4:
    st.caption("Seed RX · CSV/GeoJSON/JSON/ZIP(SHP)")
    seed_files = st.file_uploader("Seed", type=["csv","geojson","json","zip"], key="up_seed", accept_multiple_files=True)
    st.session_state["seed_layers_store"] = {}
    st.session_state["seed_gdf"] = None
    if seed_files:
        summ = []
        # If multiple vector seed files arrive, keep the last geometry for overlay
        last_seed_gdf = None
        for f in seed_files:
            grouped, gdf_orig = process_prescription(f, "seed")
            if not grouped.empty:
                key = os.path.splitext(f.name)[0].lower().replace(" ","_")
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

# ---------------------------------------------------------
# Fixed Inputs — 12 inputs in one row (compact, no scroll)
# ---------------------------------------------------------
st.subheader("Fixed Inputs ($/ac)")
r = st.columns(12, gap="small")
def _v(state_key, default=0.0):  # persistent default without crashes
    try:
        return float(st.session_state.get(state_key, default))
    except Exception:
        return float(default)

with r[0]:  chemicals      = st.number_input("Chem", min_value=0.0, value=_v("fi_chem"), step=1.0, key="fi_chem")
with r[1]:  insurance      = st.number_input("Insur", min_value=0.0, value=_v("fi_ins"), step=1.0, key="fi_ins")
with r[2]:  insecticide    = st.number_input("Insect/Fung", min_value=0.0, value=_v("fi_insect"), step=1.0, key="fi_insect")
with r[3]:  fertilizer     = st.number_input("Fert Flat", min_value=0.0, value=_v("fi_fert"), step=1.0, key="fi_fert")
with r[4]:  seed           = st.number_input("Seed Flat", min_value=0.0, value=_v("fi_seed"), step=1.0, key="fi_seed")
with r[5]:  cash_rent      = st.number_input("Cash Rent", min_value=0.0, value=_v("fi_rent"), step=1.0, key="fi_rent")
with r[6]:  machinery      = st.number_input("Mach", min_value=0.0, value=_v("fi_mach"), step=1.0, key="fi_mach")
with r[7]:  labor          = st.number_input("Labor", min_value=0.0, value=_v("fi_labor"), step=1.0, key="fi_labor")
with r[8]:  coliving       = st.number_input("Living", min_value=0.0, value=_v("fi_col"), step=1.0, key="fi_col")
with r[9]:  extra_fuel     = st.number_input("Fuel", min_value=0.0, value=_v("fi_fuel"), step=1.0, key="fi_fuel")
with r[10]: extra_interest = st.number_input("Interest", min_value=0.0, value=_v("fi_int"), step=1.0, key="fi_int")
with r[11]: truck_fuel     = st.number_input("Truck Fuel", min_value=0.0, value=_v("fi_truck"), step=1.0, key="fi_truck")

expenses = {
    "Chemicals": chemicals, "Insurance": insurance, "Insecticide/Fungicide": insecticide,
    "Fertilizer (Flat)": fertilizer, "Seed (Flat)": seed, "Cash Rent": cash_rent,
    "Machinery": machinery, "Labor": labor, "Cost of Living": coliving,
    "Extra Fuel": extra_fuel, "Extra Interest": extra_interest, "Truck Fuel": truck_fuel,
}
base_expenses_per_acre = float(sum(expenses.values()))

# ---------------------------------------------------------
# Corn/Soy strip (one row) + tiny preview (NO SCROLL)
# ---------------------------------------------------------
c1,c2,c3,c4,c5 = st.columns([1,1,1,1,1], gap="small")

with c1:
    st.caption("Corn Yield (bu/ac)")
    st.session_state["corn_yield"] = st.number_input(
        "corn_yld", min_value=0.0, value=float(st.session_state.get("corn_yield",200.0)),
        step=1.0, label_visibility="collapsed"
    )
with c2:
    st.caption("Corn Price ($/bu)")
    st.session_state["corn_price"] = st.number_input(
        "corn_px", min_value=0.0, value=float(st.session_state.get("corn_price",5.0)),
        step=0.05, label_visibility="collapsed"
    )
with c3:
    st.caption("Soy Yield (bu/ac)")
    st.session_state["bean_yield"] = st.number_input(
        "bean_yld", min_value=0.0, value=float(st.session_state.get("bean_yield",60.0)),
        step=1.0, label_visibility="collapsed"
    )
with c4:
    st.caption("Soy Price ($/bu)")
    st.session_state["bean_price"] = st.number_input(
        "bean_px", min_value=0.0, value=float(st.session_state.get("bean_price",12.0)),
        step=0.05, label_visibility="collapsed"
    )
with c5:
    # Only show when no yield map
    ydf = st.session_state.get("yield_df", pd.DataFrame())
    has_yield = isinstance(ydf, pd.DataFrame) and not ydf.empty \
                and {"latitude","longitude","yield"}.issubset({c.lower() for c in ydf.columns})
    if not has_yield:
        st.caption("Target Yield (bu/ac)")
        st.session_state["target_yield"] = st.number_input(
            "target_yld", min_value=0.0, value=float(st.session_state.get("target_yield",200.0)),
            step=1.0, label_visibility="collapsed"
        )
    else:
        st.caption("Target Yield (from map)")
        st.markdown("<div style='opacity:.65;height:28px'></div>", unsafe_allow_html=True)

# ---- Corn/Soy tiny preview (no scroll, fits width) ----
prev_df = pd.DataFrame({
    "Crop": ["Corn", "Soybeans"],
    "Yield": [st.session_state["corn_yield"], st.session_state["bean_yield"]],
    "Price": [st.session_state["corn_price"], st.session_state["bean_price"]],
    "Revenue": [
        st.session_state["corn_yield"] * st.session_state["corn_price"],
        st.session_state["bean_yield"] * st.session_state["bean_price"],
    ],
    "Fixed": [base_expenses_per_acre, base_expenses_per_acre],
})
prev_df["Breakeven"] = prev_df["Revenue"] - prev_df["Fixed"]

# Pre-format as strings so st.table stays small and crisp
prev_view = prev_df.copy()
prev_view["Yield"]     = prev_view["Yield"].map(lambda v: f"{v:.0f}")
prev_view["Price"]     = prev_view["Price"].map(lambda v: f"${v:,.2f}")
prev_view["Revenue"]   = prev_view["Revenue"].map(lambda v: f"${v:,.0f}")
prev_view["Fixed"]     = prev_view["Fixed"].map(lambda v: f"${v:,.0f}")
prev_view["Breakeven"] = prev_view["Breakeven"].map(lambda v: f"${v:,.0f}")

# Slightly tighter table cells for this one table
st.markdown("""
<style>
/* only affects static tables; keeps it compact */
table { width:100%; }
thead th, tbody td { padding:4px 8px !important; }
</style>
""", unsafe_allow_html=True)

st.table(prev_view)

# Single sell price to feed heatmaps/revenue (use Corn price)
sell_price = float(st.session_state.get("corn_price", 5.0))

# ---------------------------------------------------------
# Map
# ---------------------------------------------------------
def make_base_map():
    try:
        m = folium.Map(
            location=[39.5,-98.35], zoom_start=5, min_zoom=2,
            tiles=None, scrollWheelZoom=False, prefer_canvas=True
        )
        # safe tiles
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

        # Enable wheel after click
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
    except Exception:
        return folium.Map(location=[39.5,-98.35], zoom_start=5)

m = make_base_map()

# Zones overlay
def add_zones_overlay(m):
    zones_gdf = st.session_state.get("zones_gdf")
    if zones_gdf is None or zones_gdf.empty:
        return m
    try:
        zones_gdf = zones_gdf.to_crs(epsg=4326)
        if "Zone" not in zones_gdf.columns:
            zones_gdf["Zone"] = range(1, len(zones_gdf)+1)

        tb = zones_gdf.total_bounds
        m.location = [(tb[1]+tb[3])/2, (tb[0]+tb[2])/2]
        m.zoom_start = 15

        palette = ["#FF0000","#FF8C00","#FFFF00","#32CD32","#006400",
                   "#1E90FF","#8A2BE2","#FFC0CB","#A52A2A","#00CED1"]
        unique_vals = list(dict.fromkeys(sorted(list(zones_gdf["Zone"].astype(str).unique()))))
        color_map = {z: palette[i % len(palette)] for i,z in enumerate(unique_vals)}

        folium.GeoJson(
            zones_gdf,
            name="Zones",
            style_function=lambda feat: {
                "fillColor": color_map.get(str(feat["properties"].get("Zone","")),"#808080"),
                "color":"black","weight":1,"fillOpacity":0.08
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["Zone","Calculated Acres","Override Acres"] if c in zones_gdf.columns]
            )
        ).add_to(m)

        items = ""
        for z in unique_vals:
            items += (f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                      f"<div style='background:{color_map[z]};width:14px;height:14px;margin-right:6px;'></div>{z}</div>")
        html = f"""
        <div style="position:absolute; bottom:18px; right:18px; z-index:9999;
                    font-family:sans-serif; font-size:13px; color:white;
                    background-color:rgba(0,0,0,.65); padding:6px 10px; border-radius:5px; width:160px;">
          <div style="font-weight:600; margin-bottom:4px; cursor:pointer;" onclick="
              var x=document.getElementById('zone-legend-items');
              if (x.style.display==='none'){{x.style.display='block';}} else {{x.style.display='none';}}">
            Zone Colors ▼
          </div>
          <div id="zone-legend-items" style="display:block;">{items}</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(html))
    except Exception as e:
        st.warning(f"⚠️ Skipping zones overlay: {e}")
    return m

m = add_zones_overlay(m)

# Bounds helpers
def compute_bounds_for_heatmaps():
    try:
        bnds = []
        zg = st.session_state.get("zones_gdf")
        if zg is not None and not zg.empty:
            tb = zg.total_bounds
            bnds.append([[tb[1],tb[0]],[tb[3],tb[2]]])

        ydf = st.session_state.get("yield_df", pd.DataFrame())
        if isinstance(ydf, pd.DataFrame) and not ydf.empty:
            lat_col = find_col(ydf, ["latitude"])
            lon_col = find_col(ydf, ["longitude"])
            if lat_col and lon_col:
                lat = pd.to_numeric(ydf[lat_col], errors="coerce")
                lon = pd.to_numeric(ydf[lon_col], errors="coerce")
                if lat.notna().any() and lon.notna().any():
                    bnds.append([[lat.min(), lon.min()], [lat.max(), lon.max()]])
        if bnds:
            s = min(b[0][0] for b in bnds); w = min(b[0][1] for b in bnds)
            n = max(b[1][0] for b in bnds); e = max(b[1][1] for b in bnds)
            return s,w,n,e
    except Exception:
        pass
    return 25.0,-125.0,49.0,-66.0

def safe_fit_bounds(m, bounds):
    try:
        s,w,n,e = bounds
        m.fit_bounds([[s,w],[n,e]])
    except Exception:
        pass

# RX overlays (optional) – use seed_gdf + fert_gdfs if geometries exist
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

def infer_unit(gdf, rate_col, product_col):
    for cand in ["unit","units","uom","rate_uom","rateunit","rate_unit"]:
        if cand in gdf.columns:
            vals = gdf[cand].dropna().astype(str).str.strip()
            if not vals.empty and vals.iloc[0] != "":
                return vals.iloc[0]
    rc = str(rate_col or "").lower()
    if any(k in rc for k in ["gpa","gal","uan"]): return "gal/ac"
    if any(k in rc for k in ["lb","lbs","dry","nh3","ammonia"]): return "lb/ac"
    if "kg" in rc: return "kg/ha"
    if any(k in rc for k in ["seed","pop","plant","ksds","kseed","kseeds"]):
        try:
            med = pd.to_numeric(gdf[rate_col], errors="coerce").median()
            if 10 <= float(med) <= 90: return "k seeds/ac"
        except Exception: pass
        return "seeds/ac"
    prod_val = ""
    if product_col and product_col in gdf.columns:
        try: prod_val = str(gdf[product_col].dropna().astype(str).iloc[0]).lower()
        except Exception: prod_val = ""
    if "uan" in prod_val or "10-34-0" in prod_val: return "gal/ac"
    return None

def add_gradient_legend(name, vmin, vmax, cmap, index):
    top_offset = 20 + index * 80
    stops = [f"{mpl_colors.rgb2hex(cmap(i/100.0)[:3])} {i}%" for i in range(0,101,10)]
    gradient_css = ", ".join(stops)
    html = f"""
    <div style="position:absolute; top:{top_offset}px; left:12px; z-index:9999;
                font-family:sans-serif; font-size:12px; color:white;
                background-color:rgba(0,0,0,.65); padding:6px 10px; border-radius:5px; width:190px;">
      <div style="font-weight:600; margin-bottom:4px;">{name}</div>
      <div style="height:14px; background:linear-gradient(90deg, {gradient_css});
                  border-radius:2px; margin-bottom:4px;"></div>
      <div style="display:flex; justify-content:space-between;">
        <span>{vmin:.1f}</span><span>{vmax:.1f}</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))

def add_prescription_overlay(gdf, name, cmap, index):
    if gdf is None or gdf.empty:
        return
    try:
        g = gdf.copy()
        product_col, rate_col = None, None
        for c in g.columns:
            cl = str(c).lower()
            if product_col is None and "product" in cl: product_col = c
            if rate_col is None and ("tgt" in cl or "rate" in cl): rate_col = c

        g["RateType"] = detect_rate_type(g)
        if rate_col and pd.to_numeric(g[rate_col], errors="coerce").notna().any():
            vals = pd.to_numeric(g[rate_col], errors="coerce").dropna()
            vmin, vmax = float(vals.min()), float(vals.max())
            if vmin == vmax: vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0

        unit = infer_unit(g, rate_col, product_col)
        rate_alias = f"Target Rate ({unit})" if unit else "Target Rate"
        legend_name = f"{name} — {rate_alias}"

        def style_fn(feat):
            val = feat["properties"].get(rate_col) if rate_col else None
            if val is None or pd.isna(val):
                fill = "#808080"
            else:
                try:
                    norm = (float(val) - vmin) / (vmax - vmin) if vmax>vmin else 0.5
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
            g, name=name, style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases)
        ).add_to(m)

        add_gradient_legend(legend_name, vmin, vmax, cmap, index)
    except Exception as e:
        st.warning(f"⚠️ Skipping overlay {name}: {e}")

legend_index = 0
seed_gdf = st.session_state.get("seed_gdf")
if seed_gdf is not None and not seed_gdf.empty:
    add_prescription_overlay(seed_gdf, "Seed RX", plt.cm.Greens, legend_index); legend_index += 1

for k, gdfk in st.session_state.get("fert_gdfs", {}).items():
    if gdfk is not None and not gdfk.empty:
        add_prescription_overlay(gdfk, f"Fertilizer RX: {k}", plt.cm.Blues, legend_index); legend_index += 1

# Heatmaps
def add_heatmap_overlay(df, values, name, cmap, show_default, bounds):
    try:
        if df is None or df.empty:
            return None, None
        s,w,n,e = bounds
        vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        if vals.empty: return None, None

        lat_col = find_col(df, ["latitude"])
        lon_col = find_col(df, ["longitude"])
        if not lat_col or not lon_col: return None, None

        mask = df[[lat_col, lon_col]].applymap(np.isfinite).all(axis=1)
        if mask.sum() < 3: return None, None

        pts_lon = df.loc[mask, lon_col].astype(float).values
        pts_lat = df.loc[mask, lat_col].astype(float).values
        vals_ok = vals.loc[mask].astype(float).values

        npx = 200
        lon_lin = np.linspace(w, e, npx)
        lat_lin = np.linspace(s, n, npx)
        lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)

        grid_lin = griddata((pts_lon, pts_lat), vals_ok, (lon_grid, lat_grid), method="linear")
        grid_nn  = griddata((pts_lon, pts_lat), vals_ok, (lon_grid, lat_grid), method="nearest")
        grid = np.where(np.isnan(grid_lin), grid_nn, grid_lin)
        if grid is None or np.all(np.isnan(grid)): return None, None

        vmin = float(np.nanpercentile(vals_ok, 5)) if len(vals_ok)>0 else 0.0
        vmax = float(np.nanpercentile(vals_ok,95)) if len(vals_ok)>0 else 1.0
        if vmin == vmax: vmax = vmin + 1.0

        rgba = cmap((grid - vmin) / (vmax - vmin))
        rgba = np.flipud(rgba)
        rgba = (rgba * 255).astype(np.uint8)

        folium.raster_layers.ImageOverlay(
            image=rgba, bounds=[[s,w],[n,e]],
            opacity=0.5, name=name, overlay=True, show=show_default
        ).add_to(m)
        return vmin, vmax
    except Exception as e:
        st.warning(f"⚠️ Skipping heatmap {name}: {e}")
        return None, None

bounds = compute_bounds_for_heatmaps()

# Build yield df for heatmaps; or use target-yield fallback point
ydf = st.session_state.get("yield_df", pd.DataFrame())
if isinstance(ydf, pd.DataFrame) and not ydf.empty and "Yield" in ydf.columns:
    df_for_maps = ydf.copy()
else:
    # fallback to target yield at map center
    s,w,n,e = bounds
    lat_c = (s+n)/2.0; lon_c = (w+e)/2.0
    tgt = float(st.session_state.get("target_yield", 200.0))
    df_for_maps = pd.DataFrame({"Yield":[tgt], "Latitude":[lat_c], "Longitude":[lon_c]})

# Revenue & Profit layers (variable/fixed)
try:
    df_for_maps = df_for_maps.copy()
    latc = find_col(df_for_maps, ["latitude"]) or "Latitude"
    lonc = find_col(df_for_maps, ["longitude"]) or "Longitude"
    if latc not in df_for_maps.columns: df_for_maps[latc] = df_for_maps["Latitude"]
    if lonc not in df_for_maps.columns: df_for_maps[lonc] = df_for_maps["Longitude"]

    df_for_maps["Yield"] = pd.to_numeric(df_for_maps["Yield"], errors="coerce").fillna(0.0)
    df_for_maps["Revenue_per_acre"] = df_for_maps["Yield"] * float(sell_price)

    fert_var = 0.0
    seed_var = 0.0
    for d in st.session_state.get("fert_layers_store", {}).values():
        if not d.empty: fert_var += float(d["CostPerAcre"].sum())
    for d in st.session_state.get("seed_layers_store", {}).values():
        if not d.empty: seed_var += float(d["CostPerAcre"].sum())

    df_for_maps["NetProfit_Variable"] = df_for_maps["Revenue_per_acre"] - (base_expenses_per_acre + fert_var + seed_var)

    fixed_costs = 0.0
    fx = st.session_state.get("fixed_products", pd.DataFrame())
    if isinstance(fx, pd.DataFrame) and not fx.empty:
        if "$/ac" in fx.columns:
            fixed_costs = float(pd.to_numeric(fx["$/ac"], errors="coerce").fillna(0.0).sum())
        else:
            # compute if columns exist
            rcol = find_col(fx, ["rate"]); pcol = find_col(fx, ["costperunit"])
            if rcol and pcol:
                fixed_costs = float((pd.to_numeric(fx[rcol], errors="coerce").fillna(0.0) *
                                     pd.to_numeric(fx[pcol], errors="coerce").fillna(0.0)).sum())
    df_for_maps["NetProfit_Fixed"] = df_for_maps["Revenue_per_acre"] - (base_expenses_per_acre + fixed_costs)
except Exception:
    st.warning("⚠️ Could not compute profit metrics for heatmaps; using zeros.")
    df_for_maps["Revenue_per_acre"] = 0.0
    df_for_maps["NetProfit_Variable"] = 0.0
    df_for_maps["NetProfit_Fixed"] = 0.0

legend_i = 0
ymin,ymax = add_heatmap_overlay(df_for_maps, df_for_maps["Yield"], "Yield (bu/ac)", plt.cm.RdYlGn, False, bounds)
if ymin is not None:
    add_gradient_legend("Yield (bu/ac)", ymin, ymax, plt.cm.RdYlGn, legend_i); legend_i += 1

vmin,vmax = add_heatmap_overlay(df_for_maps, df_for_maps["NetProfit_Variable"], "Variable Rate Profit ($/ac)", plt.cm.RdYlGn, True, bounds)
if vmin is not None:
    add_gradient_legend("Variable Rate Profit ($/ac)", vmin, vmax, plt.cm.RdYlGn, legend_i); legend_i += 1

fmin,fmax = add_heatmap_overlay(df_for_maps, df_for_maps["NetProfit_Fixed"], "Fixed Rate Profit ($/ac)", plt.cm.RdYlGn, False, bounds)
if fmin is not None:
    add_gradient_legend("Fixed Rate Profit ($/ac)", fmin, fmax, plt.cm.RdYlGn, legend_i); legend_i += 1

# Fit map and show
safe_fit_bounds(m, bounds)
try:
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
except Exception:
    pass
st_folium(m, use_container_width=True, height=600)

# ---------------------------------------------------------
# Profit Summary — compact tables, no internal scroll
# ---------------------------------------------------------
st.header("Profit Summary")

# Breakeven Budget Tool (Corn vs Soy)
corn_yld = float(st.session_state.get("corn_yield", 200.0))
corn_px  = float(st.session_state.get("corn_price", 5.0))
bean_yld = float(st.session_state.get("bean_yield", 60.0))
bean_px  = float(st.session_state.get("bean_price", 12.0))

corn_rev = corn_yld * corn_px
bean_rev = bean_yld * bean_px
corn_budget = corn_rev - base_expenses_per_acre
bean_budget = bean_rev - base_expenses_per_acre

breakeven_df = pd.DataFrame({
    "Crop":["Corn","Soybeans"],
    "Yield Goal (bu/ac)":[corn_yld, bean_yld],
    "Sell Price ($/bu)":[corn_px, bean_px],
    "Revenue ($/ac)":[corn_rev, bean_rev],
    "Fixed Inputs ($/ac)":[base_expenses_per_acre, base_expenses_per_acre],
    "Breakeven Budget ($/ac)":[corn_budget, bean_budget]
})

def _hl_budget(v):
    if pd.isna(v): return ""
    if v>0: return "color:green;font-weight:700;"
    if v<0: return "color:red;font-weight:700;"
    return "font-weight:700;"

st.dataframe(
    breakeven_df.style.applymap(_hl_budget, subset=["Breakeven Budget ($/ac)"]).format({
        "Yield Goal (bu/ac)":"{:,.1f}",
        "Sell Price ($/bu)":"${:,.2f}",
        "Revenue ($/ac)":"${:,.2f}",
        "Fixed Inputs ($/ac)":"${:,.2f}",
        "Breakeven Budget ($/ac)":"${:,.2f}",
    }),
    use_container_width=True, hide_index=True, height=df_px_height(2)
)

# Fixed Input Costs table (summary) — no scroll
fixed_df = pd.DataFrame(list(expenses.items()), columns=["Expense","$/ac"])
total_fixed = pd.DataFrame([{"Expense":"Total Fixed Costs","$/ac":fixed_df["$/ac"].sum()}])
fixed_df = pd.concat([fixed_df, total_fixed], ignore_index=True)

styled_fixed = fixed_df.style.format({"$/ac":"${:,.2f}"}).apply(
    lambda x: ["font-weight:700;" if v=="Total Fixed Costs" else "" for v in x],
    subset=["Expense"]
).apply(
    lambda x: ["font-weight:700;" if i==len(fixed_df)-1 else "" for i in range(len(x))],
    subset=["$/ac"]
)

st.dataframe(styled_fixed, use_container_width=True, hide_index=True, height=df_px_height(len(fixed_df)))
