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

# ===========================
# COMPACT THEME + LAYOUT
# ===========================
import streamlit as st
import pandas as pd
import geopandas as gpd
import zipfile, os, tempfile
from typing import Optional, Tuple

def apply_compact_theme():
    st.set_page_config(page_title="Farm ROI Tool V4", layout="wide")
    st.title("Farm Profit Mapping Tool V4")

    # Pull page up + compact everything (inputs, tables, uploaders)
    st.markdown("""
    <style>
      .block-container{ padding-top:.28rem !important; }

      /* tighter column gutters */
      div[data-testid="column"]{ padding-left:.22rem !important; padding-right:.22rem !important; }

      /* reduce block spacing */
      section[data-testid="stVerticalBlock"] > div{
        padding-top:.18rem !important; padding-bottom:.18rem !important;
      }

      /* compact headers */
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
      /* ensure 12 fixed-input boxes fit one line on most screens */
      .fixed-12 div[data-testid="stNumberInput"]{ width:112px !important; max-width:112px !important; }

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
    """, unsafe_allow_html=True)

def _bootstrap_defaults():
    for k in ["chem","ins","insect","fert","seed","rent","mach","labor","col","fuel","int","truck"]:
        st.session_state.setdefault(k, 0.0)
    st.session_state.setdefault("corn_yield", 200.0)
    st.session_state.setdefault("corn_price", 5.0)
    st.session_state.setdefault("bean_yield", 60.0)
    st.session_state.setdefault("bean_price", 12.0)
    st.session_state.setdefault("sell_price", st.session_state["corn_price"])
    st.session_state.setdefault("target_yield", 200.0)

_bootstrap_defaults()

# ===========================
# HELPERS
# ===========================
def df_px_height(nrows: int, row_h: int = 26, header: int = 30, pad: int = 2) -> int:
    """Exact pixel height so Streamlit tables/editors don't scroll internally."""
    n = max(1, int(nrows))
    return int(header + n*row_h + pad)

def _find_col(df: pd.DataFrame, names) -> Optional[str]:
    """Return the first matching column (case-insensitive), else None."""
    lower_map = {c.lower(): c for c in df.columns}
    for n in names:
        if n in lower_map: return lower_map[n]
    return None

def load_vector_file(uploaded_file) -> Optional[gpd.GeoDataFrame]:
    """Read .geojson/.json/.zip(SHP)/.shp into EPSG:4326 GeoDataFrame."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith((".geojson", ".json")):
            gdf = gpd.read_file(uploaded_file)
        elif name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zpath = os.path.join(tmpdir, "in.zip")
                with open(zpath, "wb") as f: f.write(uploaded_file.getbuffer())
                with zipfile.ZipFile(zpath, "r") as zf: zf.extractall(tmpdir)
                shp = next((os.path.join(tmpdir, fn) for fn in os.listdir(tmpdir) if fn.lower().endswith(".shp")), None)
                if not shp: return None
                gdf = gpd.read_file(shp)
        elif name.endswith(".shp"):
            with tempfile.TemporaryDirectory() as tmpdir:
                shp = os.path.join(tmpdir, uploaded_file.name)
                with open(shp, "wb") as f: f.write(uploaded_file.getbuffer())
                gdf = gpd.read_file(shp)
        else:
            return None

        if gdf is None or gdf.empty: return None
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        return gdf.to_crs(epsg=4326)
    except Exception:
        return None
        
def find_col(df: pd.DataFrame, candidates) -> str | None:
    """
    Return the first column in df whose lowercased name matches any of the
    lowercased strings in `candidates`. If nothing matches, return None.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def process_prescription(file, prescrip_type="fertilizer") -> Tuple[pd.DataFrame, Optional[gpd.GeoDataFrame]]:
    """Normalize seed/fert files → grouped cost table + original gdf for overlays (if vector)."""
    if file is None:
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"]), None
    try:
        name = file.name.lower()
        gdf_orig = None
        if name.endswith((".geojson",".json",".zip",".shp")):
            gdf = load_vector_file(file)
            if gdf is None or gdf.empty:
                return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"]), None
            gdf_orig = gdf.copy()
            gdf = gdf.copy()
            gdf["Longitude"] = gdf.geometry.representative_point().x
            gdf["Latitude"]  = gdf.geometry.representative_point().y
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        else:
            df = pd.read_csv(file)

        df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
        if "product" not in df.columns:
            alias = _find_col(df, ["variety","hybrid","type","name","material"])
            df["product"] = df[alias] if alias else prescrip_type.capitalize()
        if "acres" not in df.columns: df["acres"] = 0.0
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
            Acres=("acres","sum"), CostTotal=("costtotal","sum")
        )
        grouped["CostPerAcre"] = grouped.apply(
            lambda r: r["CostTotal"]/r["Acres"] if r["Acres"]>0 else 0, axis=1
        )
        return grouped, gdf_orig
    except Exception:
        return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"]), None


# ===========================
# UPLOADERS ROW (ONE ROW)
# ===========================
def render_uploaders():
    st.subheader("Upload Maps")
    u1,u2,u3,u4 = st.columns(4)

    # ---- Zone ----
    with u1:
        st.caption("Zone Map · GeoJSON/JSON/ZIP(SHP)")
        zf = st.file_uploader("Zone", type=["geojson","json","zip"], key="up_zone", accept_multiple_files=False)
        if zf:
            gdf = load_vector_file(zf)
            if gdf is not None and not gdf.empty:
                # zone id
                cand = next((c for c in ["Zone","zone","ZONE","Name","name"] if c in gdf.columns), None)
                if not cand:
                    gdf["ZoneIndex"] = range(1, len(gdf)+1); cand="ZoneIndex"
                gdf["Zone"] = gdf[cand]

                # acres (equal-area)
                g2 = gdf.copy()
                if g2.crs is None: g2.set_crs(epsg=4326, inplace=True)
                if g2.crs.is_geographic: g2 = g2.to_crs(epsg=5070)
                gdf["Calculated Acres"] = (g2.geometry.area * 0.000247105).astype(float)
                gdf["Override Acres"]   = gdf["Calculated Acres"].astype(float)

                disp = gdf[["Zone","Calculated Acres","Override Acres"]]
                ed = st.data_editor(
                    disp, hide_index=True, num_rows="fixed", use_container_width=True,
                    column_config={
                        "Zone": st.column_config.TextColumn(disabled=True),
                        "Calculated Acres": st.column_config.NumberColumn(format="%.2f", disabled=True),
                        "Override Acres": st.column_config.NumberColumn(format="%.2f"),
                    },
                    height=df_px_height(len(disp))
                )
                ed["Override Acres"] = pd.to_numeric(ed["Override Acres"], errors="coerce") \
                                          .fillna(ed["Calculated Acres"])
                gdf["Override Acres"] = ed["Override Acres"].astype(float).values
                st.caption(f"✅ Zones: {len(gdf)} · Calc: {gdf['Calculated Acres'].sum():,.2f} ac · Override: {gdf['Override Acres'].sum():,.2f} ac")
                st.session_state["zones_gdf"] = gdf
            else:
                st.caption("❌ Could not read zone file.")
        else:
            st.caption("No zone file uploaded.")

    # ---- Yield ----
    with u2:
        st.caption("Yield Map(s) · CSV/GeoJSON/JSON/ZIP")
        yfs = st.file_uploader("Yield", type=["csv","geojson","json","zip"], key="up_yield", accept_multiple_files=True)
        st.session_state["yield_df"] = pd.DataFrame()
        if yfs:
            frames, summ = [], []
            for f in yfs:
                try:
                    nm = f.name.lower()
                    if nm.endswith(".csv"):
                        df = pd.read_csv(f)
                    else:
                        yg = load_vector_file(f)
                        df = pd.DataFrame(yg.drop(columns="geometry", errors="ignore")) if yg is not None else pd.DataFrame()
                    if not df.empty:
                        df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
                        ycol = _find_col(df, ["yield","yld_vol_dr","yld_mass_dr","yield_dry","dry_yield","wet_yield"])
                        if ycol and ycol != "Yield": df.rename(columns={ycol:"Yield"}, inplace=True)
                        elif not ycol: df["Yield"] = 0.0
                        frames.append(df)
                        summ.append({"File": f.name, "Rows": len(df)})
                except Exception as e:
                    summ.append({"File": f.name, "Rows": 0})
            st.session_state["yield_df"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if summ:
                st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True,
                             height=df_px_height(len(summ)))
        else:
            st.caption("No yield files uploaded.")

    # ---- Fert ----
    with u3:
        st.caption("Fertilizer RX · CSV/GeoJSON/JSON/ZIP")
        ffs = st.file_uploader("Fert", type=["csv","geojson","json","zip"], key="up_fert", accept_multiple_files=True)
        st.session_state["fert_layers_store"] = {}
        st.session_state["fert_gdfs"] = {}
        if ffs:
            summ=[]
            for f in ffs:
                grp, gdf = process_prescription(f, "fertilizer")
                if not grp.empty:
                    key = os.path.splitext(f.name)[0].lower().replace(" ","_")
                    st.session_state["fert_layers_store"][key] = grp
                    if gdf is not None and not gdf.empty: st.session_state["fert_gdfs"][key] = gdf
                    summ.append({"File": f.name, "Products": len(grp)})
            if summ:
                st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True,
                             height=df_px_height(len(summ)))
        else:
            st.caption("No fertilizer files uploaded.")

    # ---- Seed ----
    with u4:
        st.caption("Seed RX · CSV/GeoJSON/JSON/ZIP")
        sfs = st.file_uploader("Seed", type=["csv","geojson","json","zip"], key="up_seed", accept_multiple_files=True)
        st.session_state["seed_layers_store"] = {}
        st.session_state["seed_gdf"] = None
        if sfs:
            summ=[]; last_gdf=None
            for f in sfs:
                grp, gdf = process_prescription(f, "seed")
                if not grp.empty:
                    key = os.path.splitext(f.name)[0].lower().replace(" ","_")
                    st.session_state["seed_layers_store"][key] = grp
                    if gdf is not None and not gdf.empty: last_gdf = gdf
                    summ.append({"File": f.name, "Products": len(grp)})
            if last_gdf is not None and not last_gdf.empty:
                st.session_state["seed_gdf"] = last_gdf
            if summ:
                st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True,
                             height=df_px_height(len(summ)))
        else:
            st.caption("No seed files uploaded.")


# ===========================
# FIXED INPUTS + CORN/SOY STRIP
# ===========================
def render_fixed_inputs_and_strip():
    st.subheader("Fixed Inputs ($/ac) & Compare Strip")

    # ---- 12 fixed inputs on one line ----
    with st.container():
        cols = st.columns(12, gap="small")
        container = st.container()
    with container:
        st.markdown('<div class="fixed-12">', unsafe_allow_html=True)
        labels = [
            ("Chem", "chem"), ("Insur", "ins"), ("Insect/Fung", "insect"), ("Fert Flat", "fert"),
            ("Seed Flat", "seed"), ("Cash Rent", "rent"), ("Mach", "mach"), ("Labor", "labor"),
            ("Living", "col"), ("Fuel", "fuel"), ("Interest", "int"), ("Truck Fuel", "truck"),
        ]
        vals = {}
        for (lab, key), c in zip(labels, cols):
            with c:
                st.caption(lab)
                vals[key] = st.number_input(f"{key}_num", min_value=0.0, value=float(st.session_state.get(key, 0.0)),
                                            step=1.0, label_visibility="collapsed")
                st.session_state[key] = vals[key]
        st.markdown('</div>', unsafe_allow_html=True)

    expenses = {
        "Chemicals": vals["chem"], "Insurance": vals["ins"], "Insecticide/Fungicide": vals["insect"],
        "Fertilizer (Flat)": vals["fert"], "Seed (Flat)": vals["seed"], "Cash Rent": vals["rent"],
        "Machinery": vals["mach"], "Labor": vals["labor"], "Cost of Living": vals["col"],
        "Extra Fuel": vals["fuel"], "Extra Interest": vals["int"], "Truck Fuel": vals["truck"],
    }
    base_expenses_per_acre = float(sum(expenses.values()))
    st.session_state["base_expenses_per_acre"] = base_expenses_per_acre

    # ---- Corn/Soy inputs in one line + NO-SCROLL static table ----
    c1,c2,c3,c4 = st.columns(4, gap="small")
    with c1:
        st.caption("Corn Yield (bu/ac)")
        st.session_state["corn_yield"] = st.number_input("corn_yld", min_value=0.0, value=float(st.session_state.get("corn_yield", 200.0)),
                                                         step=1.0, label_visibility="collapsed")
    with c2:
        st.caption("Corn Price ($/bu)")
        st.session_state["corn_price"] = st.number_input("corn_px", min_value=0.0, value=float(st.session_state.get("corn_price", 5.0)),
                                                         step=0.1, label_visibility="collapsed")
    with c3:
        st.caption("Soy Yield (bu/ac)")
        st.session_state["bean_yield"] = st.number_input("bean_yld", min_value=0.0, value=float(st.session_state.get("bean_yield", 60.0)),
                                                         step=1.0, label_visibility="collapsed")
    with c4:
        st.caption("Soy Price ($/bu)")
        st.session_state["bean_price"] = st.number_input("bean_px", min_value=0.0, value=float(st.session_state.get("bean_price", 12.0)),
                                                         step=0.1, label_visibility="collapsed")

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

    # format → static table (no scroll)
    pv = prev_df.copy()
    pv["Yield"]     = pv["Yield"].map(lambda v: f"{v:.0f}")
    pv["Price"]     = pv["Price"].map(lambda v: f"${v:,.2f}")
    pv["Revenue"]   = pv["Revenue"].map(lambda v: f"${v:,.0f}")
    pv["Fixed"]     = pv["Fixed"].map(lambda v: f"${v:,.0f}")
    pv["Breakeven"] = pv["Breakeven"].map(lambda v: f"${v:,.0f}")

    st.table(pv)  # no internal scroll, hugs the content

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
            
    sell_price = float(st.session_state.get("sell_price", st.session_state.get("corn_price", 5.0)))

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

# SAFETY: compute fixed inputs total even if inputs were never touched.
_fixed_keys = ["chem","ins","insect","fert","seed","rent","mach","labor","col","fuel","int","truck"]
base_expenses_per_acre = float(sum(
    float(st.session_state.get(k, 0.0) or 0.0) for k in _fixed_keys
))
# Keep it in session so other sections (heatmaps/summary) can reuse it
st.session_state["base_expenses_per_acre"] = base_expenses_per_acre

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
