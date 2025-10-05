# =========================================================
# Farm Profit Mapping Tool V4 — COMPACT + BULLET-PROOF
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from branca.element import MacroElement, Template
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from scipy.interpolate import griddata
import geopandas as gpd
import zipfile, tempfile, os, shutil

# ---------------- Setup ----------------
st.set_page_config(page_title="Farm ROI Tool V4", layout="wide")
st.title("Farm Profit Mapping Tool V4")

# ---------------- Compact CSS ----------------
st.markdown(
    """
    <style>
      /* super tight gutters + vertical rhythm */
      div[data-testid="column"]{padding-left:.18rem!important;padding-right:.18rem!important}
      section[data-testid="stVerticalBlock"]>div{padding-top:.18rem!important;padding-bottom:.18rem!important}

      h1{margin:.25rem 0 .25rem 0!important;font-size:1.15rem!important}
      h2,h3{margin:.25rem 0 .2rem 0!important;font-size:1rem!important}

      /* tiny expanders */
      div[data-testid="stExpander"] details summary{padding:.22rem .4rem!important;font-size:.82rem!important}
      div[data-testid="stExpander"] details > div{padding:.22rem .4rem!important}

      /* compact number inputs */
      div[data-testid="stNumberInput"] label{font-size:.72rem!important;margin-bottom:.05rem!important}
      div[data-testid="stNumberInput"] div[role="spinbutton"]{min-height:22px!important;height:22px!important;padding:0 4px!important;font-size:.78rem!important}
      div[data-testid="stNumberInput"] button{padding:0!important;min-width:16px!important}

      /* global narrow boxes (works for uploaders row + fixed inputs row) */
      .narrow-nums div[data-testid="stNumberInput"]{width:96px!important;max-width:96px!important}

      /* dataframes/editors */
      div[data-testid="stDataFrame"] table, div[data-testid="stDataEditor"] table{font-size:.8rem!important}
      div[data-testid="stDataFrame"] th,div[data-testid="stDataFrame"] td,
      div[data-testid="stDataEditor"] th,div[data-testid="stDataEditor"] td{padding:2px 6px!important;line-height:1.05rem!important}

      /* tiny captions */
      div[data-testid="stCaptionContainer"]{margin:.1rem 0!important;font-size:.72rem!important}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Small helpers ----------------
def df_px_height(nrows:int, row_h:int=26, header:int=30, pad:int=2)->int:
    """Exact pixel height for a table/editor so it will not scroll internally."""
    return int(header + nrows*row_h + pad)

def _mini_num(label:str, key:str, default:float=0.0, step:float=0.1):
    st.caption(label)
    return st.number_input(key, min_value=0.0, value=float(st.session_state.get(key, default)), step=step, label_visibility="collapsed")

def load_vector_file(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith((".geojson",".json")):
            gdf = gpd.read_file(uploaded_file)
        elif uploaded_file.name.lower().endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmp:
                zpath = os.path.join(tmp,"in.zip")
                with open(zpath,"wb") as f: f.write(uploaded_file.getbuffer())
                with zipfile.ZipFile(zpath,"r") as zf: zf.extractall(tmp)
                shp=None
                for nm in os.listdir(tmp):
                    if nm.lower().endswith(".shp"): shp=os.path.join(tmp,nm); break
                if not shp: return None
                gdf = gpd.read_file(shp)
        elif uploaded_file.name.lower().endswith(".shp"):
            with tempfile.TemporaryDirectory() as tmp:
                shp=os.path.join(tmp, uploaded_file.name)
                with open(shp,"wb") as f: f.write(uploaded_file.getbuffer())
                gdf = gpd.read_file(shp)
        else:
            return None
        if gdf is None or gdf.empty: return None
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        return gdf.to_crs(epsg=4326)
    except Exception:
        return None

def process_prescription(file, prescrip_type="fertilizer"):
    if file is None: return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])
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
            if c in df.columns: df.rename(columns={c:"product"}, inplace=True); break
        else: df["product"] = prescrip_type.capitalize()
    if "acres" not in df.columns: df["acres"] = 0.0

    with st.expander(f"⚙️ {prescrip_type.capitalize()} Map Options — {file.name}", expanded=False):
        override = st.number_input("Override Acres Per Polygon", min_value=0.0, value=0.0, step=0.1, key=f"{prescrip_type}_{file.name}_override")
        if override>0: df["acres"] = override

    if "costtotal" not in df.columns:
        if {"price_per_unit","units"}.issubset(df.columns): df["costtotal"] = df["price_per_unit"] * df["units"]
        elif {"rate","price"}.issubset(df.columns):        df["costtotal"] = df["rate"] * df["price"]
        else:                                              df["costtotal"] = 0

    if df.empty: return pd.DataFrame(columns=["product","Acres","CostTotal","CostPerAcre"])
    grouped = df.groupby("product", as_index=False).agg(Acres=("acres","sum"), CostTotal=("costtotal","sum"))
    grouped["CostPerAcre"] = grouped.apply(lambda r: (r["CostTotal"]/r["Acres"]) if r["Acres"]>0 else 0, axis=1)
    return grouped

# ---------------- Session defaults (prevents None crashes) ----------------
st.session_state.setdefault("yield_df", pd.DataFrame())            # always a DataFrame
st.session_state.setdefault("fert_layers_store", {})               # dict of grouped tables
st.session_state.setdefault("seed_layers_store", {})
st.session_state.setdefault("fert_products", pd.DataFrame())
st.session_state.setdefault("seed_products", pd.DataFrame())
st.session_state.setdefault("fixed_products", pd.DataFrame())

# =========================================================
# 1) UPLOADERS — ULTRA COMPACT SINGLE ROW (4 cols)
# =========================================================
st.subheader("Upload Maps")
u1,u2,u3,u4 = st.columns(4)

with u1:
    zone_file = st.file_uploader("Zone Map", type=["geojson","json","zip","shp"], key="zone_file", accept_multiple_files=False)
    st.caption("GeoJSON / JSON / zipped SHP")
    if zone_file:
        gdf = load_vector_file(zone_file)
        if gdf is not None and not gdf.empty:
            # detect Zone col or create
            zcol = next((c for c in ["Zone","zone","ZONE","Name","name"] if c in gdf.columns), None)
            if zcol is None:
                gdf["ZoneIndex"] = range(1,len(gdf)+1); zcol = "ZoneIndex"
            gdf["Zone"] = gdf[zcol]
            # acres (equal-area)
            ga = gdf.copy()
            if ga.crs is None: ga.set_crs(epsg=4326, inplace=True)
            if ga.crs.is_geographic: ga = ga.to_crs(epsg=5070)
            gdf["Calculated Acres"] = (ga.geometry.area * 0.000247105).astype(float)
            gdf["Override Acres"]   = gdf["Calculated Acres"].astype(float)
            st.session_state["zones_gdf"] = gdf
            st.caption(f"✅ {len(gdf)} polygons")

with u2:
    yield_files = st.file_uploader("Yield Map(s)", type=["csv","geojson","json","zip","shp"], key="yield_files", accept_multiple_files=True)
    st.caption("CSV / GeoJSON / JSON / zipped SHP")
    if yield_files:
        frames=[]; total=0
        for yf in yield_files:
            try:
                if yf.name.lower().endswith(".csv"): df = pd.read_csv(yf)
                else:
                    yg = load_vector_file(yf)
                    df = pd.DataFrame(yg.drop(columns="geometry", errors="ignore")) if yg is not None else pd.DataFrame()
                if not df.empty:
                    df.columns=[c.strip().lower().replace(" ","_") for c in df.columns]
                    ycols=[c for c in df.columns if any(k in c for k in ["yld_vol_dr","yld_mass_dr","yield_dry","dry_yield","yield","wet_yield"])]
                    if ycols: df.rename(columns={ycols[0]:"Yield"}, inplace=True)
                    else: df["Yield"]=0.0
                    frames.append(df); total+=len(df)
            except Exception as e:
                st.warning(f"{yf.name}: {e}")
        st.session_state["yield_df"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        st.caption(f"✅ {len(yield_files)} file(s), rows={total:,}")

with u3:
    fert_files = st.file_uploader("Fertilizer RX", type=["csv","geojson","json","zip","shp"], key="fert_files", accept_multiple_files=True)
    st.caption("CSV / GeoJSON / JSON / zipped SHP")
    st.session_state["fert_layers_store"] = {}
    if fert_files:
        for f in fert_files:
            grp = process_prescription(f,"fertilizer")
            if not grp.empty:
                key = os.path.splitext(f.name)[0].lower().replace(" ","_")
                st.session_state["fert_layers_store"][key] = grp
        st.caption(f"✅ {len(st.session_state['fert_layers_store'])} layer(s)")

with u4:
    seed_files = st.file_uploader("Seed RX", type=["csv","geojson","json","zip","shp"], key="seed_files", accept_multiple_files=True)
    st.caption("CSV / GeoJSON / JSON / zipped SHP")
    st.session_state["seed_layers_store"] = {}
    if seed_files:
        for f in seed_files:
            grp = process_prescription(f,"seed")
            if not grp.empty:
                key = os.path.splitext(f.name)[0].lower().replace(" ","_")
                st.session_state["seed_layers_store"][key] = grp
        st.caption(f"✅ {len(st.session_state['seed_layers_store'])} layer(s)")

# =========================================================
# 2) FIXED INPUTS — 12 TIGHT BOXES IN ONE ROW
# =========================================================
st.subheader("Fixed Inputs ($/ac)")
with st.container():
    st.markdown('<div class="narrow-nums">', unsafe_allow_html=True)
    r = st.columns(12, gap="small")
    with r[0]:  chemicals      = _mini_num("Chem","chem",0.0,1.0)
    with r[1]:  insurance      = _mini_num("Insur","ins",0.0,1.0)
    with r[2]:  insecticide    = _mini_num("Insect/Fung","insect",0.0,1.0)
    with r[3]:  fertilizer     = _mini_num("Fert Flat","fert",0.0,1.0)
    with r[4]:  seed           = _mini_num("Seed Flat","seed",0.0,1.0)
    with r[5]:  cash_rent      = _mini_num("Cash Rent","rent",0.0,1.0)
    with r[6]:  machinery      = _mini_num("Mach","mach",0.0,1.0)
    with r[7]:  labor          = _mini_num("Labor","labor",0.0,1.0)
    with r[8]:  living         = _mini_num("Living","col",0.0,1.0)
    with r[9]:  fuel           = _mini_num("Fuel","fuel",0.0,1.0)
    with r[10]: interest       = _mini_num("Interest","int",0.0,1.0)
    with r[11]: truck_fuel     = _mini_num("Truck Fuel","truck",0.0,1.0)
    st.markdown('</div>', unsafe_allow_html=True)

expenses = {
    "Chemicals": chemicals, "Insurance": insurance, "Insecticide/Fungicide": insecticide,
    "Fertilizer (Flat)": fertilizer, "Seed (Flat)": seed, "Cash Rent": cash_rent,
    "Machinery": machinery, "Labor": labor, "Cost of Living": living,
    "Extra Fuel": fuel, "Extra Interest": interest, "Truck Fuel": truck_fuel,
}
base_expenses_per_acre = float(sum(expenses.values()))

# =========================================================
# 3) CORN/SOY MINI PANEL (TOP-RIGHT ABOVE MAP)
# =========================================================
# Layout trick: left is spacer, right holds the panel; map sits below full-width
spacer, panel = st.columns([3, 2])

with panel:
    st.subheader("Compare Crop Profitability (Optional)")
    p = st.columns(4, gap="small")
    with p[0]: st.caption("Corn Yield (bu/ac)"); st.session_state["corn_yield"] = st.number_input("corn_y", 0.0, value=float(st.session_state.get("corn_yield",200.0)), step=1.0, label_visibility="collapsed")
    with p[1]: st.caption("Corn Price ($/bu)"); st.session_state["corn_price"] = st.number_input("corn_p", 0.0, value=float(st.session_state.get("corn_price",5.0)), step=0.1, label_visibility="collapsed")
    with p[2]: st.caption("Soy Yield (bu/ac)");  st.session_state["bean_yield"] = st.number_input("bean_y", 0.0, value=float(st.session_state.get("bean_yield",60.0)), step=1.0, label_visibility="collapsed")
    with p[3]: st.caption("Soy Price ($/bu)");  st.session_state["bean_price"] = st.number_input("bean_p", 0.0, value=float(st.session_state.get("bean_price",12.0)), step=0.1, label_visibility="collapsed")

    # tiny 2-row table, exact height -> no scroll
    prev = pd.DataFrame({
        "Crop":["Corn","Soybeans"],
        "Yield":[st.session_state["corn_yield"], st.session_state["bean_yield"]],
        "Price":[st.session_state["corn_price"], st.session_state["bean_price"]],
        "Revenue":[st.session_state["corn_yield"]*st.session_state["corn_price"],
                   st.session_state["bean_yield"]*st.session_state["bean_price"]],
        "Fixed":[base_expenses_per_acre, base_expenses_per_acre],
    })
    prev["Breakeven"] = prev["Revenue"] - prev["Fixed"]

    def _hl_breakeven(v):
        if pd.isna(v): return ""
        if v>0:  return "color:#22c55e;font-weight:700;"
        if v<0:  return "color:#ef4444;font-weight:700;"
        return "font-weight:700;"

    st.dataframe(
        prev.style.applymap(_hl_breakeven, subset=["Breakeven"]).format({
            "Yield":"{:.0f}", "Price":"${:.2f}",
            "Revenue":"${:,.0f}", "Fixed":"${:,.0f}", "Breakeven":"${:,.0f}"
        }),
        use_container_width=True,
        hide_index=True,
        height=df_px_height(2, row_h=24, header=32, pad=2)
    )

# =========================================================
# 4) MAP & OVERLAYS (bullet-proof)
# =========================================================
def make_base_map():
    try:
        m = folium.Map(location=[39.5,-98.35], zoom_start=5, min_zoom=2, tiles=None, scrollWheelZoom=False, prefer_canvas=True)
        try:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", overlay=False, control=False
            ).add_to(m)
        except Exception: pass
        try:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", overlay=True, control=False
            ).add_to(m)
        except Exception: pass

        template = Template("""
            {% macro script(this, kwargs) %}
            var map = {{this._parent.get_name()}};
            map.scrollWheelZoom.disable();
            map.on('click', function(){ map.scrollWheelZoom.enable(); });
            map.on('mouseout', function(){ map.scrollWheelZoom.disable(); });
            {% endmacro %}
        """)
        macro = MacroElement(); macro._template = template; m.get_root().add_child(macro)
        return m
    except Exception as e:
        st.error(f"❌ Failed to build base map: {e}")
        return folium.Map(location=[39.5,-98.35], zoom_start=4)

def compute_bounds_for_heatmaps():
    try:
        b=[]
        z = st.session_state.get("zones_gdf")
        if z is not None and not z.empty:
            tb=z.total_bounds; b.append([[tb[1],tb[0]],[tb[3],tb[2]]])
        y = st.session_state.get("yield_df")
        if y is not None and not y.empty and {"latitude","longitude"}.issubset({c.lower() for c in y.columns}):
            b.append([[y["Latitude"].min(),y["Longitude"].min()],[y["Latitude"].max(),y["Longitude"].max()]])
        if b:
            south=min(x[0][0] for x in b); west=min(x[0][1] for x in b)
            north=max(x[1][0] for x in b); east=max(x[1][1] for x in b)
            return south,west,north,east
    except Exception:
        pass
    return 25.0,-125.0,49.0,-66.0

def safe_fit_bounds(m,bounds):
    try: south,west,north,east=bounds; m.fit_bounds([[south,west],[north,east]])
    except Exception: pass

def add_zones_overlay(m):
    g = st.session_state.get("zones_gdf")
    if g is None or g.empty: return
    try:
        g = g.to_crs(epsg=4326)
        if "Zone" not in g.columns: g["Zone"]=range(1,len(g)+1)
        # palette
        pal=["#FF0000","#FF8C00","#FFFF00","#32CD32","#006400","#1E90FF","#8A2BE2","#FFC0CB","#A52A2A","#00CED1"]
        uniq=list(dict.fromkeys(sorted(g["Zone"].astype(str).unique())))
        c_map={z:pal[i%len(pal)] for i,z in enumerate(uniq)}
        folium.GeoJson(
            g, name="Zones",
            style_function=lambda f: {"fillColor":c_map.get(str(f["properties"].get("Zone","")),"#808080"),
                                      "color":"black","weight":1,"fillOpacity":0.08},
            tooltip=folium.GeoJsonTooltip(fields=[c for c in ["Zone","Calculated Acres","Override Acres"] if c in g.columns])
        ).add_to(m)
        # legend
        items="".join([f"<div style='display:flex;align-items:center;margin:2px 0;'><div style='background:{c_map[z]};width:14px;height:14px;margin-right:6px;'></div>{z}</div>" for z in uniq])
        legend=f"""
        <div style="position:absolute;bottom:20px;right:20px;z-index:9999;font-family:sans-serif;font-size:12px;color:white;background:rgba(0,0,0,.65);padding:6px 10px;border-radius:5px;width:160px;">
          <div style="font-weight:600;margin-bottom:4px;cursor:pointer;" onclick="var x=document.getElementById('z-legend');x.style.display=(x.style.display==='none')?'block':'none';">Zone Colors ▼</div>
          <div id="z-legend" style="display:block;">{items}</div>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend))
    except Exception as e:
        st.warning(f"⚠️ Zones overlay: {e}")

def add_gradient_legend(name,vmin,vmax,cmap,index):
    top=20+(index*80)
    stops=", ".join([f"{mpl_colors.rgb2hex(cmap(i/100)[:3])} {i}%" for i in range(0,101,10)])
    html=f"""
    <div style="position:absolute;top:{top}px;left:10px;z-index:9999;font-family:sans-serif;font-size:12px;color:white;background:rgba(0,0,0,.65);padding:6px 10px;border-radius:5px;width:180px;">
      <div style="font-weight:600;margin-bottom:4px;">{name}</div>
      <div style="height:14px;background:linear-gradient(90deg,{stops});border-radius:2px;margin-bottom:4px;"></div>
      <div style="display:flex;justify-content:space-between;"><span>{vmin:.1f}</span><span>{vmax:.1f}</span></div>
    </div>"""
    m.get_root().html.add_child(folium.Element(html))

def add_heatmap_overlay(df, series, name, cmap, show_default, bounds):
    try:
        if df is None or df.empty: return None,None
        south,west,north,east=bounds
        vals=pd.to_numeric(pd.Series(series), errors="coerce")
        mask = vals.notna() & df[["Latitude","Longitude"]].applymap(np.isfinite).all(axis=1)
        if mask.sum()<3: return None,None
        pts_lon=df.loc[mask,"Longitude"].astype(float).values
        pts_lat=df.loc[mask,"Latitude"].astype(float).values
        vals_ok=vals.loc[mask].astype(float).values

        n=200
        lon_lin=np.linspace(west,east,n); lat_lin=np.linspace(south,north,n)
        lon_grid,lat_grid=np.meshgrid(lon_lin,lat_lin)
        grid_lin=griddata((pts_lon,pts_lat), vals_ok, (lon_grid,lat_grid), method="linear")
        grid_nn =griddata((pts_lon,pts_lat), vals_ok, (lon_grid,lat_grid), method="nearest")
        grid=np.where(np.isnan(grid_lin), grid_nn, grid_lin)
        if grid is None or np.all(np.isnan(grid)): return None,None

        vmin=float(np.nanpercentile(vals_ok,5)); vmax=float(np.nanpercentile(vals_ok,95))
        if vmin==vmax: vmax=vmin+1.0
        rgba=cmap((grid-vmin)/(vmax-vmin)); rgba=np.flipud(rgba); rgba=(rgba*255).astype(np.uint8)

        folium.raster_layers.ImageOverlay(image=rgba,bounds=[[south,west],[north,east]],opacity=0.5,name=name,overlay=True,show=show_default).add_to(m)
        return vmin,vmax
    except Exception as e:
        st.warning(f"⚠️ Heatmap {name}: {e}")
        return None,None

# Build and show map
m = make_base_map()
add_zones_overlay(m)

# Prescription overlays (if you add seed_gdf/fert_gdfs later, this still won't crash)
st.session_state.setdefault("seed_gdf", gpd.GeoDataFrame())
st.session_state.setdefault("fert_gdfs", {})

legend_ix=0
if not st.session_state["seed_gdf"].empty:
    # optional seed layer support
    pass

for k, grp in st.session_state.get("fert_gdfs", {}).items():
    if grp is not None and not grp.empty:
        pass

# Yield/profit raster (uses corn price as sell price by default)
bounds = compute_bounds_for_heatmaps()
yld_df = st.session_state.get("yield_df")
sell_price = float(st.session_state.get("corn_price", 5.0))

if yld_df is None or yld_df.empty or not {"latitude","longitude","yield"}.issubset({c.lower() for c in yld_df.columns}):
    # no yield map -> single point at map center using corn assumptions
    lat_center=(bounds[0]+bounds[2])/2.0
    lon_center=(bounds[1]+bounds[3])/2.0
    target = float(st.session_state.get("corn_yield", 200.0))
    df_map = pd.DataFrame({"Yield":[target],"Latitude":[lat_center],"Longitude":[lon_center]})
else:
    df_map = yld_df.rename(columns={c:c.title() for c in yld_df.columns})  # unify case
    if "Yield" not in df_map.columns: df_map["Yield"]=0.0

try:
    df_map["Revenue_per_acre"] = df_map["Yield"].astype(float) * sell_price
    fert_var = float(st.session_state["fert_products"]["CostPerAcre"].sum()) if not st.session_state["fert_products"].empty else 0.0
    seed_var = float(st.session_state["seed_products"]["CostPerAcre"].sum()) if not st.session_state["seed_products"].empty else 0.0
    fixed_costs = 0.0
    if not st.session_state["fixed_products"].empty:
        fx = st.session_state["fixed_products"].copy()
        fx["$/ac"] = fx.apply(lambda r:(r.get("Rate",0) or 0)*(r.get("CostPerUnit",0) or 0), axis=1)
        fixed_costs = float(fx["$/ac"].sum())
    df_map["NetProfit_per_acre_variable"] = df_map["Revenue_per_acre"] - (base_expenses_per_acre + fert_var + seed_var)
    df_map["NetProfit_per_acre_fixed"]     = df_map["Revenue_per_acre"] - (base_expenses_per_acre + fixed_costs)
except Exception:
    df_map["Revenue_per_acre"]=0.0
    df_map["NetProfit_per_acre_variable"]=0.0
    df_map["NetProfit_per_acre_fixed"]=0.0

ymin,ymax = add_heatmap_overlay(df_map, df_map["Yield"], "Yield (bu/ac)", plt.cm.RdYlGn, False, bounds)
if ymin is not None: add_gradient_legend("Yield (bu/ac)", ymin, ymax, plt.cm.RdYlGn, legend_ix); legend_ix+=1

vmin,vmax = add_heatmap_overlay(df_map, df_map["NetProfit_per_acre_variable"], "Variable Rate Profit ($/ac)", plt.cm.RdYlGn, True, bounds)
if vmin is not None: add_gradient_legend("Variable Rate Profit ($/ac)", vmin, vmax, plt.cm.RdYlGn, legend_ix); legend_ix+=1

fmin,fmax = add_heatmap_overlay(df_map, df_map["NetProfit_per_acre_fixed"], "Fixed Rate Profit ($/ac)", plt.cm.RdYlGn, False, bounds)
if fmin is not None: add_gradient_legend("Fixed Rate Profit ($/ac)", fmin, fmax, plt.cm.RdYlGn, legend_ix); legend_ix+=1

# fit & show
safe_fit_bounds(m, compute_bounds_for_heatmaps())
try: folium.LayerControl(collapsed=False, position="topright").add_to(m)
except Exception: pass
st_folium(m, use_container_width=True, height=580)

# =========================================================
# 5) PROFIT SUMMARY (unchanged math, compact tables)
# =========================================================
st.subheader("Profit Summary")

corn_y=st.session_state.get("corn_yield",200.0); corn_p=st.session_state.get("corn_price",5.0)
bean_y=st.session_state.get("bean_yield",60.0);  bean_p=st.session_state.get("bean_price",12.0)

corn_rev=corn_y*corn_p; bean_rev=bean_y*bean_p
be_df = pd.DataFrame({
    "Crop":["Corn","Soybeans"],
    "Yield Goal (bu/ac)":[corn_y,bean_y],
    "Sell Price ($/bu)":[corn_p,bean_p],
    "Revenue ($/ac)":[corn_rev,bean_rev],
    "Fixed Inputs ($/ac)":[base_expenses_per_acre,base_expenses_per_acre],
    "Breakeven Budget ($/ac)":[corn_rev-base_expenses_per_acre, bean_rev-base_expenses_per_acre]
})
def _h(v):
    if isinstance(v,(int,float)):
        if v>0: return "color:green;font-weight:700;"
        if v<0: return "color:red;font-weight:700;"
    return "font-weight:700;"
st.dataframe(
    be_df.style.applymap(_h, subset=["Breakeven Budget ($/ac)"]).format({
        "Yield Goal (bu/ac)":"{:,.1f}", "Sell Price ($/bu)":"${:,.2f}", "Revenue ($/ac)":"${:,.2f}",
        "Fixed Inputs ($/ac)":"${:,.2f}", "Breakeven Budget ($/ac)":"${:,.2f}"
    }),
    use_container_width=True, hide_index=True, height=df_px_height(2, row_h=26, header=32)
)

st.subheader("Fixed Input Costs")
fx_df = pd.DataFrame(list(expenses.items()), columns=["Expense","$/ac"])
fx_df = pd.concat([fx_df, pd.DataFrame([{"Expense":"Total Fixed Costs","$/ac":fx_df["$/ac"].sum()}])], ignore_index=True)
st.dataframe(
    fx_df.style.format({"$/ac":"${:,.2f}"}).apply(
        lambda s:["font-weight:700;" if v=="Total Fixed Costs" else "" for v in s], subset=["Expense"]
    ).apply(
        lambda s:["font-weight:700;" if i==len(s)-1 else "" for i in range(len(s))], subset=["$/ac"]
    ),
    use_container_width=True, hide_index=True, height=df_px_height(len(fx_df), row_h=26, header=32)
)
