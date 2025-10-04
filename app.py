# =========================================================
# Farm Profit Mapping Tool V4 (Ultra-Compact Edition)
# =========================================================
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import geopandas as gpd
import zipfile, os, tempfile
import matplotlib.pyplot as plt
from branca.element import MacroElement, Template
from matplotlib import colors as mpl_colors
from scipy.interpolate import griddata

st.set_page_config(page_title="Farm ROI Tool V4", layout="wide")
st.title("Farm Profit Mapping Tool V4")

# =========================================================
# CSS – Compress UI
# =========================================================
st.markdown("""
<style>
div[data-testid="column"] {padding-left:.15rem!important; padding-right:.15rem!important;}
section[data-testid="stVerticalBlock"] > div {padding-top:.15rem!important; padding-bottom:.15rem!important;}
h2,h3 {margin:.2rem 0 .2rem 0!important; font-size:1rem!important;}
div[data-testid="stExpander"] details summary {padding:.2rem .4rem!important; font-size:.8rem!important;}
div[data-testid="stNumberInput"] div[role="spinbutton"] {
  min-height:22px!important; height:22px!important; padding:0 4px!important; font-size:.75rem!important;}
div[data-testid="stNumberInput"] button {min-width:18px!important; padding:0!important;}
div[data-testid="stFileUploaderDropzone"] {padding:0.2rem!important; min-height:32px!important;}
div[data-testid="stFileUploaderDropzone"] p {font-size:0.65rem!important; margin:0!important;}
div[data-testid="stDataFrame"] table, div[data-testid="stDataEditor"] table {font-size:.75rem!important;}
div[data-testid="stDataFrame"] th, div[data-testid="stDataFrame"] td,
div[data-testid="stDataEditor"] th, div[data-testid="stDataEditor"] td {
  padding:1px 4px!important; line-height:1rem!important;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def load_vector_file(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith((".geojson",".json")):
            gdf=gpd.read_file(uploaded_file)
        elif uploaded_file.name.lower().endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path=os.path.join(tmpdir,"in.zip")
                with open(zip_path,"wb") as f: f.write(uploaded_file.getbuffer())
                with zipfile.ZipFile(zip_path,"r") as zf: zf.extractall(tmpdir)
                shp_path=None
                for fn in os.listdir(tmpdir):
                    if fn.lower().endswith(".shp"):
                        shp_path=os.path.join(tmpdir,fn); break
                if shp_path is None: return None
                gdf=gpd.read_file(shp_path)
        else: return None
        if gdf is None or gdf.empty: return None
        if gdf.crs is None: gdf.set_crs(epsg=4326,inplace=True)
        return gdf.to_crs(epsg=4326)
    except Exception: return None

def _mini_num(label,key,default=0.0,step=0.1):
    st.caption(label)
    return st.number_input(key,min_value=0.0,value=float(default),step=step,label_visibility="collapsed")

# =========================================================
# UPLOAD SECTION – one row, compact
# =========================================================
st.markdown("### Upload Maps")

c1,c2,c3,c4 = st.columns(4)
with c1: zone_file = st.file_uploader("Zone", type=["geojson","json","zip"])
with c2: yield_files = st.file_uploader("Yield", type=["csv","geojson","json","zip"], accept_multiple_files=True)
with c3: fert_files = st.file_uploader("Fert", type=["csv","geojson","json","zip"], accept_multiple_files=True)
with c4: seed_files = st.file_uploader("Seed", type=["csv","geojson","json","zip"], accept_multiple_files=True)

# =========================================================
# SECTION 4 – EXPENSES + CROP PROFITABILITY
# =========================================================
row1,row2 = st.columns(6),st.columns(6)
with row1[0]: chemicals=_mini_num("Chem ($/ac)","chem")
with row1[1]: insurance=_mini_num("Insur ($/ac)","ins")
with row1[2]: insecticide=_mini_num("Insect/Fung ($/ac)","insect")
with row1[3]: fertilizer=_mini_num("Fert Flat ($/ac)","fert")
with row1[4]: seed=_mini_num("Seed Flat ($/ac)","seed")
with row1[5]: cash_rent=_mini_num("Cash Rent ($/ac)","rent")
with row2[0]: machinery=_mini_num("Mach ($/ac)","mach")
with row2[1]: labor=_mini_num("Labor ($/ac)","labor")
with row2[2]: coliving=_mini_num("Living ($/ac)","col")
with row2[3]: extra_fuel=_mini_num("Fuel ($/ac)","fuel")
with row2[4]: extra_interest=_mini_num("Interest ($/ac)","int")
with row2[5]: truck_fuel=_mini_num("Truck Fuel ($/ac)","truck")

expenses={"Chemicals":chemicals,"Insurance":insurance,"Insecticide/Fungicide":insecticide,
"Fertilizer (Flat)":fertilizer,"Seed (Flat)":seed,"Machinery":machinery,"Labor":labor,
"Cost of Living":coliving,"Extra Fuel":extra_fuel,"Extra Interest":extra_interest,
"Truck Fuel":truck_fuel,"Cash Rent":cash_rent}
base_expenses_per_acre=float(sum(expenses.values()))

# Crop assumptions
left,right=st.columns(2)
with right:
    c1,c2=st.columns(2)
    with c1:
        st.caption("Corn (Yield / Price)")
        cy=st.number_input("cy",0.0,step=1.0,value=200.0,label_visibility="collapsed")
        cp=st.number_input("cp",0.0,step=0.1,value=5.0,label_visibility="collapsed")
    with c2:
        st.caption("Soy (Yield / Price)")
        sy=st.number_input("sy",0.0,step=1.0,value=60.0,label_visibility="collapsed")
        sp=st.number_input("sp",0.0,step=0.1,value=12.0,label_visibility="collapsed")

    preview_df=pd.DataFrame({
        "Crop":["Corn","Soy"],
        "Yield":[cy,sy],
        "Price":[cp,sp],
        "Revenue":[cy*cp,sy*sp],
        "Fixed":[base_expenses_per_acre]*2
    })
    preview_df["Breakeven"]=preview_df["Revenue"]-preview_df["Fixed"]
    st.dataframe(preview_df.style.format({
        "Yield":"{:.0f}","Price":"${:.2f}","Revenue":"${:,.0f}",
        "Fixed":"${:,.0f}","Breakeven":"${:,.0f}"
    }),use_container_width=True,hide_index=True)

with left:
    with st.expander("Fixed Rate Inputs",expanded=False):
        if "fixed_products" not in st.session_state:
            st.session_state["fixed_products"]=pd.DataFrame({
                "Type":["Seed","Fertilizer"],"Product":["",""],
                "Rate":[0.0,0.0],"CostPerUnit":[0.0,0.0], "$/ac":[0.0,0.0]})
        fixed_entries=st.data_editor(st.session_state["fixed_products"],
            num_rows="dynamic",use_container_width=True,key="fixed_editor")
        st.session_state["fixed_products"]=fixed_entries.copy().reset_index(drop=True)
    with st.expander("Variable Rate Inputs",expanded=False):
        if "fert_products" in st.session_state and not st.session_state["fert_products"].empty:
            st.dataframe(st.session_state["fert_products"],use_container_width=True,hide_index=True)
        if "seed_products" in st.session_state and not st.session_state["seed_products"].empty:
            st.dataframe(st.session_state["seed_products"],use_container_width=True,hide_index=True)
        if ("fert_products" not in st.session_state or st.session_state["fert_products"].empty) and \
           ("seed_products" not in st.session_state or st.session_state["seed_products"].empty):
            st.caption("— No VR inputs —")

# =========================================================
# BASE MAP
# =========================================================
def make_base_map():
    try:
        m=folium.Map(location=[39.5,-98.35],zoom_start=5,min_zoom=2,tiles="CartoDB positron")
        return m
    except: return folium.Map(location=[39.5,-98.35],zoom_start=4)
m=make_base_map()
st_folium(m,use_container_width=True,height=500)

# =========================================================
# PROFIT SUMMARY (tight)
# =========================================================
st.subheader("Profit Summary")
colL,colR=st.columns(2)
with colL:
    corn_revenue=cy*cp; bean_revenue=sy*sp
    corn_budget=corn_revenue-base_expenses_per_acre
    bean_budget=bean_revenue-base_expenses_per_acre
    df=pd.DataFrame({
        "Crop":["Corn","Soybeans"],
        "Yield Goal":[cy,sy],
        "Price":[cp,sp],
        "Revenue":[corn_revenue,bean_revenue],
        "Fixed Inputs":[base_expenses_per_acre]*2,
        "Breakeven":[corn_budget,bean_budget]
    })
    st.dataframe(df.style.format({"Yield Goal":"{:.1f}","Price":"${:.2f}",
        "Revenue":"${:,.0f}","Fixed Inputs":"${:,.0f}","Breakeven":"${:,.0f}"}),
        use_container_width=True,hide_index=True)

with colR:
    fixed_df=pd.DataFrame(list(expenses.items()),columns=["Expense","$/ac"])
    fixed_df.loc[len(fixed_df)] = ["Total Fixed Costs", fixed_df["$/ac"].sum()]
    st.dataframe(fixed_df.style.format({"$/ac":"${:,.2f}"}),use_container_width=True,hide_index=True)
