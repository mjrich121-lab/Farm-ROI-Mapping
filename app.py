import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import branca.colormap as cm

st.set_page_config(page_title="Farm ROI Tool", layout="wide")

st.title("🌽 Farm ROI & Profit Map Tool")

uploaded_files = st.file_uploader("Upload Yield Map CSV(s)", type="csv", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"Field: {file.name}")
        df = pd.read_csv(file)

        # --- Per-field financial inputs ---
        with st.expander("Field Inputs & Expenses", expanded=True):
            acres = st.number_input("Field Acres", min_value=1, value=100, key=f"{file.name}_acres")
            sell_price = st.number_input("Sell Price ($/bu)", min_value=0.0, value=5.0, key=f"{file.name}_sell_price")
            yield_ac = st.number_input("Yield (bu/ac)", min_value=0.0, value=200.0, key=f"{file.name}_yield")

            st.markdown("### Expenses per Acre ($)")
            chemicals = st.number_input("Chemicals", value=25.0, key=f"{file.name}_chemicals")
            insurance = st.number_input("Crop & Hail Insurance", value=50.0, key=f"{file.name}_insurance")
            insecticide = st.number_input("Insecticide/Fungicide", value=25.0, key=f"{file.name}_insecticide")
            fertilizer = st.number_input("Fertilizer", value=100.0, key=f"{file.name}_fertilizer")
            machinery = st.number_input("Machinery", value=150.0, key=f"{file.name}_machinery")
            seed = st.number_input("Seed", value=100.0, key=f"{file.name}_seed")
            cost_of_living = st.number_input("Cost of Living", value=25.0, key=f"{file.name}_cost_of_living")
            extra_fuel = st.number_input("Extra Fuel for Corn", value=20.0, key=f"{file.name}_extra_fuel")
            extra_interest = st.number_input("Extra Interest", value=20.0, key=f"{file.name}_extra_interest")
            truck_fuel = st.number_input("Truck Fuel", value=5.0, key=f"{file.name}_truck_fuel")
            labor = st.number_input("Labor", value=15.0, key=f"{file.name}_labor")
            cash_rent = st.number_input("Cash Rent", value=100.0, key=f"{file.name}_cash_rent")

        # --- ROI Calculations ---
        expense_inputs = [chemicals, insurance, insecticide, fertilizer, machinery,
                          seed, cost_of_living, extra_fuel, extra_interest, truck_fuel,
                          labor, cash_rent]
        expenses_per_acre = sum(expense_inputs)
        revenue_per_acre = yield_ac * sell_price
        net_profit_per_acre = revenue_per_acre - expenses_per_acre
        roi_percent = (net_profit_per_acre / expenses_per_acre * 100) if expenses_per_acre else 0

        # --- ROI Report ---
        report = pd.DataFrame({
            "Metric": ["Acres", "Sell Price ($/bu)", "Yield (bu/ac)",
                       "Revenue per Acre ($)", "Expenses per Acre ($)",
                       "Net Profit per Acre ($)", "ROI (%)"],
            "Value": [acres, sell_price, yield_ac,
                      round(revenue_per_acre, 2), round(expenses_per_acre, 2),
                      round(net_profit_per_acre, 2), round(roi_percent, 2)]
        })
        st.table(report)

        # --- Profit Map ---
        if "Latitude" in df.columns and "Longitude" in df.columns and "Yield" in df.columns:
            df["Revenue_per_acre"] = df["Yield"] * sell_price
            df["Expenses_per_acre"] = expenses_per_acre
            df["NetProfit_per_acre"] = df["Revenue_per_acre"] - df["Expenses_per_acre"]

            m = folium.Map(
                location=[df["Latitude"].mean(), df["Longitude"].mean()],
                zoom_start=15,
                tiles=None
            )
            # Satellite + labels
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="Esri Satellite",
                overlay=False,
                control=False
            ).add_to(m)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="Labels",
                overlay=True,
                control=False
            ).add_to(m)

            colormap = cm.LinearColormap(colors=["red", "yellow", "green"],
                                         vmin=df["NetProfit_per_acre"].min(),
                                         vmax=df["NetProfit_per_acre"].max())
            colormap.caption = "Net Profit per Acre ($)"
            colormap.add_to(m)

            for _, row in df.iterrows():
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=5,
                    color=colormap(row["NetProfit_per_acre"]),
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"Yield: {row['Yield']} bu/ac<br>Profit: ${row['NetProfit_per_acre']:.2f}/ac"
                ).add_to(m)

            st_folium(m, width=700, height=500)
        else:
            st.warning("CSV must include Latitude, Longitude, and Yield columns.")
