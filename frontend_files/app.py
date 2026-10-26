import streamlit as st
import requests
import pandas as pd
import json
import os

BACKEND_URL = os.getenv("BACKEND_URL","http://backend:7860")

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="wide"
)
st.title("SuperKart Sales Prediction")
st.write("Predict the total sales revenue for a product in a store")

tab_online, tab_batch = st.tabs(["Online", "Batch"])

with tab_online:
    col1,col2 = st.columns(2)
    with col1:
      product_weight = st.number_input("Product Weight")
      product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Medium Sugar", "High Sugar"])
      product_allocated_area = st.number_input("Product Allocated Area")
      product_mrp = st.number_input("Product MRP")
      store_age_years = st.number_input("Store Age in Years")
      product_type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])
    with col2:
      store_size = st.selectbox("Store Size", ["Small", "Medium", "Large"])
      store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
      store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])
      product_id_char = st.selectbox("Product ID Character", ["FD", "NC"])

    if st.button("Predict"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age_years,
            "Product_Type_Category": product_type_category
        }
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload)
        if response.status_code == 200:
            st.success(f"Prediction: {response.json()['prediction']}")
        else:
            st.error(f"Error: {response.text}")

with tab_batch:
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        if st.button("Predict"):
            files = {"file": uploaded_file}
            response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files)
            if response.status_code == 200:
                st.success("Predictions:")
                st.json(json.loads(response.text))
            else:
                st.error(f"Error: {response.text}")
