import streamlit as st
import pandas as pd

st.title("Idle Well Data Assistant")

uploaded = st.file_uploader("Upload CalGEM Excel (.xlsx)", type=["xlsx"])
if uploaded is not None:
    df = pd.read_excel(uploaded)
    st.success(f"Loaded {len(df):,} rows")
    st.dataframe(df.head(20))
else:
    st.info("Upload an Excel file to begin.")
    st.stop()
