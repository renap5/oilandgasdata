import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

# Load the Excel file
df = pd.read_excel("2024_IWMP_Inventory_Public.xlsx")

# Initialize the LLM (insert your actual OpenAI API key here)
llm = OpenAI(api_token="sk-proj-...")  # Replace with your key

# Wrap the DataFrame
sdf = SmartDataframe(df, config={"llm": llm})

# Streamlit UI
st.title("🛢️ Idle Well Explorer")
st.markdown("Ask questions about the CalGEM Idle Well data")

query = st.text_input("Ask your question here:")

if st.button("Submit"):
    with st.spinner("Analyzing..."):
        response = sdf.chat(query)
        st.success(response)
