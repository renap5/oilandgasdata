import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

# Get API key securely from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]

# Load sample DataFrame (replace with your Excel or CSV)
df = pd.DataFrame({
    "Well": ["A1", "B2", "C3"],
    "Production": [100, 200, 0],
    "Status": ["Active", "Active", "Shut In"]
})

# Initialize SmartDataframe
sdf = SmartDataframe(df, config={"llm": OpenAI(api_token=openai_api_key)})

# Streamlit UI
st.title("Idle Well Data Assistant")

question = st.text_input("Ask a question about your well data:")
if question:
    try:
        response = sdf.chat(question)
        st.write("Answer:")
        st.write(response)
    except Exception as e:
        st.error(f"Error: {e}")
