# idlewell_app.py

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Idle Well Data Assistant", layout="wide")
st.title("Idle Well Data Assistant")

st.write(
    "Upload your CalGEM Excel (.xlsx). The app will preview data, "
    "create a 'Year' helper column if needed, and optionally enable AI chat."
)

# ---------- File upload ----------
uploaded = st.file_uploader("Upload CalGEM Excel (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Upload an Excel file to begin.")
    st.stop()

# Read Excel (openpyxl is required; make sure it's in requirements.txt)
try:
    df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

# ---------- Normalize columns & build helpers ----------
# Strip whitespace from column names
df.columns = [str(c).strip() for c in df.columns]

# Utility to find a column ignoring case and surrounding spaces
def find_col(candidates):
    lc = {str(c).strip().casefold(): c for c in df.columns}
    for name in candidates:
        key = str(name).strip().casefold()
        if key in lc:
            return lc[key]
    return None

# Try to discover common CalGEM fields (adjust if your headers differ)
col_api = find_col(["API 10", "API", "API Number", "API10"])
col_well = find_col(["Well Designation", "Well Name", "Well"])
col_op = find_col(["Operator Name", "Operator"])
col_idle_start = find_col(["Idle Start Date", "Idle Start", "IdleStartDate"])
col_years_idle = find_col(["Years Idle", "Idle Years", "YearsIdle"])

# Create a 'Year' column if possible
if col_idle_start:
    df[col_idle_start] = pd.to_datetime(df[col_idle_start], errors="coerce")
    df["Year"] = df[col_idle_start].dt.year

if "Year" not in df.columns and col_years_idle:
    current_year = pd.Timestamp.now().year
    yi = pd.to_numeric(df[col_years_idle], errors="coerce")
    df["Year"] = current_year - yi

# Optional: coerce Years Idle to numeric for later calculations
if col_years_idle:
    df[col_years_idle] = pd.to_numeric(df[col_years_idle], errors="coerce")

# ---------- Preview ----------
with st.expander("Preview data"):
    st.write("**Columns:**", list(df.columns))
    st.dataframe(df.head(25), use_container_width=True)

# ---------- Fallback: Oldest idle well (deterministic) ----------
def compute_oldest_row(_df):
    # Prefer earliest Idle Start Date if available
    if col_idle_start and col_idle_start in _df.columns:
        tmp = _df.dropna(subset=[col_idle_start]).copy()
        if len(tmp) == 0:
            return None
        return tmp.sort_values(col_idle_start, ascending=True).iloc[0]
    # Else use max Years Idle
    if col_years_idle and col_years_idle in _df.columns:
        tmp = _df.dropna(subset=[col_years_idle]).copy()
        if len(tmp) == 0:
            return None
        return tmp.sort_values(col_years_idle, ascending=False).iloc[0]
    return None

st.subheader("Quick analysis")
if st.button("Find oldest idle well (fallback)"):
    row = compute_oldest_row(df)
    if row is None:
        st.warning(
            "Could not determine the oldest idle well. "
            "Make sure the file has either 'Idle Start Date' or 'Years Idle'."
        )
    else:
        st.success("Oldest idle well (fallback calculation):")
        st.write({
            (col_api or "API"): row.get(col_api) if col_api else None,
            (col_well or "Well"): row.get(col_well) if col_well else None,
            (col_op or "Operator"): row.get(col_op) if col_op else None,
            (col_idle_start or "Idle Start Date"): str(row.get(col_idle_start)) if col_idle_start else None,
            (col_years_idle or "Years Idle"): row.get(col_years_idle) if col_years_idle else None,
            "Year": row.get("Year") if "Year" in df.columns else None,
        })

# ---------- Optional: AI chat over the DataFrame ----------
st.subheader("Ask questions with AI (optional)")

enable_ai = st.toggle("Enable AI chat", value=False, help="Requires OpenAI API key in Streamlit Secrets.")
if enable_ai:
    try:
        from pandasai import SmartDataframe
        from pandasai.llm import OpenAI as PAIOpenAI

        # Read key from Streamlit Secrets (add it in Streamlit Cloud: Settings → Secrets)
        OPENAI_KEY = st.secrets["openai"]["api_key"]

        # Give the LLM a schema hint so it doesn't invent columns like 'Year' if missing
        schema_hint = (
            "Use only these columns: "
            + ", ".join(map(str, df.columns.tolist()))
            + ". If you need well age, prefer 'Years Idle' or compute from 'Idle Start Date'."
        )

        sdf = SmartDataframe(df, config={"llm": PAIOpenAI(api_token=OPENAI_KEY)})

        q = st.text_input("Ask a question about the uploaded data:")
        if q:
            with st.spinner("Thinking..."):
                # Prepend schema hint to steer the model
                answer = sdf.chat(schema_hint + " " + q)
            st.write("**Answer:**")
            st.write(answer)

    except KeyError:
        st.error(
            "OpenAI API key not found in Streamlit Secrets. "
            "Add it under Settings → Secrets (format:\n\n"
            "[openai]\napi_key = \"sk-...\"\n)"
        )
    except Exception as e:
        st.error(f"AI chat failed: {e}")