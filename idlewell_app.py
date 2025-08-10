# idlewell_app.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Idle Well Data Assistant", layout="wide")
st.title("Idle Well Data Assistant")
st.caption("Loads CalGEM Excel from the repository, normalizes columns, and (optionally) enables AI chat.")

# ---------- Configure data path(s) ----------
REPO_DIR = Path(__file__).parent.resolve()
CANDIDATE_PATHS = [
    REPO_DIR / "data" / "2024_IWMP_Inventory_Public.xlsx",
    REPO_DIR / "2024_IWMP_Inventory_Public.xlsx",
]

DATA_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), None)
if DATA_PATH is None:
    st.error(
        "Could not find the Excel file in the repo.\n\n"
        "Place it at **data/2024_IWMP_Inventory_Public.xlsx** or at the repo root as "
        "**2024_IWMP_Inventory_Public.xlsx**, commit, and redeploy."
    )
    st.stop()

st.write(f"**Using file:** `{DATA_PATH.relative_to(REPO_DIR)}`")

# ---------- Load Excel ----------
try:
    df = pd.read_excel(DATA_PATH)
except Exception as e:
    st.error(f"Failed to read Excel: {e}")
    st.stop()

# ---------- Normalize columns & helpers ----------
# Strip whitespace from column names
df.columns = [str(c).strip() for c in df.columns]

def find_col(frame: pd.DataFrame, candidates):
    """Return the actual column name that case-insensitively matches any candidate."""
    lookup = {str(c).strip().casefold(): c for c in frame.columns}
    for name in candidates:
        key = str(name).strip().casefold()
        if key in lookup:
            return lookup[key]
    return None

# Map typical CalGEM headers (adjust if your file differs)
col_api         = find_col(df, ["API 10", "API", "API Number", "API10"])
col_well        = find_col(df, ["Well Designation", "Well Name", "Well"])
col_op          = find_col(df, ["Operator Name", "Operator"])
col_idle_start  = find_col(df, ["Idle Start Date", "Idle Start", "IdleStartDate"])
col_years_idle  = find_col(df, ["Years Idle", "Idle Years", "YearsIdle"])

# Create 'Year' column if possible
if col_idle_start:
    df[col_idle_start] = pd.to_datetime(df[col_idle_start], errors="coerce")
    df["Year"] = df[col_idle_start].dt.year

if "Year" not in df.columns and col_years_idle:
    current_year = pd.Timestamp.now().year
    yi = pd.to_numeric(df[col_years_idle], errors="coerce")
    df["Year"] = current_year - yi

# Coerce Years Idle to numeric for later calcs
if col_years_idle:
    df[col_years_idle] = pd.to_numeric(df[col_years_idle], errors="coerce")

# ---------- Preview ----------
with st.expander("Preview data"):
    st.write("**Columns:**", list(df.columns))
    st.dataframe(df.head(25), use_container_width=True)

# ---------- Fallback: Oldest idle well ----------
def compute_oldest_row(_df: pd.DataFrame):
    if col_idle_start and col_idle_start in _df.columns:
        tmp = _df.dropna(subset=[col_idle_start]).copy()
        if len(tmp) == 0:
            return None
        return tmp.sort_values(col_idle_start, ascending=True).iloc[0]
    if col_years_idle and col_years_idle in _df.columns:
        tmp = _df.dropna(subset=[col_years_idle]).copy()
        if len(tmp) == 0:
            return None
        return tmp.sort_values(col_years_idle, ascending=False).iloc[0]
    return None

st.subheader("Quick analysis")
row = compute_oldest_row(df)
if row is None:
    st.warning("Could not determine the oldest idle well (need 'Idle Start Date' or 'Years Idle').")
else:
    st.success("Oldest idle well (deterministic fallback):")
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

        # Read from Streamlit Secrets (Streamlit Cloud: Manage app → Settings → Secrets)
        OPENAI_KEY = st.secrets["openai"]["api_key"]

        # Schema hint to reduce LLM hallucinating columns
        schema_hint = (
            "Use only these columns: "
            + ", ".join(map(str, df.columns.tolist()))
            + ". If you need well age, prefer 'Years Idle' or compute from 'Idle Start Date'."
        )

        sdf = SmartDataframe(df, config={"llm": PAIOpenAI(api_token=OPENAI_KEY)})

        q = st.text_input("Ask a question about the dataset:")
        if q:
            with st.spinner("Thinking..."):
                answer = sdf.chat(schema_hint + " " + q)
            st.write("**Answer:**")
            st.write(answer)

    except KeyError:
        st.error(
            "OpenAI API key not found in Streamlit Secrets.\n"
            "Add under Settings → Secrets:\n\n[openai]\napi_key = \"sk-...\""
        )
    except Exception as e:
        st.error(f"AI chat failed: {e}")