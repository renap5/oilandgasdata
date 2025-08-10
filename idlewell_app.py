# idlewell_app.py  — minimal chat-only UI

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Idle Well Assistant", layout="wide")

# --------- Load & prep data (silent) ---------
def find_repo_excel():
    repo_dir = Path(__file__).parent.resolve()
    candidates = [
        repo_dir / "data" / "2024_IWMP_Inventory_Public.xlsx",
        repo_dir / "2024_IWMP_Inventory_Public.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def read_excel_with_header_guess(path, header_candidates=(0, 1, 2)):
    last_err = None
    for h in header_candidates:
        try:
            _df = pd.read_excel(path, header=h)
            if _df.shape[1] >= 2:
                return _df
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise ValueError("Failed reading Excel with header guesses")

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(frame: pd.DataFrame, candidates):
        lookup = {str(c).strip().casefold(): c for c in frame.columns}
        for name in candidates:
            key = str(name).strip().casefold()
            if key in lookup:
                return lookup[key]
        return None

    # Broad candidates for common CalGEM headers
    idle_start_candidates = [
        "Idle Start Date","Idle Start","IdleStartDate","Date Idle Start",
        "Idle_Start_Date","Idle start date","Idle Start Dt","Idle Date Start"
    ]
    years_idle_candidates = [
        "Years Idle","Idle Years","YearsIdle","Years idle",
        "Years_Idle","Yrs Idle","Years of Idle","Years Idle (yrs)"
    ]

    col_idle_start = find_col(df, idle_start_candidates)
    col_years_idle = find_col(df, years_idle_candidates)

    # Build helper columns silently
    if col_idle_start:
        df[col_idle_start] = pd.to_datetime(df[col_idle_start], errors="coerce")
        df["Year"] = df[col_idle_start].dt.year

    if "Year" not in df.columns and col_years_idle:
        current_year = pd.Timestamp.now().year
        yi = pd.to_numeric(df[col_years_idle], errors="coerce")
        df["Year"] = current_year - yi

    if col_years_idle:
        df[col_years_idle] = pd.to_numeric(df[col_years_idle], errors="coerce")

    return df

# Try to load data
DATA_PATH = find_repo_excel()
if DATA_PATH is None:
    st.error("Data not available. Please add the Excel file to the repository and redeploy.")
    st.stop()

try:
    df = read_excel_with_header_guess(DATA_PATH, header_candidates=(0, 1, 2))
    df = normalize_df(df)
except Exception as e:
    st.error("Data could not be prepared. Please contact support.")
    st.stop()

# --------- AI chat (minimal interface) ---------
try:
    from pandasai import SmartDataframe
    from pandasai.llm import OpenAI as PAIOpenAI
    OPENAI_KEY = st.secrets["openai"]["api_key"]
except Exception:
    st.error("AI is unavailable. Check that OpenAI key is set in Streamlit Secrets and pandasai is installed.")
    st.stop()

# Give the model a short schema hint to avoid inventing columns
schema_hint = (
    "Use only these columns: "
    + ", ".join(map(str, df.columns.tolist()))
    + ". If you need well age, use 'Years Idle' or compute from 'Idle Start Date'."
)

sdf = SmartDataframe(df, config={"llm": PAIOpenAI(api_token=OPENAI_KEY)})

# Clean chat-only UI
st.markdown("## Ask about the dataset")
user_msg = st.chat_input("Ask a question (SQL or natural language)…")

# Optional: keep short chat history
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

if user_msg:
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = sdf.chat(schema_hint + " " + user_msg)
            st.markdown(str(answer))
            st.session_state.history.append(("assistant", str(answer)))
    except Exception as e:
        with st.chat_message("assistant"):
            st.markdown("Sorry, I couldn’t answer that.")