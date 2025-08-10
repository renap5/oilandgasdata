# idlewell_app.py — smart, minimal, chat-only Idle Well Inventory assistant

import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Idle Well Inventory Assistant", layout="wide")

# ====== Load Excel from repo (no UI noise) ====================================
REPO_DIR = Path(__file__).parent.resolve()
CANDIDATE_PATHS = [
    REPO_DIR / "data" / "2024_IWMP_Inventory_Public.xlsx",
    REPO_DIR / "2024_IWMP_Inventory_Public.xlsx",
]
DATA_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), None)
if DATA_PATH is None:
    st.error("Data not available in repository. Add the Excel to `data/…` or repo root and redeploy.")
    st.stop()

def read_excel_with_header_guess(path: Path, header_candidates=(0, 1, 2, 3)):
    last_err = None
    for h in header_candidates:
        try:
            _df = pd.read_excel(path, header=h)
            if _df is not None and _df.shape[1] >= 2:
                return _df
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise ValueError("Could not read Excel with any header row candidate")

try:
    df = read_excel_with_header_guess(DATA_PATH)
except Exception as e:
    st.error("Data could not be read. Please verify the Excel file format.")
    st.stop()

# ====== Normalize columns (silent) ============================================
df = df.copy()
df.columns = [str(c).strip() for c in df.columns]

def find_col_fuzzy(frame: pd.DataFrame, must_have=None, any_of=None):
    """
    Return best-matching column by fuzzy rules:
      - must_have: list of tokens that must all appear (case-insensitive)
      - any_of: list of tokens where any may appear
    """
    cols = [str(c).strip() for c in frame.columns]
    def tok(s): return re.findall(r"[A-Za-z0-9]+", s.casefold())
    best = None
    best_score = -1
    for c in cols:
        tokens = tok(c)
        if must_have and not all(t in tokens for t in [t.casefold() for t in must_have]):
            continue
        score = 0
        if any_of:
            score += sum(1 for t in any_of if t.casefold() in tokens)
        # prefer shorter/cleaner names when tied
        score -= 0.01 * len(c)
        if score > best_score:
            best_score, best = score, c
    return best

# Try explicit candidates first; fall back to fuzzy patterns
def pick_col(frame, candidates, fuzzy_must=None, fuzzy_any=None):
    lookup = {c.casefold(): c for c in frame.columns.astype(str)}
    for cand in candidates:
        key = str(cand).strip().casefold()
        if key in lookup:
            return lookup[key]
    return find_col_fuzzy(frame, must_have=fuzzy_must, any_of=fuzzy_any)

col_api        = pick_col(df,
    ["API 10", "API", "API Number", "API10", "API Number (10)"],
    fuzzy_any=["api","number","10"]
)
col_well       = pick_col(df,
    ["Well Designation", "Well Name", "Well", "Well ID", "Well No."],
    fuzzy_any=["well","name","designation","id","no"]
)
col_operator   = pick_col(df,
    ["Operator Name","Operator","Current Operator"],
    fuzzy_any=["operator","name"]
)
col_idle_start = pick_col(df,
    ["Idle Start Date","Idle Start","IdleStartDate","Date Idle Start","Idle_Start_Date","Idle Start Dt","Idle Date Start"],
    fuzzy_must=["idle"], fuzzy_any=["start","date"]
)
col_years_idle = pick_col(df,
    ["Years Idle","Idle Years","YearsIdle","Years idle","Years_Idle","Yrs Idle","Years of Idle","Years Idle (yrs)"],
    fuzzy_must=["idle"], fuzzy_any=["years","yrs"]
)

# Build helper columns silently
if col_idle_start:
    df[col_idle_start] = pd.to_datetime(df[col_idle_start], errors="coerce")
    df["Year"] = df[col_idle_start].dt.year

if "Year" not in df.columns and col_years_idle:
    current_year = pd.Timestamp.now().year
    df[col_years_idle] = pd.to_numeric(df[col_years_idle], errors="coerce")
    df["Year"] = current_year - df[col_years_idle]

# Minimal guard: ensure it looks like an idle well inventory (has at least one key col)
has_idle_signal = (col_idle_start in df.columns) or (col_years_idle in df.columns)
if not has_idle_signal:
    # Still allow chat, but make a light internal note to the model (below)
    pass

# ====== Deterministic fallback utilities (silent) =============================
def oldest_idle_well_row(_df: pd.DataFrame):
    # Prefer earliest Idle Start Date if present
    if col_idle_start and col_idle_start in _df.columns:
        tmp = _df.dropna(subset=[col_idle_start])
        if len(tmp):
            return tmp.sort_values(col_idle_start, ascending=True).iloc[0]
    # Else use max Years Idle
    if col_years_idle and col_years_idle in _df.columns:
        tmp = _df.copy()
        tmp[col_years_idle] = pd.to_numeric(tmp[col_years_idle], errors="coerce")
        tmp = tmp.dropna(subset=[col_years_idle])
        if len(tmp):
            return tmp.sort_values(col_years_idle, ascending=False).iloc[0]
    return None

def summarize_row(r: pd.Series):
    return {
        (col_api or "API"): r.get(col_api) if col_api else None,
        (col_well or "Well"): r.get(col_well) if col_well else None,
        (col_operator or "Operator"): r.get(col_operator) if col_operator else None,
        (col_idle_start or "Idle Start Date"): str(r.get(col_idle_start)) if col_idle_start else None,
        (col_years_idle or "Years Idle"): r.get(col_years_idle) if col_years_idle else None,
        "Year": r.get("Year") if "Year" in df.columns else None,
    }

# ====== AI chat (clean chat-only UI) ==========================================
try:
    from pandasai import SmartDataframe
    from pandasai.llm import OpenAI as PAIOpenAI
    OPENAI_KEY = st.secrets["openai"]["api_key"]
except Exception:
    st.error("AI unavailable. Ensure OpenAI key is set in Streamlit Secrets and `pandasai` is in requirements.")
    st.stop()

# Strong domain + schema context to keep answers grounded
domain_context = (
    "You are analyzing the California Geologic Energy Management (CalGEM) "
    "Idle Well Inventory dataset. Treat ALL rows as idle wells unless explicitly filtered. "
    "Use only the columns present in the dataframe. "
    "If you need the well age or how long it has been idle, use 'Years Idle' when available, "
    "or compute it from 'Idle Start Date'. "
    "When asked for 'oldest idle well', return the well with the EARLIEST 'Idle Start Date'; "
    "if that column is unavailable, use the maximum 'Years Idle'. "
)

schema_hint = "Columns available: " + ", ".join(map(str, df.columns.tolist())) + "."

sdf = SmartDataframe(df, config={"llm": PAIOpenAI(api_token=OPENAI_KEY)})

st.markdown("## Ask about the Idle Well Inventory")
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

user_msg = st.chat_input("Ask a question (e.g., 'What is the oldest idle well in California?')…")
if user_msg:
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    try:
        enriched = f"{domain_context}\n{schema_hint}\n\nQuestion: {user_msg}"
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = sdf.chat(enriched)
            st.markdown(str(answer))
            st.session_state.history.append(("assistant", str(answer)))
    except Exception as e:
        # Fallback for common request about "oldest"
        text = user_msg.casefold()
        if "oldest" in text and ("idle" in text or "well" in text):
            row = oldest_idle_well_row(df)
            with st.chat_message("assistant"):
                if row is not None:
                    st.markdown("**Oldest idle well (deterministic result):**")
                    st.json(summarize_row(row))
                    st.session_state.history.append(("assistant", "Returned deterministic oldest idle well result."))
                else:
                    st.markdown("I couldn’t infer the oldest idle well from this file.")
                    st.session_state.history.append(("assistant", "No oldest idle well could be inferred."))
        else:
            with st.chat_message("assistant"):
                st.markdown("Sorry, I couldn’t answer that.")
                st.session_state.history.append(("assistant", "Could not answer due to an internal error."))