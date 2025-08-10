# idlewell_app.py — Idle Well Inventory assistant (chat-only + microphone)

from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_mic_recorder import mic_recorder

# ---------------------------- Page setup --------------------------------------
st.set_page_config(page_title="Idle Well Inventory Assistant", layout="wide")
st.markdown("## Ask about the Idle Well Inventory")

# ---------------------------- Data loading ------------------------------------
REPO_DIR = Path(__file__).parent.resolve()
CANDIDATE_PATHS = [
    REPO_DIR / "data" / "2024_IWMP_Inventory_Public.xlsx",
    REPO_DIR / "2024_IWMP_Inventory_Public.xlsx",
]
DATA_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), None)
if DATA_PATH is None:
    st.error("Data not found in repo. Add the Excel to `data/` or repo root and redeploy.")
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
except Exception:
    st.error("Data could not be read. Please verify the Excel format.")
    st.stop()

# ---------------------------- Normalize columns -------------------------------
df = df.copy()
df.columns = [str(c).strip() for c in df.columns]

def _fuzzy_find_col(frame: pd.DataFrame, must_have=None, any_of=None):
    cols = [str(c).strip() for c in frame.columns]
    def tok(s): return re.findall(r"[A-Za-z0-9]+", s.casefold())
    best, best_score = None, -1
    for c in cols:
        tokens = tok(c)
        if must_have and not all(t in tokens for t in [t.casefold() for t in must_have]):
            continue
        score = 0
        if any_of:
            score += sum(1 for t in any_of if t.casefold() in tokens)
        score -= 0.01 * len(c)  # prefer shorter when tied
        if score > best_score:
            best, best_score = c, score
    return best

def _pick_col(frame, candidates, fuzzy_must=None, fuzzy_any=None):
    lookup = {str(c).strip().casefold(): c for c in frame.columns}
    for cand in candidates:
        key = str(cand).strip().casefold()
        if key in lookup:
            return lookup[key]
    return _fuzzy_find_col(frame, must_have=fuzzy_must, any_of=fuzzy_any)

col_api        = _pick_col(df, ["API 10","API","API Number","API10","API Number (10)"], fuzzy_any=["api","number","10"])
col_well       = _pick_col(df, ["Well Designation","Well Name","Well","Well ID","Well No."], fuzzy_any=["well","name","designation","id","no"])
col_operator   = _pick_col(df, ["Operator Name","Operator","Current Operator"], fuzzy_any=["operator","name"])
col_idle_start = _pick_col(
    df,
    ["Idle Start Date","Idle Start","IdleStartDate","Date Idle Start","Idle_Start_Date","Idle Start Dt","Idle Date Start"],
    fuzzy_must=["idle"], fuzzy_any=["start","date"]
)
col_years_idle = _pick_col(
    df,
    ["Years Idle","Idle Years","YearsIdle","Years idle","Years_Idle","Yrs Idle","Years of Idle","Years Idle (yrs)"],
    fuzzy_must=["idle"], fuzzy_any=["years","yrs"]
)

# helper columns (silent)
if col_idle_start:
    df[col_idle_start] = pd.to_datetime(df[col_idle_start], errors="coerce")
    df["Year"] = df[col_idle_start].dt.year

if "Year" not in df.columns and col_years_idle:
    current_year = pd.Timestamp.now().year
    df[col_years_idle] = pd.to_numeric(df[col_years_idle], errors="coerce")
    df["Year"] = current_year - df[col_years_idle]

# ---------------------- Deterministic fallback logic --------------------------
def _oldest_idle_well_row(_df: pd.DataFrame):
    if col_idle_start and col_idle_start in _df.columns:
        tmp = _df.dropna(subset=[col_idle_start])
        if len(tmp):
            return tmp.sort_values(col_idle_start, ascending=True).iloc[0]
    if col_years_idle and col_years_idle in _df.columns:
        tmp = _df.copy()
        tmp[col_years_idle] = pd.to_numeric(tmp[col_years_idle], errors="coerce")
        tmp = tmp.dropna(subset=[col_years_idle])
        if len(tmp):
            return tmp.sort_values(col_years_idle, ascending=False).iloc[0]
    return None

def _summarize_row(r: pd.Series):
    return {
        (col_api or "API"): r.get(col_api) if col_api else None,
        (col_well or "Well"): r.get(col_well) if col_well else None,
        (col_operator or "Operator"): r.get(col_operator) if col_operator else None,
        (col_idle_start or "Idle Start Date"): str(r.get(col_idle_start)) if col_idle_start else None,
        (col_years_idle or "Years Idle"): r.get(col_years_idle) if col_years_idle else None,
        "Year": r.get("Year") if "Year" in df.columns else None,
    }

# ---------------------------- AI backend --------------------------------------
try:
    from pandasai import SmartDataframe
    from pandasai.llm import OpenAI as PAIOpenAI
    from openai import OpenAI as OAClient
    OPENAI_KEY = st.secrets["openai"]["api_key"]
except Exception:
    st.error("AI unavailable. Ensure OpenAI key is set in Streamlit Secrets and deps are installed.")
    st.stop()

domain_context = (
    "You are analyzing the California CalGEM Idle Well Inventory dataset. "
    "Treat all rows as idle wells unless explicitly filtered. "
    "Use only columns present in the dataframe. "
    "For 'oldest idle well', prefer the earliest 'Idle Start Date'; if absent, use maximum 'Years Idle'."
)
schema_hint = "Columns available: " + ", ".join(map(str, df.columns.tolist())) + "."

sdf = SmartDataframe(df, config={"llm": PAIOpenAI(api_token=OPENAI_KEY)})

# ---------------------------- Chat UI + Mic -----------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

st.caption("Tap to speak, or type below.")
audio = mic_recorder(
    start_prompt="🎤 Start recording",
    stop_prompt="■ Stop",
    just_once=False,
    use_container_width=True,
    key="mic",
)

user_msg = None

# Voice → text via Whisper
if audio and "bytes" in audio and audio["bytes"]:
    try:
        client = OAClient(api_key=OPENAI_KEY)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio["bytes"])
            tmp.flush()
            with open(tmp.name, "rb") as f:
                # Whisper transcription (new-style client)
                tr = client.audio.transcriptions.create(model="whisper-1", file=f)
        user_msg = tr.text if hasattr(tr, "text") else None
    except Exception as e:
        st.warning(f"Voice transcription failed; please type instead. ({e})")

if not user_msg:
    typed = st.chat_input("Type a question…")
    if typed:
        user_msg = typed

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
    except Exception:
        # Fallback for "oldest idle well" type questions
        text = user_msg.casefold()
        with st.chat_message("assistant"):
            if "oldest" in text and ("idle" in text or "well" in text):
                row = _oldest_idle_well_row(df)
                if row is not None:
                    st.markdown("**Oldest idle well (deterministic result):**")
                    st.json(_summarize_row(row))
                    st.session_state.history.append(("assistant", "Returned deterministic oldest idle well result."))
                else:
                    st.markdown("I couldn’t infer the oldest idle well from this file.")
                    st.session_state.history.append(("assistant", "No oldest idle well could be inferred."))
            else:
                st.markdown("Sorry, I couldn’t answer that.")
                st.session_state.history.append(("assistant", "Could not answer due to an internal error."))