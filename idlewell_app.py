# idlewell_app.py — Idle Well Inventory assistant (robust loader + mic)

from pathlib import Path
import re
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------- Page setup --------------------------------------
st.set_page_config(page_title="Idle Well Inventory Assistant", layout="wide")

# ---------------------------- Data paths --------------------------------------
REPO_DIR = Path(__file__).parent.resolve()
CANDIDATE_PATHS = [
    REPO_DIR / "data" / "2024_IWMP_Inventory_Public.xlsx",
    REPO_DIR / "2024_IWMP_Inventory_Public.xlsx",
]
DATA_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), None)
if DATA_PATH is None:
    st.error("Data not found in repo. Add the Excel to `data/` or repo root and redeploy.")
    st.stop()

# ---------------------------- Robust Excel reader -----------------------------
HEADER_CANDIDATES = tuple(range(0, 10))  # try first 10 rows as header

SHEET_PREFERENCE = ["inventory", "idle", "iwmp", "public", "sheet1"]

def pick_best_sheet(xl: pd.ExcelFile) -> str:
    names = xl.sheet_names
    # if only one, use it
    if len(names) == 1:
        return names[0]
    # prefer names containing keywords
    ranked = sorted(
        names,
        key=lambda n: (
            -sum(k in n.strip().lower() for k in SHEET_PREFERENCE),
            len(n)
        ),
    )
    return ranked[0]

def read_excel_robust(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet = pick_best_sheet(xl)
    last_err = None
    for h in HEADER_CANDIDATES:
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=h)
            if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
                return df
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise ValueError("Could not find a valid header row in the first 10 rows.")

try:
    df = read_excel_robust(DATA_PATH)
except Exception:
    st.error("Data could not be read. Please verify the Excel format.")
    st.stop()

# ---------------------------- Normalize columns -------------------------------
df = df.copy()
df.columns = [str(c).strip() for c in df.columns]

def _tokens(s: str):
    return re.findall(r"[A-Za-z0-9]+", s.casefold())

def _fuzzy_find_col(frame: pd.DataFrame, must_have=None, any_of=None):
    cols = [str(c).strip() for c in frame.columns]
    best, best_score = None, -1
    for c in cols:
        toks = _tokens(c)
        if must_have and not all(t.casefold() in toks for t in must_have):
            continue
        score = 0
        if any_of:
            score += sum(1 for t in any_of if t.casefold() in toks)
        score -= 0.01 * len(c)  # prefer shorter names when tied
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

# Broadened candidates
col_api        = _pick_col(df, ["API 10","API","API Number","API10","API Number (10)"], fuzzy_any=["api","number","10"])
col_well       = _pick_col(df, ["Well Designation","Well Name","Well","Well ID","Well No.","WellNumber"], fuzzy_any=["well","name","designation","id","no","number"])
col_operator   = _pick_col(df, ["Operator Name","Operator","Current Operator","Operator of Record"], fuzzy_any=["operator","name","record"])
col_idle_start = _pick_col(
    df,
    [
        "Idle Start Date","Idle Start","IdleStartDate","Date Idle Start",
        "Idle_Start_Date","Idle Start Dt","Idle Date Start","Idle Date","Date Idle",
        "Begin Idle Date","Initial Idle Date","Date Went Idle"
    ],
    fuzzy_must=["idle"], fuzzy_any=["start","date","begin","initial","went"]
)
col_years_idle = _pick_col(
    df,
    [
        "Years Idle","Idle Years","YearsIdle","Years idle","Years_Idle","Yrs Idle",
        "Years of Idle","Years Idle (yrs)","Idle Duration Years","IdleYears"
    ],
    fuzzy_must=["idle"], fuzzy_any=["years","yrs","duration"]
)

# ---------------------------- Helper columns ----------------------------------
def _coerce_idle_start(series: pd.Series) -> pd.Series:
    # Handle strings, datetimes, and Excel serial numbers
    s = series.copy()
    # Try parse strings first
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    # If still NaT and numeric-like, try Excel serial (days since 1899-12-30)
    needs_serial = parsed.isna() & pd.to_numeric(s, errors="coerce").notna()
    if needs_serial.any():
        nums = pd.to_numeric(s[needs_serial], errors="coerce")
        serial_dt = pd.to_datetime(nums, unit="D", origin="1899-12-30", errors="coerce")
        parsed.loc[needs_serial] = serial_dt
    return parsed

if col_idle_start:
    df[col_idle_start] = _coerce_idle_start(df[col_idle_start])
    if "Year" not in df.columns:
        df["Year"] = df[col_idle_start].dt.year

if "Year" not in df.columns and col_years_idle:
    current_year = pd.Timestamp.now().year
    df[col_years_idle] = pd.to_numeric(df[col_years_idle], errors="coerce")
    df["Year"] = current_year - df[col_years_idle]

# ---------------------- Deterministic fallback logic --------------------------
def _oldest_idle_well_row(_df: pd.DataFrame):
    # prefer earliest Idle Start Date
    if col_idle_start and col_idle_start in _df.columns:
        tmp = _df.dropna(subset=[col_idle_start])
        if len(tmp):
            return tmp.sort_values(col_idle_start, ascending=True).iloc[0]
    # else max Years Idle
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
    st.error("AI unavailable. Ensure OpenAI key is set in Streamlit Secrets and dependencies are installed.")
    st.stop()

domain_context = (
    "You are analyzing the California CalGEM Idle Well Inventory dataset. "
    "Treat all rows as idle wells unless explicitly filtered. "
    "Use only columns present in the dataframe. "
    "For 'oldest idle well', prefer the earliest 'Idle Start Date'; if absent, use maximum 'Years Idle'."
)
schema_hint = "Columns available: " + ", ".join(map(str, df.columns.tolist())) + "."
sdf = SmartDataframe(df, config={"llm": PAIOpenAI(api_token=OPENAI_KEY)})

# ---------------------------- Mic (quiet) -------------------------------------
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_OK = True
except Exception:
    MIC_OK = False

# ---------------------------- Minimal UI --------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_text" not in st.session_state:
    st.session_state.pending_text = ""

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

audio = None
if MIC_OK:
    audio = mic_recorder(
        start_prompt="🎤 Speak",
        stop_prompt="■ Stop",
        just_once=False,
        use_container_width=True,
        key="mic",
    )

# If voice provided, transcribe silently to the text box
if audio and "bytes" in audio and audio["bytes"]:
    try:
        client = OAClient(api_key=OPENAI_KEY)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio["bytes"])
            tmp.flush()
            with open(tmp.name, "rb") as f:
                tr = client.audio.transcriptions.create(model="whisper-1", file=f)
        if hasattr(tr, "text") and tr.text:
            st.session_state.pending_text = tr.text
    except Exception:
        pass  # stay silent; user can still type

# Single compact input area (pre-filled by mic transcript if present)
prompt = st.text_input(
    "Ask about the Idle Well Inventory…",
    value=st.session_state.pending_text,
    label_visibility="collapsed",
    placeholder="e.g., What is the oldest idle well and who is the operator?"
)
confirm_ok = st.checkbox("✔️ Transcription looks correct (or I typed my question)", value=bool(prompt))

col_run, col_clear = st.columns([1,1])
with col_run:
    run_clicked = st.button("Ask")
with col_clear:
    if st.button("Clear"):
        st.session_state.history = []
        st.session_state.pending_text = ""
        st.experimental_rerun()

# Handle query
if run_clicked and confirm_ok and prompt.strip():
    user_msg = prompt.strip()
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    try:
        enriched = f"{domain_context}\n{schema_hint}\n\nQuestion: {user_msg}"
        with st.chat_message("assistant"):
            answer = sdf.chat(enriched)
            st.markdown(str(answer))
            st.session_state.history.append(("assistant", str(answer)))
    except Exception:
        # Silent fallback for "oldest idle well" type
        text = user_msg.casefold()
        with st.chat_message("assistant"):
            if "oldest" in text and ("idle" in text or "well" in text):
                row = _oldest_idle_well_row(df)
                if row is not None:
                    result = _summarize_row(row)
                    st.markdown(
                        f"**Oldest idle well (deterministic):**\n\n"
                        f"- API: `{result.get(col_api or 'API')}`\n"
                        f"- Well: `{result.get(col_well or 'Well')}`\n"
                        f"- Operator: `{result.get(col_operator or 'Operator')}`\n"
                        f"- Idle Start Date: `{result.get(col_idle_start or 'Idle Start Date')}`\n"
                        f"- Years Idle: `{result.get(col_years_idle or 'Years Idle')}`"
                    )
                    st.session_state.history.append(("assistant", "Returned deterministic oldest idle well result."))
                else:
                    st.markdown("No idle wells detected from this file.")
                    st.session_state.history.append(("assistant", "No deterministic result."))
            else:
                st.markdown("I couldn’t answer that.")
                st.session_state.history.append(("assistant", "Unhandled error."))