import streamlit as st
import pandas as pd
import subprocess
import io
from datetime import datetime
from typing import Dict, Any, List

# --- Optional deps for map & mic ---
try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC = True
except Exception:
    HAS_MIC = False

try:
    import speech_recognition as sr
    HAS_SR = True
except Exception:
    HAS_SR = False

# =========================
# CONFIG & THEME (Dark Only)
# =========================
st.set_page_config(page_title="KhetAI – Farmer Assistant", layout="wide")

bg = "#0f172a"
text = "#f1f5f9"
card = "#1e293b"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@400;600&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Inter', 'Poppins', sans-serif;
        color: {text};
        background: {bg};
    }}
    h1 {{
        font-family: 'Poppins', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: {text} !important;
        margin-bottom: 0 !important;
    }}
    .subheader {{
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: .3rem;
    }}
    .card {{
        background: {card};
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.1);
    }}
    .user-question {{
        padding: 14px 16px;
        border-radius: 16px;
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: #fff;
        font-weight: 500;
        box-shadow: 0 6px 14px rgba(37,99,235,.18);
        margin-bottom: 10px;
        max-width: 75%;
    }}
    .assistant-answer {{
        padding: 14px 16px;
        border-radius: 16px;
        background: {card};
        border-left: 5px solid #22c55e;
        box-shadow: 0 10px 18px rgba(2,6,23,.15);
        color: {text};
        line-height: 1.7;
        margin-bottom: 24px;
        max-width: 85%;
        white-space: pre-wrap;
    }}
    .history-box {{
        background: {card};
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 12px;
        color: #94a3b8;
    }}
    .stTextInput > div > div > input {{
        border-radius: 14px !important;
        border: 1.5px solid #475569 !important;
        padding: 14px 16px !important;
        background: {card};
        color: {text};
        font-size: 15px;
        box-shadow: 0 0 10px rgba(37,99,235,.25);
    }}
    .stButton button {{
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        border: 0;
        color: #fff;
        font-weight: 600;
        border-radius: 12px;
        padding: .6rem 1.1rem;
        box-shadow: 0 8px 16px rgba(37,99,235,.2);
    }}
    .stButton button:hover {{
        filter: brightness(1.05);
    }}
    /* Inline mic + input styling */
    .mic-btn {{
        background-color: #2563eb;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        cursor: pointer;
        font-size: 20px;
        margin-right: 10px;
    }}
    .mic-btn:hover {{
        filter: brightness(1.1);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(name)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_all_data() -> Dict[str, pd.DataFrame]:
    return {
        "villages": load_csv("Villages.csv"),
        "soil": load_csv("SoilHealth.csv"),
        "cropcal": load_csv("CropCalendars.csv"),
        "irrig": load_csv("IrrigationAdvisory.csv"),
        "market": load_csv("MarketPrices.csv"),
        "schemes": load_csv("GovernmentSchemes.csv"),
        "pests": load_csv("PestsDiseases.csv"),
        "weathernorm": load_csv("WeatherNormals.csv"),
    }

DATA = load_all_data()
df = DATA["villages"]
if df.empty:
    st.error("Villages.csv could not be loaded. Please place it in the app directory.")
    st.stop()

# =========================
# MODEL HELPERS with letter-wise streaming
# =========================
def llama_local_stream(prompt: str, model: str = "llama2"):
    proc = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
    )
    try:
        proc.stdin.write(prompt)
        proc.stdin.flush()
        proc.stdin.close()
    except Exception as e:
        yield f"⚠️ Error sending prompt: {e}"
        return

    output = ""
    while True:
        char = proc.stdout.read(1)
        if not char:
            break
        output += char
        yield output

    proc.stdout.close()
    proc.wait()

# Expanded category list (single-label)
CATEGORIES = [
    "weather", "irrigation", "irrigation scheduling", "water quality",
    "seed variety", "crop planning", "sowing", "harvest",
    "soil health", "soil testing", "fertilizers", "organic farming",
    "pest management", "disease management", "weed management",
    "pesticides", "market prices", "post-harvest", "storage", "logistics",
    "finance and loans", "kisan credit", "government schemes", "subsidies",
    "machinery", "farm mechanization", "precision ag", "iotsensors",
    "insurance", "crop insurance", "transport", "export",
    "climate change", "risk alerts", "other"
]

def classify_question(question: str) -> str:
    cat_prompt = f"""
You are an agricultural assistant.
Pick ONE best-fit category from this list (exact text):
{", ".join(CATEGORIES)}.

Question: {question}

Reply with ONLY the category name (lowercase), nothing else.
"""
    out = ""
    try:
        result = subprocess.run(
            ["ollama", "run", "llama2"],
            input=cat_prompt,
            capture_output=True,
            text=True,
            check=True
        )
        out = result.stdout.strip().lower()
    except Exception:
        out = "other"
    return out if out in CATEGORIES else "other"

# =========================
# VOICE (Speech-to-Text)
# =========================
def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    if not HAS_SR:
        return ""
    r = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = r.record(source)
        return r.recognize_google(audio)
    except Exception:
        return ""

# =========================
# CONTEXT BUILDERS (by Pincode)
# =========================
def context_for_pincode(pincode: int, state: str) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}

    soil_df = DATA["soil"]
    if not soil_df.empty:
        soil_row = soil_df[soil_df["Pincode"] == pincode]
        if not soil_row.empty:
            ctx["soil"] = soil_row.iloc[0].to_dict()

    wn = DATA["weathernorm"]
    if not wn.empty:
        mon = datetime.now().strftime("%b")
        wn_row = wn[(wn["Pincode"] == pincode) & (wn["Month"] == mon)]
        if not wn_row.empty:
            ctx["weathernorm"] = wn_row.iloc[0].to_dict()

    mkt = DATA["market"]
    if not mkt.empty:
        rows = mkt[mkt["PincodeServiceArea"] == pincode]
        if rows.empty:
            rows = mkt[mkt["State"].str.lower() == state.lower()]
        rows = rows.sort_values("Date", ascending=False).head(3)
        ctx["market"] = rows.to_dict(orient="records")

    sch = DATA["schemes"]
    if not sch.empty:
        sch_rows = sch[sch["State"].str.lower() == state.lower()]
        ctx["schemes"] = sch_rows.to_dict(orient="records")[:5]

    cc = DATA["cropcal"]
    if not cc.empty:
        cc_rows = cc[cc["State"].str.lower() == state.lower()]
        ctx["cropcal"] = cc_rows.to_dict(orient="records")[:5]

    pdx = DATA["pests"]
    if not pdx.empty:
        pdx_rows = pdx[pdx["State"].str.lower() == state.lower()]
        ctx["pests"] = pdx_rows.to_dict(orient="records")[:5]

    return ctx

def irrigation_snippets(crop: str | None, soil_type: str | None) -> List[Dict[str, Any]]:
    irr = DATA["irrig"]
    if irr.empty:
        return []
    q = irr.copy()
    if crop:
        q = q[q["Crop"].str.lower() == crop.lower()]
    if soil_type:
        q = q[q["SoilType"].str.lower() == soil_type.lower()]
    if q.empty and crop:
        q = irr[irr["Crop"].str.lower() == crop.lower()]
    return q.head(5).to_dict(orient="records")

def build_llm_context(category: str, location_meta: Dict[str, Any], user_q: str) -> str:
    state = location_meta["state"]
    pincode = location_meta["pincode"]
    ctx = context_for_pincode(pincode, state)

    possible_crops = set([
        "paddy", "rice", "cotton", "wheat", "ragi", "groundnut", "maize", "sugarcane", "tur", "chilli", "soybean"
    ])
    crop_mentioned = None
    for tok in user_q.lower().replace(",", " ").split():
        if tok in possible_crops:
            crop_mentioned = "rice" if tok == "paddy" else tok
            break

    soil_type = ctx.get("soil", {}).get("SoilType")
    irr = irrigation_snippets(crop_mentioned, soil_type)

    parts = [
        f"[Category: {category}]",
        f"[Location: State={location_meta['state']}, District={location_meta['district']}, Mandal={location_meta['mandal']}, "
        f"Pincode={location_meta['pincode']}, Lat={location_meta['lat']}, Lon={location_meta['lon']}]",
    ]

    if ctx.get("soil"):
        s = ctx["soil"]
        parts.append(
            f"[Soil: Type={s.get('SoilType')}, pH={s.get('pH')}, N={s.get('N_kg_ha')} kg/ha, P={s.get('P_kg_ha')} kg/ha, "
            f"K={s.get('K_kg_ha')} kg/ha, OC={s.get('OrganicCarbon_%')}%, EC={s.get('EC_dS_m')} dS/m]"
        )

    if ctx.get("weathernorm"):
        w = ctx["weathernorm"]
        parts.append(
            f"[WeatherNormals: Month={w.get('Month')}, NormalRain={w.get('NormalRain_mm')} mm, NormalTemp={w.get('NormalTemp_C')} °C]"
        )

    if ctx.get("market"):
        mk_lines = []
        for r in ctx["market"]:
            mk_lines.append(
                f"{r['Date']}: {r['Market']} – {r['Commodity']} ({r['Variety']}), {r['Unit']} modal ₹{r['ModalPriceINR']}"
            )
        parts.append("[MarketPrices:\n  " + "\n  ".join(mk_lines) + "\n]")

    if ctx.get("schemes"):
        sc_lines = [f"{r['Scheme']} – {r['Benefit']} (Apply: {r['HowToApply']})" for r in ctx["schemes"]]
        parts.append("[Schemes:\n  " + "\n  ".join(sc_lines) + "\n]")

    if ctx.get("cropcal"):
        cc_lines = [f"{r['Crop']}: Sowing {r['SowingWindow']}, Harvest {r['HarvestWindow']}, Varieties: {r['RecommendedVarieties']}" for r in ctx["cropcal"]]
        parts.append("[CropCalendar:\n  " + "\n  ".join(cc_lines) + "\n]")

    if ctx.get("pests"):
        px_lines = [f"{r['Crop']}: {r['PestDisease']} – {r['Symptoms']}. Advice: {r['Advisory']}" for r in ctx["pests"]]
        parts.append("[Pest&Disease:\n  " + "\n  ".join(px_lines) + "\n]")

    if irr:
        ir_lines = [f"{r['Crop']} ({r['SoilType']}–{r['Stage']}): ETc {r['ETc_mm_day']} mm/day, interval {r['IrrigationInterval_days']} d. {r['Note']}" for r in irr]
        parts.append("[Irrigation:\n  " + "\n  ".join(ir_lines) + "\n]")

    parts.append(f"[User Question]\n{user_q}\n")
    parts.append("Provide a detailed, accurate, and location-specific answer for the farmer. Cite data you used from the context when relevant.")
    return "\n".join(parts)

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_voice_text" not in st.session_state:
    st.session_state.pending_voice_text = ""

# =========================
# HEADER
# =========================
header_l, header_r = st.columns([1, 8], vertical_alignment="center")
with header_l:
    st.image("khetai.png", width=82)
with header_r:
    st.markdown("<h1>KhetAI – Smart Farming Assistant</h1>", unsafe_allow_html=True)
    st.markdown('<div class="subheader">Accurate, location-aware advisory for farmers</div>', unsafe_allow_html=True)

st.markdown("---")

# =========================
# MAIN LAYOUT
# =========================
left, right = st.columns([2, 3])

# ---------- LEFT: LOCATION ----------
with left:
    st.markdown("### Select Location")
    selected_pincode = st.selectbox("Pincode", sorted(df["Pincode"].unique()))
    record = df[df["Pincode"] == selected_pincode].iloc[0]
    state, district, mandal = record["State"], record["District"], record["Mandal"]
    lat, lon = record["Latitude"], record["Longitude"]

    st.markdown(
        f"""
        <div class="card" style="margin-top:8px;">
          <div><b>State:</b> {state}</div>
          <div><b>District:</b> {district}</div>
          <div><b>Mandal:</b> {mandal}</div>
          <div><b>Pincode:</b> {selected_pincode}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Map")
    if HAS_FOLIUM:
        m = folium.Map(location=[lat, lon], zoom_start=10)
        popup_text = f"{state}, {district}, {mandal}<br>Pincode: {selected_pincode}"
        folium.Marker([lat, lon], popup=popup_text).add_to(m)
        st_folium(m, height=280, use_container_width=True)
    else:
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=10)

# ---------- RIGHT: CHAT PANEL ----------
with right:
    st.markdown("### 💬 Chat With Assistant")

    # Mic + Input
    input_col1, input_col2 = st.columns([1, 8])
    if HAS_MIC:
        with input_col1:
            audio = mic_recorder(
                start_prompt="🎙️",
                stop_prompt="⏹️",
                just_once=True,
                use_container_width=True,
                format="wav"
            )
            if audio and "bytes" in audio:
                transcript = transcribe_audio_bytes(audio["bytes"])
                if transcript:
                    st.session_state.pending_voice_text = transcript
                    st.success(f"Voice captured: {transcript}")
                else:
                    st.warning("Couldn't understand the audio. Try again.")

    with input_col2:
        query = st.text_input(
            "Your Question",
            value=st.session_state.pending_voice_text,
            placeholder="Type or speak your farming question...",
            label_visibility="collapsed",
            key="query_input"
        )

    get_ans = st.button(
        "Ask KhetAI",
        use_container_width=True,
        key="get_answer_btn",
        help="Click to get answer",
    )

    if get_ans:
        text_now = st.session_state.get("query_input", "").strip()
        if text_now:
            with st.spinner("Analyzing your question…"):
                category = classify_question(text_now)
                location_meta = {
                    "state": state,
                    "district": district,
                    "mandal": mandal,
                    "pincode": int(selected_pincode),
                    "lat": float(lat),
                    "lon": float(lon),
                }
                full_prompt = build_llm_context(category, location_meta, text_now)

                output_placeholder = st.empty()
                answer_text = ""

                for partial_answer in llama_local_stream(full_prompt):
                    answer_text = partial_answer
                    output_placeholder.markdown(
                        f'<div class="assistant-answer"><b>Assistant:</b><br>{answer_text}</div>',
                        unsafe_allow_html=True,
                    )

                st.session_state.history.append({"q": text_now, "a": answer_text, "cat": category})
                st.session_state.pending_voice_text = ""
        else:
            st.warning("Please enter a question or use voice input.")

    # Display latest conversation
    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end;">
                <div class="user-question">
                    <b>You:</b><br>{latest['q']}
                    <div style="opacity:.7; font-size:.8rem; margin-top:4px;">
                        Category: {latest['cat'].capitalize()}
                    </div>
                </div>
            </div>
            <div style="display:flex; justify-content:flex-start;">
                <div class="assistant-answer">
                    <b>Assistant:</b><br>{latest['a']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------- BOTTOM: CONTEXT PREVIEW ----------
st.markdown("## 📚 Context Snapshot (from datasets)")
with st.expander("Show context used for current pincode"):
    pin = int(selected_pincode)
    ctx = context_for_pincode(pin, state)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Soil**")
        df_soil = pd.DataFrame([ctx.get("soil", {})])
        st.write(df_soil if not df_soil.empty and df_soil.notna().any().any() else "—")

        st.markdown("**Weather Normal (this month)**")
        df_wn = pd.DataFrame([ctx.get("weathernorm", {})])
        st.write(df_wn if not df_wn.empty and df_wn.notna().any().any() else "—")

        st.markdown("**Irrigation (matched to crop/soil if detected)**")
        df_irrig = pd.DataFrame(irrigation_snippets(None, ctx.get("soil", {}).get("SoilType", None)))
        st.write(df_irrig if not df_irrig.empty else "—")
    with colB:
        st.markdown("**Market (latest)**")
        df_market = pd.DataFrame(ctx.get("market", []))
        st.write(df_market if not df_market.empty else "—")

        st.markdown("**Schemes (state)**")
        df_schemes = pd.DataFrame(ctx.get("schemes", []))
        st.write(df_schemes if not df_schemes.empty else "—")

        st.markdown("**Pests/Diseases (state)**")
        df_pests = pd.DataFrame(ctx.get("pests", []))
        st.write(df_pests if not df_pests.empty else "—")

        st.markdown("**Crop Calendar (state)**")
        df_cropcal = pd.DataFrame(ctx.get("cropcal", []))
        st.write(df_cropcal if not df_cropcal.empty else "—")

# ---------- BOTTOM: HISTORY ----------
st.markdown("## 📜 Conversation History")
if st.session_state.history:
    st.markdown(
        """
        <style>
        .history-scroll {
            max-height: 300px;
            overflow-y: auto;
            padding-right: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='history-scroll'>", unsafe_allow_html=True)
    for entry in reversed(st.session_state.history):
        st.markdown(
            f"""
            <div class='history-box'>
                <b>You:</b> {entry['q']}<br>
                <b>Assistant:</b> {entry['a']}
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No past questions yet.")
