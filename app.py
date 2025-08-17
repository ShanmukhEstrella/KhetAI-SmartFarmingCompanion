# app.py
import streamlit as st
import pandas as pd
import subprocess
import io

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
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("Villages.csv")

df = load_data()


# =========================
# OLLAMA / LLAMA HELPERS
# =========================
def llama_local(prompt: str, model: str = "llama2") -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"⚠️ Error connecting to model: {e}"

def classify_question(question: str) -> str:
    cat_prompt = f"""
You are an agricultural assistant.
Classify the farmer's question into ONE category from this list:
weather, irrigation, seed variety, soil health, fertilizers,
pesticides, market prices, finance and loans, government schemes,
machinery, storage, transport, export, insurance, climate change, other.

Question: {question}
Reply with ONLY the category name (lowercase).
"""
    return llama_local(cat_prompt).strip().lower()


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
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []


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

    # Voice input
    voice_text = ""
    if HAS_MIC:
        audio = mic_recorder(
            start_prompt="🎙️ Speak",
            stop_prompt="⏹️ Stop",
            just_once=True,
            use_container_width=True,
            format="wav"
        )
        if audio and "bytes" in audio:
            transcript = transcribe_audio_bytes(audio["bytes"])
            if transcript:
                st.success(f"Voice captured: {transcript}")
                voice_text = transcript
            else:
                st.warning("Couldn't understand the audio. Try again.")

    # Chat input box
    query = st.text_input(
        "Question",
        value=voice_text or "",
        placeholder="Type or speak your farming question...",
        label_visibility="collapsed"
    )

    get_ans = st.button(
    "Ask KhetAI",
    use_container_width=True,
    key="get_answer_btn",
    help="Click to get answer",
    )

    st.markdown(
        """
        <style>
        /* Force styling only for Ask KhetAI button */
        button[kind="secondary"][key="get_answer_btn"],
        div.stButton > button {
            background-color: #0a174e !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 10px 20px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
        }

        button[kind="secondary"][key="get_answer_btn"]:hover,
        div.stButton > button:hover {
            background-color: #112d91 !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



    if get_ans:
        if query.strip():
            with st.spinner("Analyzing your question…"):
                category = classify_question(query)
                full_prompt = (
                    f"[Category: {category}]\n"
                    f"[Location: State={state}, District={district}, Mandal={mandal}, "
                    f"Pincode={selected_pincode}, Lat={lat}, Lon={lon}]\n"
                    f"User Question: {query}\n\n"
                    f"Provide a detailed, accurate, and location-specific answer for the farmer."
                )
                answer = llama_local(full_prompt)

                st.session_state.history.append({"q": query, "a": answer, "cat": category})
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
    for entry in st.session_state.history:
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
