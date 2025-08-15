import streamlit as st
import pandas as pd
import ollama  # using Ollama Python API for streaming

CATEGORIES = [
    "weather",
    "irrigation",
    "seed variety",
    "market prices",
    "finance and loans",
    "government schemes",
    "other"
]

@st.cache_data
def load_data():
    return pd.read_csv("Villages.csv")

def llama_local_stream(prompt, model="llama2"):
    """Run local LLaMA via Ollama API with streaming output."""
    output_placeholder = st.empty()
    full_response = ""

    # Stream partial responses from Ollama
    for chunk in ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ):
        if "message" in chunk and "content" in chunk["message"]:
            token = chunk["message"]["content"]
            full_response += token
            output_placeholder.markdown(f"### 📌 Answer:\n{full_response}")

    return full_response.strip()

def classify_question(text):
    category_prompt = f"""
    You are an agricultural assistant.
    Classify the following question into one of these categories:
    {', '.join(CATEGORIES)}.
    Question: {text}
    Reply with ONLY the category name.
    """
    # No need for streaming here, just get the category quickly
    resp = ollama.chat(
        model="llama2",
        messages=[{"role": "user", "content": category_prompt}]
    )
    return resp["message"]["content"].strip().lower()

def generate_answer(category, question, record):
    lat, lon = record["Latitude"], record["Longitude"]
    prompt = f"""
    You are an agricultural assistant.
    Location: State={record['State']}, District={record['District']}, Mandal={record['Mandal']}
    Latitude={lat}, Longitude={lon}
    Question Category: {category}
    User Question: {question}
    Provide a detailed, accurate, and location-specific answer for the farmer.
    """
    return llama_local_stream(prompt)

def main():
    st.set_page_config(page_title="KhetAI", layout="wide")
    col1, col2 = st.columns([10, 1])
    with col1:
        st.markdown("## 🌾KhetAI")
    with col2:
        st.image("khetai.png", width=100)
    st.markdown("---")

    df = load_data().dropna(subset=["State", "District", "Mandal", "Latitude", "Longitude"]).drop_duplicates()

    with st.sidebar:
        st.header("🌐 Select Location")
        selected_State = st.selectbox("Select State", sorted(df["State"].unique()))
        df1 = df[df["State"] == selected_State]
        selected_region = st.selectbox("Select District", sorted(df1["District"].unique()))
        df2 = df1[df1["District"] == selected_region]
        selected_Mandal = st.selectbox("Select Mandal", sorted(df2["Mandal"].unique()))
        record = df2[df2["Mandal"] == selected_Mandal].iloc[0]

    st.title("Your Farming Companion")
    st.markdown("This is your smart assistant. Ask any question related to the selected region.")
    st.success(f"🧭 **{selected_State} → {selected_region} → {selected_Mandal}**")

    question = st.text_input("💬 Ask your question:")

    if question:
        with st.spinner("🔍 Understanding your question..."):
            category = classify_question(question)
            st.markdown(f"🧠 *Detected Category:* `{category}`")
            generate_answer(category, question, record)  # Streams output live

if __name__ == "__main__":
    main()
