import os, requests, pandas as pd, streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Music Recommender 🎵", page_icon="🎧", layout="centered")
st.title("🎶 Instant Music Recommender")

@st.cache_data(ttl=300)
def fetch_songs(q: str) -> list[str]:
    try:
        r = requests.get(f"{API_URL}/songs", params={"q": q, "limit": 1000}, timeout=20)
        r.raise_for_status()
        return r.json().get("items", [])
    except Exception:
        return []

# Le select reprend le look de ta capture : un champ de recherche puis la liste
q = st.text_input("🎵 Select a song:", value="love")
options = fetch_songs(q) if q else []
selected = st.selectbox("", options, index=0 if options else None, label_visibility="collapsed")

def recommend(song: str, k: int = 5) -> pd.DataFrame | None:
    r = requests.get(f"{API_URL}/recommend", params={"song": song, "top_n": k}, timeout=30)
    r.raise_for_status()
    js = r.json()
    if not js.get("found"):
        return None
    return pd.DataFrame(js["recommendations"])

if st.button("🚀 Recommend Similar Songs"):
    with st.spinner("Finding similar songs..."):
        if not selected:
            st.warning("Please select a song.")
        else:
            df = recommend(selected, 5)
            if df is None or df.empty:
                st.warning("Sorry, song not found.")
            else:
                st.success("Top similar songs:")
                st.table(df)
