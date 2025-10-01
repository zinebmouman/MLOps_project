import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Music Recommender 🎵", page_icon="🎧", layout="centered")
st.title("🎶 Instant Music Recommender")

@st.cache_data(ttl=300)
def fetch_songs(q: str) -> list[str]:
    """Appelé uniquement si l'utilisateur clique sur 'Search suggestions'."""
    try:
        r = requests.get(f"{API_URL}/songs", params={"q": q, "limit": 1000}, timeout=20)
        r.raise_for_status()
        return r.json().get("items", [])
    except Exception:
        return []

def recommend(song: str, k: int = 5) -> pd.DataFrame | None:
    r = requests.get(f"{API_URL}/recommend", params={"song": song, "top_n": k}, timeout=30)
    r.raise_for_status()
    js = r.json()
    if not js.get("found"):
        return None
    return pd.DataFrame(js["recommendations"])

# ---- UI ----
q = st.text_input("🎵 Type a song title", value="", placeholder="Type at least 2 letters")

# Ne FETCH RIEN au chargement : seulement si on clique le bouton
suggestions = []
if st.button("🔎 Search suggestions"):
    q_clean = q.strip()
    if len(q_clean) < 2:
        st.info("Type at least 2 characters, then click Search.")
    else:
        with st.spinner("Searching suggestions…"):
            suggestions = fetch_songs(q_clean)
            if not suggestions:
                st.warning("No suggestions found. You can still request recommendations with the typed title.")

selected = None
if suggestions:
    selected = st.selectbox("Suggestions (optional)", suggestions, index=0)

# Quand on recommande, on n'appelle PAS /songs : on prend la sélection OU le texte saisi
song_input = (selected or q.strip())

if st.button("🚀 Recommend Similar Songs"):
    if not song_input:
        st.warning("Please type a song title (or pick a suggestion).")
    else:
        with st.spinner("Finding similar songs…"):
            try:
                df = recommend(song_input, 5)
                if df is None or df.empty:
                    st.warning(f"Sorry, song '{song_input}' not found.")
                else:
                    st.success(f"Top similar songs for: {song_input}")
                    st.table(df)
            except requests.RequestException as e:
                st.error(f"API error: {e}")
