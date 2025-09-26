import os, requests, pandas as pd, streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Music Recommender 🎵", page_icon="🎧", layout="centered")
st.title("🎶 Instant Music Recommender")

with st.sidebar:
    st.markdown("**Backend API**")
    api_url = st.text_input("API_URL", value=API_URL)
    st.link_button("Swagger /docs", f"{api_url}/docs")
    st.link_button("Metrics", f"{api_url}/metrics")
    st.link_button("MLflow", f"{api_url}/mlflow")

@st.cache_data(ttl=300)
def fetch_songs(prefix: str) -> list[str]:
    try:
        r = requests.get(f"{api_url}/songs", params={"q": prefix, "limit": 300}, timeout=20)
        r.raise_for_status()
        return r.json().get("items", [])
    except Exception:
        return []

def recommend(song: str, top_n: int = 5) -> pd.DataFrame | None:
    r = requests.get(f"{api_url}/recommend", params={"song": song, "top_n": top_n}, timeout=30)
    r.raise_for_status()
    js = r.json()
    if not js.get("found"):
        return None
    return pd.DataFrame(js["recommendations"])

# Recherche + sélection
query = st.text_input("🔎 Rechercher une chanson", value="love")
suggestions = fetch_songs(query) if query else []
chosen = st.selectbox("🎵 Choisis une chanson", suggestions) if suggestions else st.text_input("🎵 Chanson (texte libre)")

col1, col2 = st.columns([1,1])
with col1:
    top_n = st.slider("Nombre de recommandations", min_value=3, max_value=20, value=5)
with col2:
    go = st.button("🚀 Recommander")

if go:
    with st.spinner("Calcul en cours..."):
        df = recommend(chosen, top_n=top_n)
        if df is None or df.empty:
            st.warning("Aucune recommandation trouvée (ou chanson inconnue).")
        else:
            st.success("Voici les titres similaires :")
            st.dataframe(df, use_container_width=True)

# Health
try:
    health = requests.get(f"{api_url}/health", timeout=5).json()
    st.caption(f"Health: {health.get('status', 'unknown')}")
except Exception:
    st.caption("Health: unreachable")
