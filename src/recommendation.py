import joblib
from pathlib import Path
from typing import Optional
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

_df: Optional[pd.DataFrame] = None
_cos: Optional[any] = None

def _load():
    global _df, _cos
    if _df is None:
        _df = joblib.load(MODELS / 'df_cleaned.pkl')
    if _cos is None:
        _cos = joblib.load(MODELS / 'cosine_sim.pkl')
    return _df, _cos

def recommend_songs(song_name: str, top_n: int = 5) -> pd.DataFrame | None:
    df, cosine_sim = _load()
    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        return None
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]
    out = df[['artist','song']].iloc[song_indices].reset_index(drop=True)
    out.index = out.index + 1
    out.index.name = "rank"
    return out
