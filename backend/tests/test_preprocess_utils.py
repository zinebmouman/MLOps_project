from src.preprocess import preprocess_text_factory, ensure_nltk, _sha256
import types
import pathlib

def preprocess_text_factory():
    stop_words = set(stopwords.words('english'))
    # remplacer tous les non-lettres par des espaces
    non_letters = re.compile(r"[^A-Za-z]+")
    def _clean(text: str) -> str:
        text = non_letters.sub(" ", str(text)).lower()
        # compacter les espaces multiples
        text = re.sub(r"\s+", " ", text).strip()
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w and w not in stop_words]
        return " ".join(tokens)
    return _clean


def test_sha256_roundtrip(tmp_path: pathlib.Path):
    p = tmp_path / "f.txt"
    p.write_text("abc", encoding="utf-8")
    h = _sha256(p)
    assert len(h) == 64

def test_ensure_nltk_downloads(monkeypatch):
    calls = []
    def _find(_): raise LookupError
    def _download(pkg): calls.append(pkg)
    monkeypatch.setattr("src.preprocess.nltk.data.find", _find)
    monkeypatch.setattr("src.preprocess.nltk.download", _download)
    ensure_nltk()
    assert any(pkg in calls for pkg in ["punkt", "punkt_tab", "stopwords"])
