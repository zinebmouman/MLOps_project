from src.preprocess import preprocess_text_factory, ensure_nltk, _sha256
import types
import pathlib

def test_preprocess_text_factory_basic(monkeypatch):
    # remplace TOUT l’objet stopwords par un faux
    fake_sw = types.SimpleNamespace(words=lambda _lang: ["the", "and"])
    monkeypatch.setattr("src.preprocess.stopwords", fake_sw, raising=True)
    # tokenization light
    monkeypatch.setattr("src.preprocess.word_tokenize", lambda s: s.split())

    cleaner = preprocess_text_factory()
    assert cleaner("Hello, THE-world!! and music.") == "hello world music"

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
