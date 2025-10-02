from fastapi.testclient import TestClient
from api.main import app

def test_recommend_found():
    c = TestClient(app)
    r = c.get("/recommend", params={"song": "Love", "top_n": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert isinstance(body["recommendations"], list)
    assert len(body["recommendations"]) >= 1


