import subprocess, time, requests, os, signal

def test_health_and_predict():
    # Lancer l'API localement (uvicorn)
    proc = subprocess.Popen(["uvicorn", "api.main:app", "--port", "8001"])
    time.sleep(3.0)

    try:
        r = requests.get("http://127.0.0.1:8001/health")
        assert r.status_code == 200

        payload = {
            "sepal_length": 5.1, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2
        }
        rr = requests.post("http://127.0.0.1:8001/predict", json=payload)
        assert rr.status_code == 200
        assert "label" in rr.json()
    finally:
        os.kill(proc.pid, signal.SIGTERM)
