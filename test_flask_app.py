import io
import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from app import app, get_recent_history, DB_PATH

def run_tests():
    client = app.test_client()

    print("[*] 1. Testing GET / (HTML Interface)...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    html = res.data.decode("utf-8")
    assert "Iris Classification System" in html
    assert "Batch CSV Classification" in html
    assert "Prediction History Log" in html
    print("    -> Passed! All UI sections rendered.")

    print("[*] 2. Testing POST /predict (Form submission + SQLite logging)...")
    res = client.post("/predict", data={
        "sepal_length": "5.1",
        "sepal_width": "3.5",
        "petal_length": "1.4",
        "petal_width": "0.2"
    })
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Iris setosa" in html
    
    # Check SQLite database logging
    history = get_recent_history()
    assert len(history) > 0, "Expected at least 1 record in history database"
    assert history[0]["species"] == "setosa"
    print(f"    -> Passed! Classified as Setosa and logged to SQLite (DB: {os.path.basename(DB_PATH)}).")

    print("[*] 3. Testing GET /sample-csv (Template download)...")
    res = client.get("/sample-csv")
    assert res.status_code == 200
    assert "sepal_length,sepal_width,petal_length,petal_width" in res.data.decode("utf-8")
    print("    -> Passed! Sample CSV generated successfully.")

    print("[*] 4. Testing POST /batch-predict (CSV batch processing)...")
    sample_csv_content = b"""sepal_length,sepal_width,petal_length,petal_width
5.1,3.5,1.4,0.2
6.5,3.0,5.5,1.8
5.9,3.0,4.2,1.5"""
    data = {
        "file": (io.BytesIO(sample_csv_content), "test_flowers.csv")
    }
    res = client.post("/batch-predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    df_out = pd.read_csv(io.StringIO(res.data.decode("utf-8")))
    assert "predicted_species" in df_out.columns
    assert "confidence" in df_out.columns
    assert len(df_out) == 3
    assert df_out["predicted_species"].tolist() == ["setosa", "virginica", "versicolor"]
    print("    -> Passed! Batch CSV processed 3 rows and returned predicted species: setosa, virginica, versicolor.")

    print("[*] 5. Testing POST /clear-history...")
    res = client.post("/clear-history")
    assert res.status_code in [302, 200]
    cleared_history = get_recent_history()
    # History after batch was logged, then cleared
    assert len(cleared_history) == 0, "Expected empty history after clear"
    print("    -> Passed! History database cleared successfully.")

    print("\n[ALL TESTS PASSED] 100% of features (Single, Batch CSV, SQLite, UI) verified successfully!")

if __name__ == "__main__":
    run_tests()
