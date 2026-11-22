# Iris Classification System (Pure Python, HTML & CSS)

A complete, production-grade machine learning classification system built exclusively with **Python**, **HTML5**, and **CSS3**, featuring **Batch CSV Processing**, an **SQLite Prediction History Database**, **Vercel Serverless deployment**, and **Docker containerization**.

---

## 🌸 Key Features

1. **Scikit-Learn ML Classification**:
   * Evaluated across 4 model families using **5-Fold Stratified Cross-Validation**.
   * Production **Random Forest Classifier** achieving **96.67% hold-out test accuracy**, **0.9666 Macro F1**, and **0.9883 ROC-AUC**.
   * Feature importance analysis revealing petal measurements account for **87.4%** of classification decisions.
2. **Pure Python + HTML5 + CSS3 Web Application**:
   * Standard semantic HTML `<form method="POST">` interface.
   * Handcrafted pure CSS stylesheet ([static/css/style.css](file:///C:/Users/USER/.gemini/antigravity/scratch/iris-classifier/static/css/style.css)) with zero external CSS or JavaScript dependencies.
3. **Batch CSV Classification**:
   * Upload any `.csv` file containing flower measurements.
   * Python classifies all rows in bulk and returns the processed CSV with predictions and confidence scores for download.
   * Built-in sample CSV generator: [Download Sample CSV](http://localhost:8000/sample-csv).
4. **SQLite Prediction History Database**:
   * Automatically persists every classification with timestamps, inputs, predicted species, and confidence into `predictions_history.db`.
   * Displays recent predictions in a clean log table with a one-click "Clear Log" action.
5. **Multiple Deployment Targets**:
   * **Local Run**: Pure Python execution via `run_local_server.py`.
   * **Vercel Serverless**: Configured with `vercel.json` and `@vercel/python`.
   * **Docker**: Pre-configured `Dockerfile` and `docker-compose.yml`.

---

## 🚀 How to Run the System Locally

```powershell
cd C:\Users\USER\.gemini\antigravity\scratch\iris-classifier
& ".\.venv\Scripts\python.exe" run_local_server.py
```

* **Interactive Web Application**: [http://localhost:8000/](http://localhost:8000/)
* **API Metadata**: [http://localhost:8000/api/predict](http://localhost:8000/api/predict)
* **Sample CSV Download**: [http://localhost:8000/sample-csv](http://localhost:8000/sample-csv)

---

## 🐳 Running with Docker

You can build and run the entire application inside a Docker container with a single command:

```bash
docker-compose up --build
```
Then visit [http://localhost:8000/](http://localhost:8000/) in your browser.

---

## 🧪 Verification & Automated Testing

Run the automated test suite verifying single predictions, SQLite logging, batch CSV processing, and template rendering:

```powershell
& "C:\Users\USER\.gemini\antigravity\scratch\iris-classifier\.venv\Scripts\python.exe" test_flask_app.py
```
