import sys
import os
import argparse
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "iris_rf_model_full.joblib")
FEATURE_NAMES = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
CLASSES = ['setosa', 'versicolor', 'virginica']

def classify(sl, sw, pl, pw):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([[sl, sw, pl, pw]], columns=FEATURE_NAMES)
    
    pred_idx = model.predict(df)[0]
    probs = model.predict_proba(df)[0]
    pred_class = CLASSES[pred_idx]
    
    print("\n" + "=" * 50)
    print("      🌸 IRIS CLASSIFICATION RESULT 🌸")
    print("=" * 50)
    print(f"Input Features:")
    print(f"  • Sepal Length: {sl:.1f} cm")
    print(f"  • Sepal Width:  {sw:.1f} cm")
    print(f"  • Petal Length: {pl:.1f} cm")
    print(f"  • Petal Width:  {pw:.1f} cm")
    print("-" * 50)
    print(f"Predicted Species:  IRIS {pred_class.upper()}")
    print(f"Model Confidence:   {probs[pred_idx] * 100:.1f}%")
    print("-" * 50)
    print("Class Probabilities:")
    for i, c in enumerate(CLASSES):
        bar = "█" * int(probs[i] * 30)
        print(f"  • Iris {c.capitalize():10s}: {probs[i]*100:5.1f}% | {bar}")
    print("=" * 50 + "\n")
    return pred_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify Iris flower measurements.")
    parser.add_argument("--sl", type=float, default=5.1, help="Sepal length in cm")
    parser.add_argument("--sw", type=float, default=3.5, help="Sepal width in cm")
    parser.add_argument("--pl", type=float, default=1.4, help="Petal length in cm")
    parser.add_argument("--pw", type=float, default=0.2, help="Petal width in cm")
    args = parser.parse_args()

    classify(args.sl, args.sw, args.pl, args.pw)
