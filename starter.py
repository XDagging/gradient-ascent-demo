"""
🦩 GRADIENT ASCENT — Level 2: The Flamingo Files
================================================
OBJECTIVE: Train a neural network on tabular flamingo data.
METRIC:    AUC-ROC ≥ 0.95
DEPLOY:    FastAPI  →  POST /predict

pip install tensorflow scikit-learn pandas numpy fastapi uvicorn
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

class PredictRequest(BaseModel):
    rows: list[list[float]]


class PredictResponse(BaseModel):
    predictions: list[float]


if __name__ == "__main__":
    # ─── Data loading (given) ─────────────────────────────────────────────────────
    df = pd.read_csv(Path(__file__).parent / "flamingo_training.csv")
    feature_cols = [c for c in df.columns if c != "is_flamingo"]
    X = df[feature_cols].values
    y = df["is_flamingo"].values
    print(f"Loaded {len(df)} samples | features: {feature_cols}")


    # 1. Split into train and validation sets (e.g. 80/20, stratify on y).
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 2. Scale your features with sklearn.preprocessing.StandardScaler.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)


    # ─── Build and train a neural network ───────────────────────────────────
    # Use tensorflow.keras.Sequential.
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')], run_eagerly=True)

    early_stopping = callbacks.EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True)

    print("Starting training...")
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    # Save your scaler and model when done:
    joblib.dump(scaler, Path(__file__).parent / "scaler.pkl")
    model.save(Path(__file__).parent / "model.keras")


    # After training, compute AUC on the validation set:
    preds = model.predict(X_val_scaled).flatten()
    auc_score = roc_auc_score(y_val, preds)
    print(f"Validation AUC: {auc_score:.4f}")

    if auc_score >= 0.95:
        print("SUCCESS: Target AUC reached!")
    else:
        print("FAILURE: Target AUC not reached. Try adjusting the architecture or training parameters.")


# ─── Deploy (complete this once your AUC passes) ──────────────────────────────
import joblib
import tensorflow as tf

app = FastAPI(title="Level 2 - Flamingo Classifier")

# Load model and scaler for deployment
try:
    _scaler = joblib.load(Path(__file__).parent / "scaler.pkl")
    _model  = tf.keras.models.load_model(Path(__file__).parent / "model.keras")
except:
    print("Model or scaler not found. Make sure to train first.")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Scale input
    X_input = np.array(req.rows)
    X_scaled = _scaler.transform(X_input)
    
    # Run model
    preds = _model.predict(X_scaled).flatten()
    
    return PredictResponse(predictions=preds.tolist())


@app.get("/health")
def health():
    return {"status": "ok", "level": 2}

# Run locally:   uvicorn level_2_starter:app --reload
# Deploy:        Render / Railway / Vercel  (see grad learn deployment)
# Submit:        grad submit 2 https://your-app.com
