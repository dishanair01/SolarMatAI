
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from joblib import dump

from matminer.datasets import load_dataset

print("=" * 55)
print("SolarMatAI - Band Gap Prediction")
print("=" * 55)

start = time.time()

output = Path("artifacts")
output.mkdir(parents=True, exist_ok=True)

# 1. Load the real dataset
print("\n1. Loading dataset...")
df = load_dataset("matbench_mp_gap")
print(f"Loaded {len(df):,} material records.")

# 2. Convert crystal structures into composition features
print("\n2. Extracting elemental composition features...")
print("This may take a few minutes.")

structures = df["structure"].tolist()
target = pd.to_numeric(df["gap pbe"], errors="coerce").to_numpy()

# Find every element represented in the dataset
elements = sorted({
    element.symbol
    for structure in structures
    for element in structure.composition.elements
})

element_index = {symbol: i for i, symbol in enumerate(elements)}
X = np.zeros((len(structures), len(elements)), dtype=np.float32)

for row, structure in enumerate(structures):
    composition = structure.composition.fractional_composition
    for element, fraction in composition.items():
        X[row, element_index[element.symbol]] = float(fraction)

# Keep valid target records
valid = np.isfinite(target)
X = X[valid]
y = target[valid]

print(f"Number of records: {len(y):,}")
print(f"Number of elemental features: {len(elements)}")
print(f"Band gap range: {y.min():.4f} to {y.max():.4f} eV")

# 3. Split the data before training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining records: {len(y_train):,}")
print(f"Test records: {len(y_test):,}")

# 4. Establish a simple baseline
print("\n3. Evaluating baseline...")
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

baseline_mae = mean_absolute_error(y_test, baseline_pred)
print(f"Baseline MAE: {baseline_mae:.4f} eV")

# 5. Train an ML model using a log-transformed target
print("\n5. Training log-transformed Ridge regression...")

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
import json

# Train using all training records.
# log1p safely handles zero band-gap values.
model = make_pipeline(
    SimpleImputer(strategy="constant", fill_value=0),
    StandardScaler(),
    Ridge(alpha=10.0)
)

model.fit(X_train, np.log1p(y_train))

# Predict in log space, convert back to eV, and prevent negative outputs.
predictions = np.maximum(
    np.expm1(model.predict(X_test)),
    0
)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nFINAL MODEL RESULTS")
print(f"MAE:  {mae:.4f} eV")
print(f"RMSE: {rmse:.4f} eV")
print(f"R²:   {r2:.4f}")
print(f"Test records: {len(y_test)}")

# 6. Save the trained model
dump(model, output / "bandgap_model.joblib")

# Save evaluation metrics
metrics = {
    "dataset": "Matbench MP band gap",
    "records_used": int(len(X)),
    "features": int(X.shape[1]),
    "target": "PBE band gap (eV)",
    "baseline_mae_ev": float(baseline_mae),
    "mae_ev": float(mae),
    "rmse_ev": float(rmse),
    "r2": float(r2),
    "test_fraction": 0.2,
    "random_state": 42,
    "model_type": "Ridge regression with log-transformed target",
    "negative_predictions_after_clipping": int(
        (np.maximum(np.expm1(model.predict(X_test)), 0) < 0).sum()
    ),
    "limitations": (
        "Predictions are estimates from elemental composition features. "
        "They do not account fully for crystal structure or experimental "
        "conditions. The model was evaluated on a random test split."
    )
}

with open(output / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save predictions for the evaluation dashboard
pd.DataFrame({
    "actual_gap_ev": np.asarray(y_test),
    "predicted_gap_ev": predictions
}).to_csv(output / "test_predictions.csv", index=False)

print("\nSaved model, metrics, and test predictions to:", output)
print("Training complete!")