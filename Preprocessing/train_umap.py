import pandas as pd
import numpy as np
import umap
import joblib
from sklearn.preprocessing import StandardScaler

# === Chargement du dataset ===
df = pd.read_csv("murex_data_less_extreme.csv")  # Remplace par ton fichier source si besoin

# === Préparation des données ===
features = ["NotionalAmount", "PriceOrRate", "Latency(ms)"]
X = df[features].dropna().values

# === Normalisation ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Entraînement du modèle UMAP ===
print("Entraînement de UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42)
reducer.fit(X_scaled)

# === Sauvegarde ===
joblib.dump(reducer, "umap_model.joblib")
print("✅ Modèle UMAP sauvegardé sous 'umap_model.joblib'")
