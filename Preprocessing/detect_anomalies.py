import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
# si vous voulez des métriques
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# On importe la classe, pour que Python puisse la retrouver au unpickling
from ensemble import VotingAnomalyDetector

def main():
    # 1) Charger le scaler + le modèle
    try:
        with open("my_scaler.pkl", "rb") as sf:
            scaler = pickle.load(sf)
        with open("my_model.pkl", "rb") as mf:
            model = pickle.load(mf)
        print("[INFO] Scaler et modèle chargés avec succès.")
    except FileNotFoundError as e:
        print(f"[ERREUR] Fichier introuvable : {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERREUR] Impossible de charger le scaler ou le modèle : {str(e)}")
        sys.exit(1)

    # 2) Charger le CSV à analyser
    DATA_FILE = "murex_data.csv"  # Ajustez le nom
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"[INFO] Fichier CSV '{DATA_FILE}' chargé. Shape={df.shape}")
    except FileNotFoundError:
        print(f"[ERREUR] Le fichier CSV '{DATA_FILE}' est introuvable.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERREUR] Impossible de lire le CSV : {str(e)}")
        sys.exit(1)

    # 3) Vérifier les features
    features = ["NotionalAmount", "PriceOrRate", "Latency(ms)"]
    missing = [col for col in features if col not in df.columns]
    if missing:
        print(f"[ERREUR] Colonnes manquantes dans le CSV : {missing}")
        sys.exit(1)

    X = df[features].values

    # 4) Normalisation + prédiction
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)   # 0=normal, 1=anomalie

    # On ajoute la colonne 'predicted_anomaly'
    df["predicted_anomaly"] = y_pred

    # 5) Si 'is_anomaly' existe, calculer des métriques
    if "is_anomaly" in df.columns:
        y_true = df["is_anomaly"].values
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred, digits=4)

        print("\n=== Évaluation Anomalies ===")
        print(f"Accuracy : {acc:.4f}")
        print("Matrice de Confusion:\n", cm)
        print("Classification Report:\n", cr)
    else:
        print("[INFO] Aucune colonne 'is_anomaly', pas de métriques calculées.")

    # 6) Exporter un fichier output avec seulement les anomalies
    anomalies_df = df[df["predicted_anomaly"] == 1].copy()
    OUTPUT_FILE = "output_anomalies.csv"
    anomalies_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Fichier '{OUTPUT_FILE}' généré, avec {len(anomalies_df)} lignes.")

    # (Optionnel) Export complet
    # df.to_csv("output_with_prediction.csv", index=False)
    # print("[INFO] Fichier 'output_with_prediction.csv' généré avec toutes les lignes.")

if __name__ == "__main__":
    main()
