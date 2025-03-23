import pandas as pd
import joblib  # ✅ Import joblib à la place de pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# On importe la classe depuis ensemble.py
from ensemble import VotingAnomalyDetector

def main():
    # 1) Charger un dataset pour l'entraînement
    TRAIN_CSV = "murex_data_less_extreme.csv"  # Adaptez le nom
    df = pd.read_csv(TRAIN_CSV)
    print(f"[INFO] Fichier '{TRAIN_CSV}' chargé, shape={df.shape}")

    if "is_anomaly" not in df.columns:
        print("[WARN] Pas de colonne 'is_anomaly' => pas d'évaluation supervisée possible.")

    features = ["NotionalAmount", "PriceOrRate", "Latency(ms)"]
    X = df[features].values
    y = df["is_anomaly"].values if "is_anomaly" in df.columns else None

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if y is not None else None
    )

    # 3) Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4) Entraînement du voting
    model = VotingAnomalyDetector(
        if_contamination=0.01,
        svm_nu=0.01,
        threshold_knn=0.6,
        n_neighbors=10,
        random_state=42
    )
    model.fit(X_train_scaled)

    # 5) Évaluation (si y_test dispo)
    if y_test is not None:
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, digits=4)

        print("[INFO] Évaluation sur le test :")
        print(f"Accuracy = {acc:.4f}")
        print("Matrice de Confusion:\n", cm)
        print("Classification Report:\n", cr)
    else:
        print("[INFO] Aucune colonne 'is_anomaly', pas d'évaluation.")

    # 6) Sauvegarde du scaler + modèle avec joblib
    joblib.dump(scaler, "my_scaler.joblib")
    joblib.dump(model, "my_model.joblib")

    print("[INFO] Fichiers 'my_scaler.joblib' et 'my_model.joblib' créés.")

if __name__ == "__main__":
    main()
