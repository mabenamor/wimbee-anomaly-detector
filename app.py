from flask import Flask, render_template, request, send_file, redirect, url_for, session
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
from werkzeug.utils import secure_filename

from ensemble import VotingAnomalyDetector
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
STATIC_FOLDER = "static"
ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.secret_key = "supersecret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["STATIC_FOLDER"] = STATIC_FOLDER

with open("my_scaler.pkl", "rb") as sf:
    scaler = pickle.load(sf)
with open("my_model.pkl", "rb") as mf:
    model = pickle.load(mf)

# Charger le modèle UMAP pré-entraîné
umap_model = joblib.load("umap_model.pkl")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "" or not allowed_file(file.filename):
            return render_template("test.html", error="Please upload a valid .csv file")

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        session["last_csv"] = filepath

        required_cols = ["NotionalAmount", "PriceOrRate", "Latency(ms)"]
        if not all(col in df.columns for col in required_cols):
            return render_template("test.html", error="Missing required columns")

        X = df[required_cols].values
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        df["predicted_anomaly"] = y_pred
        anomalies = df[df["predicted_anomaly"] == 1]

        # UMAP projection with pre-trained model
        X_umap = umap_model.transform(X_scaled)
        fig_name = f"umap_proj_{uuid.uuid4().hex[:6]}.png"
        plt.figure(figsize=(6, 5))
        colors = ['#2ca02c' if y == 0 else '#d62728' for y in y_pred]
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, s=10, edgecolor='k', linewidth=0.3)
        plt.title("UMAP Projection\nGreen = Normal, Red = Anomaly")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='#2ca02c', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='#d62728', markersize=6)
        ]
        plt.legend(handles=legend_elements, loc='best')
        plt.savefig(os.path.join(STATIC_FOLDER, fig_name))
        plt.close()

        session["umap_image"] = fig_name

        output_file = f"anomalies_{filename}"
        anomalies.to_csv(os.path.join(OUTPUT_FOLDER, output_file), index=False)

        metrics = None
        if "is_anomaly" in df.columns:
            y_true = df["is_anomaly"].values
            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            cr_dict = classification_report(y_true, y_pred, output_dict=True)
            cr_df = pd.DataFrame(cr_dict).transpose()
            conf_img = f"conf_{uuid.uuid4().hex[:6]}.png"
            rep_img = f"report_{uuid.uuid4().hex[:6]}.png"

            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(STATIC_FOLDER, conf_img))
            plt.close()

            plt.figure(figsize=(7, 4))
            sns.heatmap(cr_df.iloc[:-1, :-1], annot=True, cmap='YlGnBu', fmt=".2f")
            plt.title("Classification Report")
            plt.tight_layout()
            plt.savefig(os.path.join(STATIC_FOLDER, rep_img))
            plt.close()

            metrics = {
                "accuracy": f"{acc:.4f}",
                "conf_image": conf_img,
                "report_image": rep_img
            }

        return render_template(
            "test.html",
            success=True,
            anomalies_count=len(anomalies),
            output_file=output_file,
            metrics=metrics,
            anomaly_table=anomalies.head(10)
        )

    return render_template("test.html")

@app.route("/explanation")
def explanation():
    image_name = session.get("umap_image")
    return render_template("explanation.html", umap_image=image_name, download_link=url_for('static', filename=image_name))

@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(STATIC_FOLDER, exist_ok=True)
    app.run(debug=True)