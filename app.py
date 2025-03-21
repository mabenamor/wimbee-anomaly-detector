from flask import Flask, render_template, request, send_file, redirect, url_for
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

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
STATIC_FOLDER = "static"
ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["STATIC_FOLDER"] = STATIC_FOLDER

# Load model and scaler
with open("my_scaler.pkl", "rb") as sf:
    scaler = pickle.load(sf)
with open("my_model.pkl", "rb") as mf:
    model = pickle.load(mf)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("test.html", error="No file selected.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("test.html", error="Invalid filename.")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            required_cols = ["NotionalAmount", "PriceOrRate", "Latency(ms)"]
            if not all(col in df.columns for col in required_cols):
                return render_template("test.html", error="Missing required columns in the file.")

            X = df[required_cols].values
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)

            df["predicted_anomaly"] = y_pred
            anomalies = df[df["predicted_anomaly"] == 1]

            metrics = None
            if "is_anomaly" in df.columns:
                y_true = df["is_anomaly"].values
                acc = accuracy_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                cr_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
                cr_df = pd.DataFrame(cr_dict).transpose()
                cr_filename = f"classification_report_{uuid.uuid4().hex[:6]}.png"

                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal (0)', 'Anomaly (1)'], yticklabels=['Normal (0)', 'Anomaly (1)'])
                plt.title("Confusion Matrix (1 = Anomaly)")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                conf_filename = f"conf_matrix_{uuid.uuid4().hex[:6]}.png"
                plt.tight_layout()
                plt.savefig(os.path.join(STATIC_FOLDER, conf_filename))
                plt.close()

                renamed_index = {
                    '0': 'Normal (0)',
                    '1': 'Anomaly (1)',
                    'accuracy': 'Accuracy',
                    'macro avg': 'Macro Avg',
                    'weighted avg': 'Weighted Avg'
                }
                cr_df.rename(index=renamed_index, inplace=True)

                plt.figure(figsize=(8, 4))
                sns.heatmap(cr_df.iloc[:-1, :-1], annot=True, cmap='YlGnBu', fmt=".2f")
                plt.title("Classification Report")
                plt.tight_layout()
                plt.savefig(os.path.join(STATIC_FOLDER, cr_filename))
                plt.close()

                metrics = {
                    "accuracy": f"{acc:.4f}",
                    "conf_image": conf_filename,
                    "report_image": cr_filename
                }

            output_file = f"anomalies_{filename}"
            anomalies.to_csv(os.path.join(app.config["OUTPUT_FOLDER"], output_file), index=False)

            return render_template(
                "test.html",
                success=True,
                anomalies_count=len(anomalies),
                output_file=output_file,
                metrics=metrics,
                anomaly_table=anomalies.head(10)
            )
        else:
            return render_template("test.html", error="Unsupported file format.")

    return render_template("test.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(STATIC_FOLDER, exist_ok=True)
    app.run(debug=True)
