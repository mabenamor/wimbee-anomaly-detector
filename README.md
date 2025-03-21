# Wimbee – Local AI Anomaly Detection Platform

This is a local demo application built by Wimbee's AI Factory to detect anomalies in Murex trade data using machine learning.

The model uses a combination of:
- ✅ One-Class SVM
- ✅ A custom DBSCAN-like density method
- ❌ Isolation Forest (excluded due to poor precision)

---

## 🖥️ How to Run Locally (Windows)

### 1. ✅ Requirements
- Python 3.9.6
- pip installed (comes with Python)

### 2. 📁 Project Structure
```
project_folder/
├── app.py
├── ensemble.py
├── my_model.pkl
├── my_scaler.pkl
├── requirements.txt
├── static/
│   └── (for images & output graphs)
├── templates/
│   ├── layout.html
│   ├── index.html
│   ├── test.html
│   └── about.html
└── uploads/ (auto-created)
```

### 3. 🐍 Create a virtual environment
Open a terminal (CMD or Git Bash):

```bash
python -m venv venv
```

Activate it:
```bash
venv\Scripts\activate
```

### 4. 📦 Install dependencies
```bash
pip install -r requirements.txt
```

If you don’t have the file, install manually:
```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn
```

### 5. ▶️ Launch the app
```bash
python app.py
```
You’ll see:
```
 * Running on http://127.0.0.1:5000/
```

### 6. 🌍 Open in your browser
Go to:
[http://localhost:5000](http://localhost:5000)

You can now:
- Upload Murex-like CSV trade data
- Detect anomalies in real time
- View accuracy, confusion matrix, and animated preview

---

## 💡 Tips
- To enable auto-reloading on code change:
```bash
set FLASK_APP=app.py
set FLASK_ENV=development
flask run
```

---

Built with ❤️ by Wimbee’s AI Factory
