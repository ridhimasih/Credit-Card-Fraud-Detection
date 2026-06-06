# Credit Card Fraud Detection
**MCA Project — Ridhi Masih**

## Project Structure
```
fraud_project/
├── creditcard.csv        ← your dataset (place here)
├── train_model.py        ← trains and saves the ML model
├── app.py                ← Flask backend server
├── requirements.txt      ← Python dependencies
├── fraud_model.pkl       ← generated after training
├── scaler.pkl            ← generated after training
├── feature_cols.pkl      ← generated after training
└── templates/
    └── index.html        ← HTML dashboard (served by Flask) 
```

## Setup & Run (VS Code)

### Step 1 — Install dependencies
Open the terminal in VS Code and run:
```bash
pip install -r requirements.txt
```

### Step 2 — Place your dataset
Copy `creditcard.csv` into the `fraud_project/` folder.

### Step 3 — Train the model
```bash
python train_model.py
```
This creates `fraud_model.pkl`, `scaler.pkl`, `feature_cols.pkl`.

### Step 4 — Start the Flask server
```bash
python app.py
```

### Step 5 — Open the dashboard
Go to: **http://127.0.0.1:5000**

---

## Dataset
Download from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Model
- Algorithm: Random Forest Classifier
- Features: V1–V28 (PCA), Amount (scaled), Time (scaled)
- Class weight: balanced (handles imbalanced data)
