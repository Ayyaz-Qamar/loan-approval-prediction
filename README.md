# 🏦 LoanPredictAI – End-to-End Loan Approval Prediction System

A complete, production-ready machine learning project that predicts whether a
loan application will be **Approved** or **Rejected**, complete with a
professional Bootstrap-based web UI.

Built with **Flask + Scikit-Learn + XGBoost + Bootstrap 5**.

---

## ✨ Features

| Layer        | What's inside                                                            |
|--------------|--------------------------------------------------------------------------|
| **ML**       | 3 models compared: Logistic Regression, Random Forest, XGBoost           |
| **Pipeline** | Cleaning, missing-value imputation, feature engineering, encoding, scaling |
| **Eval**     | Accuracy, Precision, Recall, Confusion Matrix, full classification report |
| **Backend**  | Flask app with `/`, `/predict`, `/about`, `/health`, and JSON `/api/predict` |
| **Frontend** | Responsive Bootstrap 5 UI with custom theme, hero, validation & risk score |
| **Bonus**    | Risk band badge, confidence %, input validation, error pages, JSON API |
| **Deploy**   | `Procfile`, `render.yaml`, `runtime.txt`, `requirements.txt` ready to go  |

---

## 📁 Folder Structure

```
loan_approval_system/
├── app.py                  # Flask application
├── train_model.py          # ML training + evaluation script
├── requirements.txt        # Python dependencies
├── Procfile                # For Render / Railway / Heroku
├── render.yaml             # One-click Render blueprint
├── runtime.txt             # Python version
├── README.md               # You are here
├── .gitignore
│
├── data/
│   └── loan_train.csv      # Training data (Kaggle CSV or auto-generated)
│
├── model/
│   └── loan_model.pkl      # Trained model artifact (created by train_model.py)
│
├── utils/
│   ├── __init__.py
│   └── preprocessing.py    # Cleaning, feature eng, input prep, risk band
│
├── templates/
│   ├── base.html           # Shared layout (navbar, footer)
│   ├── index.html          # Home page + application form
│   ├── result.html         # Prediction result page
│   ├── about.html          # Model performance / metrics page
│   └── error.html          # 404 / 500 error page
│
└── static/
    ├── css/style.css       # Custom theme
    └── js/script.js        # Client-side validation
```

---

## 📊 Dataset

This project uses the well-known **Loan Prediction** dataset by *Analytics Vidhya / Dream Housing Finance*, available on Kaggle:

- **Kaggle:** <https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset>
- **Mirror:** <https://www.kaggle.com/datasets/ninzaami/loan-predication>

After downloading, save the file as:

```
data/loan_train.csv
```

> 💡 If `data/loan_train.csv` is **not** present, `train_model.py` will
> automatically generate a realistic synthetic dataset with the same schema
> so you can still run the project end-to-end out of the box.

The dataset's columns are:

| Column              | Type       | Description                            |
|---------------------|------------|----------------------------------------|
| Loan_ID             | string     | Unique loan id (dropped during cleanup) |
| Gender              | categorical| Male / Female                          |
| Married             | categorical| Yes / No                               |
| Dependents          | categorical| 0 / 1 / 2 / 3+                         |
| Education           | categorical| Graduate / Not Graduate                |
| Self_Employed       | categorical| Yes / No                               |
| ApplicantIncome     | numeric    | Monthly income of applicant            |
| CoapplicantIncome   | numeric    | Monthly income of co-applicant         |
| LoanAmount          | numeric    | Loan amount in thousands               |
| Loan_Amount_Term    | numeric    | Term in months                         |
| Credit_History      | numeric    | 1 = good, 0 = poor                     |
| Property_Area       | categorical| Urban / Semiurban / Rural              |
| **Loan_Status**     | target     | Y (Approved) / N (Rejected)            |

---

## 🚀 Run Locally — Step-by-Step

### 1. Clone / extract the project

```bash
cd loan_approval_system
```

### 2. Create a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Add the Kaggle dataset

Download `loan_train.csv` from Kaggle and place it in `data/`.
Skip this step if you're happy with the auto-generated synthetic data.

### 5. Train the models

```bash
python train_model.py
```

You'll see something like:

```
[INFO] Loading dataset from data/loan_train.csv
[INFO] Train shape: (491, 13) | Test shape: (123, 13)

========== Logistic Regression ==========
Accuracy : 0.7967
Precision: 0.7895
Recall   : 0.9740
...

========== Random Forest ==========
Accuracy : 0.8211
...

========== XGBoost ==========
Accuracy : 0.8049
...

[INFO] Best model: Random Forest (accuracy = 0.8211)
[INFO] Saved best model to model/loan_model.pkl
```

### 6. Launch the Flask app

```bash
python app.py
```

Open <http://localhost:5000> in your browser. 🎉

---

## 🧪 Use the JSON API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
  }'
```

Response:

```json
{
  "prediction": "Approved",
  "approval_proba": 0.8421,
  "confidence": 0.8421,
  "risk": {"label": "Low Risk", "color": "success", "score": 84,
           "description": "Strong profile – very likely to be approved."},
  "model": "Random Forest"
}
```

---

## ☁️ Deployment

### Option A – Render (recommended, free tier)

1. Push the project to GitHub.
2. Go to <https://render.com> → **New + → Blueprint** → connect your repo.
3. Render will auto-detect `render.yaml` and set everything up.
4. The first build runs `python train_model.py` to create the model file, then
   starts the app with Gunicorn.

### Option B – Railway

1. <https://railway.app> → **New Project → Deploy from GitHub**.
2. Railway reads the `Procfile` and `runtime.txt` automatically.
3. Add a **Build Command**: `pip install -r requirements.txt && python train_model.py`.
4. Deploy. Done. 🚀

### Option C – Heroku

```bash
heroku create your-app-name
git push heroku main
heroku run python train_model.py   # generates model/loan_model.pkl
heroku open
```

### Option D – Docker (advanced)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python train_model.py
EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2"]
```

Then:

```bash
docker build -t loan-predictor .
docker run -p 5000:5000 loan-predictor
```

---

## 🔧 Configuration

Set these environment variables for production:

| Variable           | Purpose                              | Default      |
|--------------------|--------------------------------------|--------------|
| `FLASK_SECRET_KEY` | Session/flash signing key            | dev fallback |
| `PORT`             | Port to bind to                      | `5000`       |
| `FLASK_DEBUG`      | Set to `1` to enable debug locally   | `0`          |

---

## 🧠 How the Model Works

### Feature Engineering

| Feature             | Formula                                       |
|---------------------|-----------------------------------------------|
| `Total_Income`      | `ApplicantIncome + CoapplicantIncome`         |
| `Debt_Income_Ratio` | `(LoanAmount × 1000) / Total_Income`          |

### Preprocessing pipeline

```
ColumnTransformer
├── numeric  → StandardScaler
└── categorical → OneHotEncoder(handle_unknown="ignore")
```

### Risk Scoring

The approval probability is mapped to four bands:

| Probability | Band              | Color   |
|-------------|-------------------|---------|
| ≥ 80%       | Low Risk          | green   |
| 60% – 80%   | Moderate Risk     | blue    |
| 40% – 60%   | Elevated Risk     | yellow  |
| < 40%       | High Risk         | red     |

---

## 🛠️ Troubleshooting

| Problem                                          | Fix                                                              |
|--------------------------------------------------|------------------------------------------------------------------|
| "The prediction model is not loaded"             | Run `python train_model.py` first                                |
| `xgboost` install fails on macOS                 | `brew install libomp` then `pip install xgboost`                 |
| Port 5000 in use (macOS AirPlay)                 | `PORT=5050 python app.py`                                        |
| Render build times out                           | Switch to a smaller dataset or use a paid plan                   |

---

## 👨‍💻 Developer

**Ayyaz Qamar** — Machine Learning Engineer

- 🐙 GitHub: [@Ayyaz-Qamar](https://github.com/Ayyaz-Qamar)
- 💼 LinkedIn: [ayyaz-qamar](https://www.linkedin.com/in/ayyaz-qamar-41bb51383)

---

## 📜 License

MIT – do whatever you want, attribution appreciated.

---

## 🙏 Credits

- Dataset: [Analytics Vidhya Loan Prediction](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/)
- UI: [Bootstrap 5](https://getbootstrap.com), [Bootstrap Icons](https://icons.getbootstrap.com)
- Models: [scikit-learn](https://scikit-learn.org), [XGBoost](https://xgboost.readthedocs.io)
