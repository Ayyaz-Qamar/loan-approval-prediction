"""
app.py
======
Flask backend for the Loan Approval Prediction System.

Routes
------
GET  /              -> Home page with the input form
POST /predict       -> Accepts form data, returns prediction page
GET  /about         -> Project information page
GET  /api/predict   -> JSON API endpoint (POST) for programmatic access
GET  /health        -> Health check (useful for deployment platforms)
"""

import os
import pickle
import logging
from datetime import datetime

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for

from utils.preprocessing import prepare_input, get_risk_band

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join("model", "loan_model.pkl")


# ---------------------------------------------------------------------------
# Load the trained model artifact once at startup
# ---------------------------------------------------------------------------
def load_model_artifact():
    """Load the pickled model + metadata."""
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found at %s. "
                     "Run `python train_model.py` first.", MODEL_PATH)
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)
        logger.info("Model loaded: %s (accuracy=%.4f)",
                    artifact.get("best_name", "unknown"),
                    artifact.get("metrics", {}).get("accuracy", 0))
        return artifact
    except Exception as exc:                          # noqa: BLE001
        logger.exception("Failed to load model: %s", exc)
        return None


MODEL_ARTIFACT = load_model_artifact()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
]


def validate_form(form) -> tuple[bool, str]:
    """Return (is_valid, error_message)."""
    for field in REQUIRED_FIELDS:
        if field not in form or form.get(field) in (None, ""):
            return False, f"Missing required field: {field}"

    # Numeric range checks
    try:
        income     = float(form["ApplicantIncome"])
        coincome   = float(form["CoapplicantIncome"])
        loan_amt   = float(form["LoanAmount"])
        term       = float(form["Loan_Amount_Term"])
        credit     = float(form["Credit_History"])
    except ValueError:
        return False, "Income, loan amount, term and credit history must be numeric."

    if income < 0 or coincome < 0:
        return False, "Income values cannot be negative."
    if income == 0 and coincome == 0:
        return False, "Either applicant or co-applicant must have income greater than 0."
    if loan_amt <= 0:
        return False, "Loan amount must be greater than 0."
    if term <= 0:
        return False, "Loan term must be greater than 0."
    if credit not in (0.0, 1.0):
        return False, "Credit history must be 0 or 1."

    return True, ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    """Render the home page with the loan application form."""
    return render_template("index.html", year=datetime.now().year)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission, run inference, and render the result page."""
    if MODEL_ARTIFACT is None:
        flash("The prediction model is not loaded. "
              "Please run `python train_model.py` and restart the app.",
              "danger")
        return redirect(url_for("home"))

    is_valid, err = validate_form(request.form)
    if not is_valid:
        flash(err, "warning")
        return redirect(url_for("home"))

    try:
        # Convert form -> single-row dataframe with engineered features
        input_df = prepare_input(request.form.to_dict())

        # Predict
        model = MODEL_ARTIFACT["model"]
        proba = float(model.predict_proba(input_df)[0][1])  # P(Approved)
        pred  = int(proba >= 0.5)

        risk = get_risk_band(proba)

        result = {
            "prediction":     "Approved" if pred == 1 else "Rejected",
            "approved":       pred == 1,
            "confidence":     round(proba * 100, 2) if pred == 1
                              else round((1 - proba) * 100, 2),
            "approval_proba": round(proba * 100, 2),
            "risk":           risk,
            "model_name":     MODEL_ARTIFACT.get("best_name", "Model"),
            "form":           request.form.to_dict(),
            "submitted_at":   datetime.now().strftime("%d %b %Y, %H:%M"),
        }
        return render_template("result.html", **result, year=datetime.now().year)

    except Exception as exc:                              # noqa: BLE001
        logger.exception("Prediction failed: %s", exc)
        flash(f"Something went wrong while making the prediction: {exc}",
              "danger")
        return redirect(url_for("home"))


@app.route("/about")
def about():
    """Render an info page describing the model and dataset."""
    metrics = MODEL_ARTIFACT.get("metrics", {}) if MODEL_ARTIFACT else {}
    all_results = MODEL_ARTIFACT.get("all_results", []) if MODEL_ARTIFACT else []
    return render_template(
        "about.html",
        model_name=MODEL_ARTIFACT.get("best_name") if MODEL_ARTIFACT else "—",
        metrics=metrics,
        all_results=all_results,
        year=datetime.now().year,
    )


# ---------------------------------------------------------------------------
# JSON API (bonus)
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint – useful for testing or integrating with other apps."""
    if MODEL_ARTIFACT is None:
        return jsonify({"error": "Model not loaded"}), 503

    payload = request.get_json(silent=True) or {}
    is_valid, err = validate_form(payload)
    if not is_valid:
        return jsonify({"error": err}), 400

    try:
        input_df = prepare_input(payload)
        model    = MODEL_ARTIFACT["model"]
        proba    = float(model.predict_proba(input_df)[0][1])
        pred     = int(proba >= 0.5)
        return jsonify({
            "prediction":      "Approved" if pred == 1 else "Rejected",
            "approval_proba":  round(proba, 4),
            "confidence":      round(proba if pred == 1 else 1 - proba, 4),
            "risk":            get_risk_band(proba),
            "model":           MODEL_ARTIFACT.get("best_name"),
        })
    except Exception as exc:                              # noqa: BLE001
        logger.exception("API prediction failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    """Simple health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_ARTIFACT is not None,
        "timestamp": datetime.utcnow().isoformat(),
    })


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404,
                           message="Page not found.",
                           year=datetime.now().year), 404


@app.errorhandler(500)
def internal_error(_):
    return render_template("error.html", code=500,
                           message="Internal server error.",
                           year=datetime.now().year), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
