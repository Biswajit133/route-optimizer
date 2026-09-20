"""
No-Show & Failure Prediction.

Predicts the probability that a booked visit will fail - the patient cancels,
denies, or isn't reachable - using a classifier trained on historical visits.
This is different from TatBreachRiskService's TAT-breach risk: that predicts
a *late* arrival from live traffic on an already-committed visit; this
predicts the visit never happening at all, learned from past cancellation
patterns, so a risky booking can be flagged before a phlebotomist is even
dispatched to it.

Label
-----
failure = 1  : visit ended Cancelled for a patient-side reason (see
               FAILURE_REASONS below - a deliberate whitelist, not "any
               Cancelled row"). Duplicate orders, porter/offboarding mixups,
               unpaid-order cancellations and phlebotomist-fault cancellations
               are excluded on purpose: those are ops mistakes, not something
               a booking-time model could ever predict from patient/visit
               attributes - keeping them in would just teach the model noise.
failure = 0  : visit reached a real completion status (Completed, Sample
               Collected, Dropped In Lab, or later in the pipeline).
excluded     : still in progress (Scheduled, Accepted, Phlebo Assigned, ...)
               - no outcome yet - or a Cancelled row whose reason isn't in
               the whitelist.

FAILURE_REASONS is a judgment call made from reading the data, not domain
knowledge of this business - review and adjust it before trusting the model.

Features (all knowable at booking time - nothing derived from what happened
during or after the visit, so the model can't leak the outcome to itself):
city_id, state_id, day of week, slot hour, lead time in days (booking to
slot), payment_type, is_child, price.

Data access
-----------
Training reads real historical visits (read-only SELECT, no writes) via
PyMySQL, using DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD read from .env
(or the real environment) - never hardcode credentials in this file. Copy
.env.example to .env and fill it in; .env is gitignored because this repo is
public. The query selects only operational fields: no patient name, phone,
or address leaves the database.

The serving endpoint (/predict-no-show) does NOT touch the database - it
takes the same fields as JSON from the caller, mirroring how /auto-assign and
/optimize in this repo already work, so DB credentials only ever need to
exist wherever training is run, never on the live web service.

Usage
-----
Train:  python no_show_prediction.py train
Serve:  python no_show_prediction.py
        (loads the saved model from no_show_model.joblib and runs Flask)

Needs `pip install pandas scikit-learn joblib pymysql python-dotenv` on top
of app.py's requirements.
"""

import os
import sys

import joblib
import jwt
import pandas as pd
from dotenv import load_dotenv
from flask import Blueprint, Flask, request, jsonify

# Real values live in .env (gitignored - this repo is public); .env.example is
# the committed template. Anything already exported in the shell wins, so a
# deployment that injects real env vars needs no .env file at all.
load_dotenv()

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "no_show_model.joblib")

FAILURE_REASONS = {
    "patient cancelled",
    "address details provided were incorrect or incomplete.",
    "address not found",
    "patient was not present at the given address.",
    "customer denied",
    "i have no time",
    "i dont want to go",
    "collection requested for other day",
    "difficult phlebotomy",
    "not ready to give sample",
    "patient has denied",
    "patient requested to cancel",
    "doctor denied",
    "patient did not pick up call",
    "my tests requirement got changed",
    "not available at scheduled time",
    "customer not available",
    "unforeseen circumstances",
}

SUCCESS_STATUSES = {
    "completed", "sample collected", "dropped in lab", "report available",
    "enroute to lab", "enroute to runner", "enroute to ship",
    "enroute to shipping", "dropped in courier", "collected",
}

FEATURE_COLUMNS = [
    "city_id", "state_id", "day_of_week", "slot_hour",
    "lead_time_days", "payment_type", "is_child", "price",
]
CATEGORICAL_COLUMNS = ["city_id", "state_id", "day_of_week", "payment_type"]


def _db_connection():
    import pymysql
    return pymysql.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", 3306)),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
        cursorclass=pymysql.cursors.DictCursor,
    )


def load_training_rows():
    conn = _db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                select status, cancellation_reason, city_id, state_id,
                       slot_date, slot_time, booked_date, payment_type,
                       is_child, price
                from visits
                where is_active = 1 and deleted_at is null
                  and slot_date is not null and slot_date != '0000-00-00'
                  and booked_date is not null and booked_date != ''
            """)
            return cur.fetchall()
    finally:
        conn.close()


def _slot_hour(slot_time):
    if not slot_time:
        return None
    try:
        first = slot_time.split('-')[0].strip()
        return int(first.split(':')[0])
    except (ValueError, IndexError):
        return None


def _lead_time_days(booked_date, slot_date):
    try:
        b = pd.to_datetime(booked_date)
        s = pd.to_datetime(slot_date)
        return max(0, (s - b).days)
    except (ValueError, TypeError):
        return None


def label_row(row):
    status = (row.get('status') or '').strip().lower()
    reason = (row.get('cancellation_reason') or '').strip().lower()

    if status == 'cancelled':
        return 1 if reason in FAILURE_REASONS else None
    if status in SUCCESS_STATUSES:
        return 0
    return None  # still in progress - no outcome yet


def build_features(row):
    slot_date = row.get('slot_date')
    return {
        "city_id": row.get('city_id') or 0,
        "state_id": row.get('state_id') or 0,
        "day_of_week": pd.to_datetime(slot_date).dayofweek if slot_date else None,
        "slot_hour": _slot_hour(row.get('slot_time')),
        "lead_time_days": _lead_time_days(row.get('booked_date'), slot_date),
        "payment_type": (row.get('payment_type') or 'unknown').strip() or 'unknown',
        "is_child": int(row.get('is_child') or 0),
        "price": float(row.get('price') or 0),
    }


def _to_frame(rows_features):
    df = pd.DataFrame(rows_features)
    df = df.dropna(subset=["day_of_week", "slot_hour", "lead_time_days"])
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str)
    return df


def train():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report

    raw_rows = load_training_rows()
    print(f"Fetched {len(raw_rows)} visits")

    features, labels = [], []
    for row in raw_rows:
        label = label_row(row)
        if label is None:
            continue
        features.append(build_features(row))
        labels.append(label)

    df = _to_frame(features)
    labels = pd.Series(labels).loc[df.index]

    print(f"Training rows: {len(df)}  (failures: {int(labels.sum())}, {labels.mean() * 100:.1f}%)")

    X = pd.get_dummies(df[FEATURE_COLUMNS], columns=CATEGORICAL_COLUMNS)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")
    print(classification_report(y_test, preds, target_names=["completed", "failure"]))

    joblib.dump({"model": model, "feature_names": feature_names}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def predict_one(visit):
    bundle = joblib.load(MODEL_PATH)
    model, feature_names = bundle["model"], bundle["feature_names"]

    df = _to_frame([build_features(visit)])
    if df.empty:
        return None  # missing slot_date, slot_time, or booked_date

    X = pd.get_dummies(df[FEATURE_COLUMNS], columns=CATEGORICAL_COLUMNS)
    X = X.reindex(columns=feature_names, fill_value=0)

    return float(model.predict_proba(X)[0, 1])


def customer_id_from_token():
    """Same check as delay_prediction.py's - this endpoint's own math never
    uses customer_id (the model only sees city_id/state_id/slot/payment/price,
    nothing tenant-specific), but every other live endpoint in this app
    requires a valid caller token, and this one shouldn't be the one gap an
    anonymous caller can hit. Returns (customer_id, error_message)."""
    secret = os.environ.get('JWT_SECRET')
    if not secret:
        return None, "JWT_SECRET is not configured on this service"

    header = request.headers.get('Authorization', '')
    if not header.startswith('Bearer '):
        return None, "Token is required"

    try:
        claims = jwt.decode(header[7:], secret, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return None, "Token has expired"
    except jwt.InvalidTokenError:
        return None, "Token is invalid"

    customer_id = claims.get('customer_id')
    if not customer_id:
        return None, "Token carries no customer_id"

    return customer_id, None


bp = Blueprint('no_show', __name__)


@bp.route('/predict-no-show', methods=['POST'])
def predict_no_show_endpoint():
    try:
        customer_id, auth_error = customer_id_from_token()
        if auth_error:
            return jsonify({"success": False, "message": auth_error}), 401

        visit = request.json
        risk = predict_one(visit)
        if risk is None:
            return jsonify({"error": "visit missing slot_date, slot_time or booked_date"}), 400

        return jsonify({
            "riskPercent": round(risk * 100, 1),
            "severity": "high" if risk >= 0.5 else ("medium" if risk >= 0.25 else "low"),
            "customerId": customer_id,
        })
    except FileNotFoundError:
        return jsonify({"error": "model not trained yet - run `python no_show_prediction.py train` first"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train()
    else:
        # Standalone mode - app.py mounts this blueprint instead when run together.
        app = Flask(__name__)
        app.register_blueprint(bp)
        port = int(os.environ.get("PORT", 5002))
        app.run(host='0.0.0.0', port=port)
