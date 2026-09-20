"""
Predictive Delay Alerts - will this visit miss its slot?

Scores a visit on three signals and reports whichever is most alarmed:

  history         a model trained on past breaches - lead time, slot hour,
                  zone and phlebo track record. The only signal available
                  before anyone is dispatched, and the only one that knows
                  this slot has been running late for months.
  live traffic    Google Distance Matrix, traffic-aware, from the phlebo's
                  current position to the patient. Needs the caller to pass
                  phlebo_lat/phlebo_lng.
  field progress  where the visit actually is - a phlebo already on site is
                  scored on how long they have been standing there, one still
                  sitting at "assigned" is charged for not having set off yet,
                  and one "enroute" is trusted to be genuinely moving.

History alone answers "how risky is this booking"; the live two answer "is it
going wrong right now". They are deliberately not averaged - see the endpoint.

The live pair only fires for today's still-open visits, the same scope the PHP
/operations-overview/tat-breach-risk endpoint runs against. History is scored
for any visit on any date, so the endpoint always returns something.

Label
-----
breached = 1 : the sample was collected after its slot_time window ended.
breached = 0 : the sample was collected within its slot window.
excluded     : visit was never collected (cancelled, or no outcome yet) -
               no TAT outcome to learn from either way.

"Collected" is the earliest visit_status row in the same family PHP's
collectedVisitStatuses treats as done, not literally "Sample Collected":
only ~3.5k visits ever record that exact status while ~17.7k record
"Completed", so matching the single status would throw away most of the
history. For visits that do record it, it is naturally the earliest of the
family anyway; for the rest, the next status in the pipeline stands in.

Features (all knowable at booking time - nothing derived from what happened
during or after the visit): city_id, state_id, day of week, slot hour and
slot duration, plus the assigned phlebo's historical breach rate and the drop
pincode's historical breach rate.

Everything time-related is derived from slot_date/slot_time, never
booked_date: that column is NULL across large parts of the table (all of
customer 75's visits, for one), so depending on it silently dropped those
visits from training and made them unscorable at serving time.

The two rate features are looked up from a table baked into the saved model
at training time (see train()), not recomputed from the database, so the
serving endpoint below stays DB-free the same way /predict-no-show is.

Data access
-----------
Training reads real historical visits + visit_status (read-only SELECT, no
writes) via PyMySQL, using DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD read
from .env (or the real environment) - never hardcode credentials in this
file. Copy .env.example to .env and fill it in; .env is gitignored because
this repo is public.

Unlike /predict-no-show, /predict-delay reads the database too: it is a list
endpoint, not a calculator. Call it with the caller's JWT and it finds today's
open visits itself, pulls live phlebo positions from Firebase, and returns the
ranked list with the display fields already joined in - the same shape as
PHP's /operations-overview/tat-breach-risk, so a dashboard can swap between
them. That means DB credentials must exist wherever this is *served*, not
only where it is trained.

Which customer's rows come back is decided by the customer_id claim inside
that token and nothing else - see customer_id_from_token(). The rows carry
patient names and phone numbers, so a body parameter would be an open door.
Verifying it needs the same JWT_SECRET the PHP side signs with.

Serving needs GOOGLE_MAPS_API_KEY for the traffic leg. Without it - or for a
visit with no phlebo position - rows still come back, scored on history alone.
Fetched ETAs are cached for a few minutes in visit_tat_eta_cache, the same
table PHP uses, so the two endpoints don't bill Google twice for one pair.

Usage
-----
Train:  python delay_prediction.py train
Serve:  python delay_prediction.py
        (loads the saved model from delay_model.joblib and runs Flask)

Needs `pip install pandas scikit-learn joblib pymysql python-dotenv requests
PyJWT` on top of app.py's requirements.
"""

import math
import os
import sys

import joblib
import jwt
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Blueprint, Flask, request, jsonify

# Real values live in .env (gitignored - this repo is public); .env.example is
# the committed template. Anything already exported in the shell wins, so a
# deployment that injects real env vars needs no .env file at all.
load_dotenv()

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delay_model.joblib")

FEATURE_COLUMNS = [
    "city_id", "state_id", "day_of_week", "slot_hour", "slot_duration_minutes",
    "phlebo_breach_rate", "zone_breach_rate",
]
CATEGORICAL_COLUMNS = ["city_id", "state_id", "day_of_week"]

MAPS_ENDPOINT = "https://maps.googleapis.com/maps/api/distancematrix/json"
MAPS_TIMEOUT_SECONDS = 6

FIREBASE_LOCATIONS = "https://hsc-phlebotomist-app-default-rtdb.firebaseio.com/phlebo_locations"
FIREBASE_TIMEOUT_SECONDS = 5

# A slot ending further out than this has time to resolve itself and isn't worth
# checking yet. There is deliberately no bound on the other side: a visit whose
# slot passed hours ago and still hasn't happened is the whole point of the list.
WINDOW_AHEAD_MINUTES = 120

# Shared with PHP through the visit_tat_eta_cache table, so a dashboard refresh
# inside this window doesn't bill Google twice for the same phlebo-visit pair.
ETA_CACHE_MINUTES = 5

# Same idea, for /predict-delay-batch: a visit already scored within this many
# minutes is left alone rather than re-run through the model and a fresh
# traffic check on the next batch cycle.
RISK_CACHE_MINUTES = 5

# Statuses where the phlebo's work is done - there is no arrival left to be late
# for. Same set OrderDashboardNewController treats as terminal for tat-breach-risk.
TERMINAL_STATUSES = {
    'sample collected', 'completed', 'enroute to lab', 'enroute to ship',
    'enroute to runner', 'dropped in lab', 'dropped in courier',
    'sample delivered to lab', 'collected', 'collecting', 'cancelled',
}

# How long a phlebo can reasonably be on-site before the dwell itself becomes
# the risk, and how long it takes to get someone moving who hasn't left yet.
REASONABLE_COLLECTION_MINUTES = 15
ASSIGNMENT_OVERHEAD_MINUTES = 10

# How long past its deadline a visit that is genuinely moving still earns a
# fresh traffic check. Past this - or immediately, for one that never set off -
# the ETA stops changing what the desk should do, so it isn't worth buying.
WINDOW_BEHIND_MINUTES = 60


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
                select v.id, v.city_id, v.state_id, v.slot_date, v.slot_time,
                       v.phlebo_id, v.patient_pincode, v.status,
                       oc.collected_at
                from visits v
                left join (
                    select visit_id, min(created_at) as collected_at
                    from visit_status
                    where lower(trim(status)) in (
                        'sample collected', 'completed', 'enroute to lab',
                        'enroute to runner', 'enroute to ship', 'dropped in lab',
                        'dropped in courier', 'sample delivered to lab'
                    )
                    group by visit_id
                ) oc on oc.visit_id = v.id
                where v.is_active = 1 and v.deleted_at is null
                  and v.slot_date is not null and v.slot_date != '0000-00-00'
            """)
            return cur.fetchall()
    finally:
        conn.close()


def _slot_bounds(slot_time):
    """("11:00 - 12:00" | "11:00") -> (start_hour, duration_minutes). Bare start means a 30-min window."""
    if not slot_time:
        return None, None
    parts = str(slot_time).split('-')
    try:
        start_h, start_m = (int(x) for x in parts[0].strip().split(':'))
    except (ValueError, IndexError):
        return None, None

    duration = 30
    if len(parts) == 2:
        try:
            end_h, end_m = (int(x) for x in parts[1].strip().split(':'))
            duration = (end_h * 60 + end_m) - (start_h * 60 + start_m)
            if duration <= 0:
                duration = 30
        except (ValueError, IndexError):
            pass

    return start_h, duration


def _slot_end(slot_date, slot_time):
    if not slot_date or not slot_time:
        return None
    date_str = slot_date.strftime('%Y-%m-%d') if hasattr(slot_date, 'strftime') else str(slot_date)
    parts = str(slot_time).split('-')
    try:
        start = pd.Timestamp(f"{date_str} {parts[0].strip()}")
    except (ValueError, TypeError):
        return None
    if len(parts) == 2:
        try:
            return pd.Timestamp(f"{date_str} {parts[1].strip()}")
        except (ValueError, TypeError):
            return start + pd.Timedelta(minutes=30)
    return start + pd.Timedelta(minutes=30)


def label_row(row):
    collected_at = row.get('collected_at')
    if not collected_at:
        return None  # cancelled, or still in progress - no TAT outcome yet

    slot_end = _slot_end(row.get('slot_date'), row.get('slot_time'))
    if slot_end is None:
        return None

    return 1 if pd.to_datetime(collected_at) > slot_end else 0


def build_base_features(row):
    """Everything here comes from slot_date and slot_time, which every visit has.
    The two breach-rate columns are filled in separately by the caller (train()
    from history, predict_one() from the bundle), since neither can be derived
    from a single row alone."""
    slot_date = row.get('slot_date')
    slot_hour, slot_duration = _slot_bounds(row.get('slot_time'))
    return {
        "city_id": row.get('city_id') or 0,
        "state_id": row.get('state_id') or 0,
        "day_of_week": pd.to_datetime(slot_date).dayofweek if slot_date else None,
        "slot_hour": slot_hour,
        "slot_duration_minutes": slot_duration,
    }


def train():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report

    raw_rows = load_training_rows()
    print(f"Fetched {len(raw_rows)} visits")

    labeled = []
    for row in raw_rows:
        label = label_row(row)
        if label is None:
            continue
        base = build_base_features(row)
        base['breached'] = label
        base['phlebo_id'] = row.get('phlebo_id')
        base['patient_pincode'] = row.get('patient_pincode')
        base['slot_date'] = row.get('slot_date')
        labeled.append(base)

    df = pd.DataFrame(labeled)
    df = df.dropna(subset=["day_of_week", "slot_hour"])
    df = df.sort_values('slot_date').reset_index(drop=True)

    print(f"Training rows: {len(df)}  (breached: {int(df['breached'].sum())}, {df['breached'].mean() * 100:.1f}%)")

    overall_rate = df['breached'].mean()

    # Leakage-safe for the labels used to fit the model: an expanding mean shifted
    # by one row, so a visit only ever sees breach outcomes strictly before it -
    # never its own, never a later one. Cold start (no prior history) falls back
    # to the overall rate.
    df['phlebo_breach_rate'] = df.groupby('phlebo_id')['breached'].transform(
        lambda s: s.shift().expanding().mean()
    ).fillna(overall_rate)
    df['zone_breach_rate'] = df.groupby('patient_pincode')['breached'].transform(
        lambda s: s.shift().expanding().mean()
    ).fillna(overall_rate)

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str)

    X = pd.get_dummies(df[FEATURE_COLUMNS], columns=CATEGORICAL_COLUMNS)
    feature_names = X.columns.tolist()
    y = df['breached']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")
    print(classification_report(y_test, preds, target_names=["on_time", "breached"]))

    # Plain (non-shifted) historical rates for serving - unlike the labels above,
    # a live booking really does get to see every past outcome, so there is no
    # leakage to guard against here.
    phlebo_rates = {str(k): v for k, v in df.groupby('phlebo_id')['breached'].mean().to_dict().items()}
    zone_rates = {str(k): v for k, v in df.groupby('patient_pincode')['breached'].mean().to_dict().items()}

    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "phlebo_rates": phlebo_rates,
        "zone_rates": zone_rates,
        "overall_rate": overall_rate,
    }, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def fetch_todays_open_visits(conn, customer_id, city_id=None, state_id=None, slot=None):
    """Today's visits that haven't finished yet, with the display fields the caller
    needs, so the list can be rendered without a second round-trip to PHP."""
    where = ["visits.customer_id = %s", "visits.is_active = 1", "visits.deleted_at is null",
             "visits.slot_date = curdate()"]
    params = [customer_id]

    placeholders = ",".join(["%s"] * len(TERMINAL_STATUSES))
    where.append(f"lower(trim(visits.status)) not in ({placeholders})")
    params += sorted(TERMINAL_STATUSES)

    if city_id:
        where.append("visits.city_id = %s")
        params.append(city_id)
    if state_id:
        where.append("visits.state_id = %s")
        params.append(state_id)
    if slot:
        where.append("visits.slot_time = %s")
        params.append(slot)

    sql = f"""
        select visits.id, visits.order_id, visits.task_id, visits.order_number,
               visits.main_patient_name, visits.status, visits.slot_date, visits.slot_time,
               visits.phlebo_id, visits.lat, visits.lng, visits.city_id, visits.state_id,
               visits.patient_pincode,
               cities.city_name as city_name,
               (select areacodes.areacode_name
                  from pincodes
                  inner join areacodes on areacodes.id = pincodes.area_code_id
                       and areacodes.deleted_at is null
                  where pincodes.pincode = visits.patient_pincode
                        and pincodes.deleted_at is null
                  limit 1) as zone_name,
               concat(coalesce(phlebos.first_name, ''), ' ', coalesce(phlebos.last_name, '')) as phlebo_name,
               phlebos.phone as phlebo_phone
        from visits
        left join cities on cities.id = visits.city_id and cities.deleted_at is null
        left join phlebos on phlebos.id = visits.phlebo_id and phlebos.deleted_at is null
        where {' and '.join(where)}
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def fetch_phlebo_locations(customer_id):
    """Live phlebo positions from the same Firebase path the phlebo app writes to."""
    try:
        resp = requests.get(f"{FIREBASE_LOCATIONS}/{int(customer_id)}.json",
                            timeout=FIREBASE_TIMEOUT_SECONDS)
        payload = resp.json() if resp.ok else None
    except (requests.RequestException, ValueError):
        return {}

    if not isinstance(payload, dict):
        return {}

    locations = {}
    for phlebo_id, loc in payload.items():
        if not isinstance(loc, dict):
            continue
        # Written as latitude/longitude on some paths and lat/lng on others.
        lat = loc.get('latitude', loc.get('lat'))
        lng = loc.get('longitude', loc.get('lng'))
        if lat is not None and lng is not None:
            locations[int(phlebo_id)] = (float(lat), float(lng))

    return locations


def fetch_reached_times(conn, visit_ids):
    """When each on-site phlebo actually arrived - the moment dwell time counts from."""
    if not visit_ids:
        return {}

    placeholders = ",".join(["%s"] * len(visit_ids))
    with conn.cursor() as cur:
        cur.execute(f"""
            select visit_id, max(created_at) as reached_at
            from visit_status
            where visit_id in ({placeholders}) and lower(trim(status)) = 'phlebo reached'
            group by visit_id
        """, list(visit_ids))
        return {int(r['visit_id']): r['reached_at'] for r in cur.fetchall()}


def fetch_cached_etas(conn, visit_ids):
    if not visit_ids:
        return {}

    placeholders = ",".join(["%s"] * len(visit_ids))
    cutoff = (pd.Timestamp.now() - pd.Timedelta(minutes=ETA_CACHE_MINUTES)).strftime('%Y-%m-%d %H:%M:%S')
    with conn.cursor() as cur:
        cur.execute(f"""
            select visit_id, eta_minutes, distance_meters, traffic_aware
            from visit_tat_eta_cache
            where visit_id in ({placeholders}) and checked_at >= %s
        """, list(visit_ids) + [cutoff])
        return {
            int(r['visit_id']): {
                "minutes": int(r['eta_minutes']),
                "meters": int(r['distance_meters']) if r['distance_meters'] is not None else None,
                "trafficAware": bool(r['traffic_aware']),
                "source": "cached",
            }
            for r in cur.fetchall()
        }


def store_etas(conn, etas, phlebo_by_visit):
    """Best-effort write into the cache PHP also reads - a failure here must not
    fail a request that already has its answer."""
    if not etas:
        return

    now = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with conn.cursor() as cur:
            for visit_id, eta in etas.items():
                cur.execute("""
                    insert into visit_tat_eta_cache
                        (visit_id, phlebo_id, eta_minutes, distance_meters, traffic_aware,
                         checked_at, created_at, updated_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s)
                    on duplicate key update
                        phlebo_id = values(phlebo_id), eta_minutes = values(eta_minutes),
                        distance_meters = values(distance_meters),
                        traffic_aware = values(traffic_aware),
                        checked_at = values(checked_at), updated_at = values(updated_at)
                """, (visit_id, phlebo_by_visit[visit_id], eta['minutes'], eta['meters'],
                      eta['trafficAware'], now, now, now))
        conn.commit()
    except Exception:
        pass


def traffic_eta_minutes(origin, destination):
    """Traffic-aware driving minutes from Google Distance Matrix.

    Returns {"minutes", "meters", "trafficAware"} or None. A missing ETA must
    read as "unknown" to the caller, never as zero - an unreachable Maps API is
    a gap in what we know, not evidence the phlebo is already there."""
    key = os.environ.get('GOOGLE_MAPS_API_KEY')
    if not key:
        return None

    try:
        resp = requests.get(MAPS_ENDPOINT, params={
            "origins": origin,
            "destinations": destination,
            "mode": "driving",
            # departure_time is what makes Google return duration_in_traffic at
            # all; without it the answer is free-flow and ignores congestion.
            "departure_time": "now",
            "traffic_model": "best_guess",
            "key": key,
        }, timeout=MAPS_TIMEOUT_SECONDS)
    except requests.RequestException:
        return None

    if not resp.ok:
        return None

    element = (resp.json().get('rows') or [{}])[0].get('elements', [{}])[0]
    if element.get('status') != 'OK':
        return None

    in_traffic = element.get('duration_in_traffic')
    duration = in_traffic or element.get('duration')
    if not duration:
        return None

    return {
        "minutes": math.ceil(duration['value'] / 60),
        "meters": (element.get('distance') or {}).get('value'),
        "trafficAware": bool(in_traffic),
        "source": "fresh",
    }


def risk_from_slack(slack_minutes):
    """Logistic curve on slack - 50% at exactly zero minutes to spare, rising as
    the projected finish slides past the end of the slot."""
    return 100 / (1 + math.exp(slack_minutes / 12))


def live_risk(visit, eta=None):
    """The two signals the trained model cannot see: live traffic and field
    progress. `eta` is a pre-fetched Distance Matrix result (or None) so the
    caller can batch and cache the Google calls for a whole list at once.

    Returns a dict describing what was measured, or None when there is nothing
    live to measure - in which case the caller falls back to the model's score.

    Scope is today's open visits, matching the PHP
    /operations-overview/tat-breach-risk endpoint. "Today" bounds the date, not
    the time: a 18:30 slot still sitting untouched at 20:30 is exactly what the
    desk needs to see, so an already-passed slot scores higher, never null. Only
    a *different day* is dropped - comparing a live ETA to a deadline that died
    yesterday is noise, and would pin every stale row at 100%.

    Slot times are wall-clock local, so "now" is read the same way."""
    slot_end = _slot_end(visit.get('slot_date'), visit.get('slot_time'))
    if slot_end is None:
        return None

    now = pd.Timestamp.now()
    if slot_end.date() != now.date():
        return None

    status = (visit.get('status') or '').strip().lower()
    if status in TERMINAL_STATUSES:
        return None

    minutes_left = (slot_end - now).total_seconds() / 60

    breached = minutes_left < 0
    started = status == 'phlebo enroute'

    # Field progress: already on site, so there is no travel left to predict -
    # what matters is how long they have been standing there uncollected.
    if status == 'phlebo reached':
        dwell = visit.get('dwell_minutes')
        slack = minutes_left
        if dwell is not None:
            slack = min(minutes_left, REASONABLE_COLLECTION_MINUTES - float(dwell))
        return {
            "riskPercent": round(risk_from_slack(slack), 1),
            "basis": "on_site_dwell",
            "breached": breached,
            "slackMinutes": round(slack),
            "minutesToSlotEnd": round(minutes_left),
            "dwellMinutes": dwell,
        }

    # Past the deadline with nothing to show for it. One that never set off has
    # already failed no matter what traffic says, so it gets no grace at all;
    # one genuinely enroute keeps earning a real ETA for another hour in case it
    # is nearly there. Either way this needs no Google call - it is late now,
    # and the curve saturates the longer it drags on, so an hour overdue and six
    # hours overdue don't collapse to the same flat number.
    if breached and (not started or minutes_left < -WINDOW_BEHIND_MINUTES):
        slack = minutes_left if started else minutes_left - ASSIGNMENT_OVERHEAD_MINUTES
        return {
            "riskPercent": round(risk_from_slack(slack), 1),
            "basis": "overdue_no_progress",
            "breached": True,
            "slackMinutes": round(slack),
            "minutesToSlotEnd": round(minutes_left),
            "minutesOverdue": round(-minutes_left),
        }

    if eta is None:
        return None

    # Field progress again: someone still sitting at "assigned" has not set off,
    # so their travel has not started counting down yet.
    overhead = 0 if started else ASSIGNMENT_OVERHEAD_MINUTES
    slack = minutes_left - eta['minutes'] - overhead
    meters = eta.get('meters')

    return {
        "riskPercent": round(risk_from_slack(slack), 1),
        "basis": "enroute" if started else "assigned_not_moving",
        "breached": breached,
        "slackMinutes": round(slack),
        "minutesToSlotEnd": round(minutes_left),
        "etaMinutes": eta['minutes'],
        "distanceKm": round(meters / 1000, 1) if meters is not None else None,
        "trafficAware": eta['trafficAware'],
        "etaSource": eta.get('source'),
    }


_bundle = None


def load_bundle():
    """Loaded once per process - a 25-row list shouldn't re-read the model 25 times."""
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_one(visit, bundle=None):
    bundle = bundle or load_bundle()
    model, feature_names = bundle['model'], bundle['feature_names']

    base = build_base_features(visit)
    if base['day_of_week'] is None or base['slot_hour'] is None:
        return None  # missing or unparseable slot_date / slot_time

    base['phlebo_breach_rate'] = bundle['phlebo_rates'].get(str(visit.get('phlebo_id')), bundle['overall_rate'])
    base['zone_breach_rate'] = bundle['zone_rates'].get(str(visit.get('patient_pincode')), bundle['overall_rate'])

    df = pd.DataFrame([base])
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str)

    X = pd.get_dummies(df[FEATURE_COLUMNS], columns=CATEGORICAL_COLUMNS)
    X = X.reindex(columns=feature_names, fill_value=0)

    return float(model.predict_proba(X)[0, 1])


bp = Blueprint('delay', __name__)


def customer_id_from_token():
    """The customer_id claim out of the caller's JWT - the same token tymon's
    JWTAuth issues on the PHP side, verified here with the shared JWT_SECRET.

    This is the only place customer_id may come from. Taking it from the request
    body instead would let anyone reach any customer's list just by changing a
    number, and these rows carry patient names and phone numbers.

    Returns (customer_id, error_message)."""
    secret = os.environ.get('JWT_SECRET')
    if not secret:
        return None, "JWT_SECRET is not configured on this service"

    header = request.headers.get('Authorization', '')
    # The scheme is case-insensitive per RFC 7235, and callers do send "bearer"
    # lowercase - matching only "Bearer " rejected valid tokens.
    if not header[:7].lower() == 'bearer ':
        return None, "Token is required"

    try:
        claims = jwt.decode(header[7:].strip(), secret, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return None, "Token has expired"
    except jwt.InvalidTokenError:
        return None, "Token is invalid"

    customer_id = claims.get('customer_id')
    if not customer_id:
        return None, "Token carries no customer_id"

    return customer_id, None


def collect_etas(conn, candidates, locations):
    """One traffic lookup per visit that still has travel ahead of it, cache first.

    Skipped on purpose: anyone already on site (nothing left to drive) and anyone
    so far past their deadline that a fresh ETA no longer changes what the desk
    should do - see live_risk. Those are exactly the calls not worth buying."""
    pairs = {}
    for v in candidates:
        if v['reached'] or v['skip_eta'] or not v['phlebo_id']:
            continue
        origin = locations.get(int(v['phlebo_id']))
        if not origin or not v['lat'] or not v['lng']:
            continue
        pairs[int(v['id'])] = (origin, (v['lat'], v['lng']), int(v['phlebo_id']))

    if not pairs:
        return {}

    etas = fetch_cached_etas(conn, list(pairs))
    fresh = {}
    for visit_id, (origin, dest, _) in pairs.items():
        if visit_id in etas:
            continue
        eta = traffic_eta_minutes(f"{origin[0]},{origin[1]}", f"{dest[0]},{dest[1]}")
        if eta:
            fresh[visit_id] = eta

    store_etas(conn, fresh, {vid: pairs[vid][2] for vid in fresh})
    etas.update(fresh)
    return etas


def breach_line(live):
    if not live:
        return None
    if live['breached']:
        return f"Breached {abs(live['minutesToSlotEnd'])} min ago"
    return f"Predicted breach in ~{live['minutesToSlotEnd']} min"


def fetch_fresh_risk_ids(conn, visit_ids):
    """visit_ids already scored within RISK_CACHE_MINUTES - /predict-delay-batch
    skips these rather than spending a fresh model + traffic check on them."""
    if not visit_ids:
        return set()

    placeholders = ",".join(["%s"] * len(visit_ids))
    cutoff = (pd.Timestamp.now() - pd.Timedelta(minutes=RISK_CACHE_MINUTES)).strftime('%Y-%m-%d %H:%M:%S')
    with conn.cursor() as cur:
        cur.execute(f"""
            select visit_id from visit_delay_risk
            where visit_id in ({placeholders}) and computed_at >= %s
        """, list(visit_ids) + [cutoff])
        return {int(r['visit_id']) for r in cur.fetchall()}


def store_delay_risk(conn, rows):
    """Best-effort upsert into visit_delay_risk, the table order-list reads by
    visit_id. Same reasoning as store_etas() above: a write failure here must
    not fail a batch run that already has its answers."""
    if not rows:
        return

    now = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute("""
                    insert into visit_delay_risk
                        (visit_id, customer_id, risk_percent, severity, risk_label,
                         breach_line, eta_minutes, computed_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s)
                    on duplicate key update
                        customer_id = values(customer_id), risk_percent = values(risk_percent),
                        severity = values(severity), risk_label = values(risk_label),
                        breach_line = values(breach_line), eta_minutes = values(eta_minutes),
                        computed_at = values(computed_at)
                """, (r['visitId'], r['customerId'], r['riskPercent'], r['severity'],
                      r['riskLabel'], r['breachLine'], r['etaMinutes'], now))
        conn.commit()
    except Exception:
        pass


@bp.route('/predict-delay', methods=['POST'])
def predict_delay_endpoint():
    try:
        customer_id, auth_error = customer_id_from_token()
        if auth_error:
            return jsonify({"success": False, "message": auth_error}), 401

        data = request.json or {}
        limit = min(int(data.get('limit', 25)), 50)
        min_risk = int(data.get('min_risk', 30))
        bundle = load_bundle()

        conn = _db_connection()
        try:
            rows = fetch_todays_open_visits(conn, customer_id, data.get('city'),
                                            data.get('state'), data.get('slot'))
            now = pd.Timestamp.now()

            candidates = []
            for row in rows:
                slot_end = _slot_end(row['slot_date'], row['slot_time'])
                if slot_end is None:
                    continue
                minutes_left = (slot_end - now).total_seconds() / 60
                # Only the far side is bounded - see WINDOW_AHEAD_MINUTES.
                if minutes_left > WINDOW_AHEAD_MINUTES:
                    continue

                status = (row['status'] or '').strip().lower()
                started = status == 'phlebo enroute'
                row['reached'] = status == 'phlebo reached'
                row['skip_eta'] = minutes_left < 0 and (
                    not started or minutes_left < -WINDOW_BEHIND_MINUTES)
                row['slot_end'] = slot_end
                candidates.append(row)

            # Soonest deadline first, so a cap keeps the most overdue rows.
            candidates.sort(key=lambda r: r['slot_end'])
            candidates = candidates[:limit]

            reached_at = fetch_reached_times(
                conn, [int(v['id']) for v in candidates if v['reached']])
            locations = fetch_phlebo_locations(customer_id)
            etas = collect_etas(conn, candidates, locations)
        finally:
            conn.close()

        at_risk = []
        counts = {"high": 0, "medium": 0, "low": 0, "breached": 0, "total": 0}

        for row in candidates:
            visit_id = int(row['id'])
            visit = dict(row)
            if row['reached'] and visit_id in reached_at:
                visit['dwell_minutes'] = (now - pd.Timestamp(reached_at[visit_id])).total_seconds() / 60

            risk = predict_one(visit, bundle)
            if risk is None:
                continue

            history_percent = round(risk * 100, 1)
            live = live_risk(visit, etas.get(visit_id))

            # Whichever signal is more alarmed wins. A visit must not be called
            # safe just because the other had no opinion: history can't see a
            # phlebo stuck in traffic right now, and traffic can't see that this
            # slot and this zone have been running late for months.
            final = max(history_percent, live['riskPercent']) if live else history_percent
            if final < min_risk:
                continue

            severity = "high" if final >= 50 else ("medium" if final >= 25 else "low")
            counts[severity] += 1
            counts['total'] += 1
            if live and live['breached']:
                counts['breached'] += 1

            city = (row['city_name'] or '').strip()
            zone = (row['zone_name'] or '').strip()
            phlebo = (row['phlebo_name'] or '').strip()
            patient = (row['main_patient_name'] or '').strip()
            booking_id = row['task_id'] or row['order_number']
            city_zone = ' · '.join(x for x in (city, zone) if x)

            at_risk.append({
                "visitId": visit_id,
                "orderId": row['order_id'],
                "bookingId": booking_id,
                "patient": patient or None,
                "city": city or None,
                "zone": zone or None,
                "cityZone": city_zone or None,
                "title": ', '.join(x for x in (f"{booking_id} · {patient}".strip(' ·'), city) if x),
                "phlebotomist": phlebo or None,
                "phleboId": int(row['phlebo_id']) if row['phlebo_id'] else None,
                "phleboPhone": row['phlebo_phone'],
                "assigned": bool(row['phlebo_id']),
                "status": row['status'],
                "slotDate": str(row['slot_date']),
                "slotTime": row['slot_time'],
                "riskPercent": final,
                "riskLabel": f"{final}% risk",
                "severity": severity,
                "breached": bool(live and live['breached']),
                "breachLine": breach_line(live),
                "slackMinutes": live['slackMinutes'] if live else None,
                "etaMinutes": live.get('etaMinutes') if live else None,
                "distance": live.get('distanceKm') if live else None,
                "unit": "km",
                "trafficAware": bool(live and live.get('trafficAware')),
                "etaSource": live.get('etaSource') if live else None,
                "reached": bool(row['reached']),
                "dwellMinutes": round(visit['dwell_minutes']) if 'dwell_minutes' in visit else None,
                "basis": live['basis'] if live else "history_only",
                "signals": {"history": {"riskPercent": history_percent}, "live": live},
            })

        at_risk.sort(key=lambda r: -r['riskPercent'])

        return jsonify({
            "success": True,
            "data": {
                "date": now.strftime('%Y-%m-%d'),
                "generatedAt": now.strftime('%Y-%m-%d %H:%M:%S'),
                "unit": "km",
                "counts": counts,
                "atRisk": at_risk,
            },
        })
    except FileNotFoundError:
        return jsonify({"error": "model not trained yet - run `python delay_prediction.py train` first"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/predict-delay-batch', methods=['POST'])
def predict_delay_batch_endpoint():
    """Internal counterpart to /predict-delay, for keeping visit_delay_risk (the
    table order-list reads by visit_id) filled in - triggered by a button in
    the dashboard/order-list, not a schedule.

    Takes no body: it finds its own candidates exactly the way /predict-delay
    does, via the same fetch_todays_open_visits() call, scoped to the caller's
    own customer_id from the JWT. The two endpoints deliberately share that one
    function instead of each parsing "today's open visits" their own way -
    two independent readings of that would eventually disagree.

    The one difference from /predict-delay: this scores and stores every
    candidate in the window, not just the riskiest `limit` above `min_risk` -
    order-list wants a badge on any of today's visits that has one, not only
    the worst 25. Visits already scored within RISK_CACHE_MINUTES are skipped
    - see fetch_fresh_risk_ids() - so reposting on every poll doesn't re-run
    the model or spend a fresh Maps call on rows that haven't gone stale yet.

    Nothing is returned but a count: the table is the answer, there is no
    dashboard waiting on this response."""
    try:
        customer_id, auth_error = customer_id_from_token()
        if auth_error:
            return jsonify({"success": False, "message": auth_error}), 401

        bundle = load_bundle()
        now = pd.Timestamp.now()

        conn = _db_connection()
        try:
            rows = fetch_todays_open_visits(conn, customer_id)

            candidates = []
            for row in rows:
                slot_end = _slot_end(row['slot_date'], row['slot_time'])
                if slot_end is None:
                    continue
                minutes_left = (slot_end - now).total_seconds() / 60
                if minutes_left > WINDOW_AHEAD_MINUTES:
                    continue

                status = (row['status'] or '').strip().lower()
                started = status == 'phlebo enroute'
                row['reached'] = status == 'phlebo reached'
                row['skip_eta'] = minutes_left < 0 and (
                    not started or minutes_left < -WINDOW_BEHIND_MINUTES)
                candidates.append(row)

            fresh_ids = fetch_fresh_risk_ids(conn, [int(v['id']) for v in candidates])
            pending = [v for v in candidates if int(v['id']) not in fresh_ids]

            reached_at = fetch_reached_times(
                conn, [int(v['id']) for v in pending if v['reached']])
            locations = fetch_phlebo_locations(customer_id)
            etas = collect_etas(conn, pending, locations)

            results = []
            for row in pending:
                visit_id = int(row['id'])
                visit = dict(row)
                if row['reached'] and visit_id in reached_at:
                    visit['dwell_minutes'] = (now - pd.Timestamp(reached_at[visit_id])).total_seconds() / 60

                risk = predict_one(visit, bundle)
                if risk is None:
                    continue

                history_percent = round(risk * 100, 1)
                live = live_risk(visit, etas.get(visit_id))
                final = max(history_percent, live['riskPercent']) if live else history_percent
                severity = "high" if final >= 50 else ("medium" if final >= 25 else "low")

                results.append({
                    "visitId": visit_id,
                    "customerId": customer_id,
                    "riskPercent": final,
                    "riskLabel": f"{final}% risk",
                    "severity": severity,
                    "breachLine": breach_line(live),
                    "etaMinutes": live.get('etaMinutes') if live else None,
                })

            store_delay_risk(conn, results)
        finally:
            conn.close()

        return jsonify({
            "success": True,
            "data": {
                "candidates": len(candidates),
                "computed": len(results),
                "skippedFresh": len(candidates) - len(pending),
                "computedAt": now.strftime('%Y-%m-%d %H:%M:%S'),
            },
        })
    except FileNotFoundError:
        return jsonify({"error": "model not trained yet - run `python delay_prediction.py train` first"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train()
    else:
        # Standalone mode - app.py mounts this blueprint instead when run together.
        app = Flask(__name__)
        app.register_blueprint(bp)
        port = int(os.environ.get("PORT", 5003))
        app.run(host='0.0.0.0', port=port)
