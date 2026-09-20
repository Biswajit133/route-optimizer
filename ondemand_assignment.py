"""
On-demand assignment - pick a phlebotomist the moment the order is created.

Every other service in this repo answers "who *should* take this" and leaves
the writing to PHP. This one writes. That is the whole point: an on-demand
order has just been created and nobody has been told anything yet, so there is
no promise to break by deciding immediately - and a booking that sits
unassigned while a human notices it is the failure this is meant to prevent.

Because it writes, two things matter more here than anywhere else in this repo:

  correctness of scope   the visit is looked up by id *and* by the customer_id
                         in the caller's token. Without the second half, any
                         caller could staff any customer's booking.
  correctness under load two orders created in the same second must not both
                         be handed the same phlebotomist. The write is done
                         inside a transaction that locks that phlebotomist's
                         rows for the slot first, and the losing request falls
                         through to its next candidate. See try_assign().

Only `phlebo_id` is written. The status transition and its visit_status audit
row stay with PHP, which owns that pipeline - this service has no business
inventing history in a table it does not otherwise touch.

How the choice is made
----------------------
The candidate pool is whoever is reporting a live position for this customer in
Firebase and is not clocked out - the same pool /reallocation-suggestions uses.
A phlebotomist whose app is not reporting will not be given on-demand work.
That is a real gap, and a deliberate one: sending someone whose position is
unknown is a guess dressed up as a decision.

Candidates are then filtered hard - the right serving area, not already booked
in this exact slot, still under capacity - and ranked one of two ways:

  imminent slot   within the next two hours, so where they are standing now
                  actually predicts when they arrive. Ranked by a traffic-aware
                  Google ETA, and only those who can arrive before the slot ends
                  are eligible at all.
  distant slot    further out than that, where a current position predicts
                  nothing - they will have moved several times before the slot
                  opens. Ranked by who is carrying the least work that day, and
                  no Google call is made, because buying an ETA for a journey
                  that starts in five hours is buying noise.

If nobody clears the bar the visit is left unassigned with a reason, on purpose.
Handing an order to someone who cannot make it turns a staffing problem into a
broken promise, and the desk can no longer see which is which.

Data access
-----------
Reads visits and writes visits.phlebo_id via PyMySQL, using
DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD from .env - never hardcode
credentials here. GOOGLE_MAPS_API_KEY is optional; without it an imminent slot
falls back to straight-line proximity and the response says so.

Usage
-----
    python ondemand_assignment.py

Needs `pip install flask pymysql requests python-dotenv PyJWT`.
"""

import math
import json
import os
from datetime import datetime, timedelta

import jwt
import pymysql
import requests
from dotenv import load_dotenv
from flask import Blueprint, Flask, request, jsonify

load_dotenv()

bp = Blueprint('ondemand', __name__)

FIREBASE_LOCATIONS = "https://hsc-phlebotomist-app-default-rtdb.firebaseio.com/phlebo_locations"
FIREBASE_TIMEOUT_SECONDS = 5
MAPS_ENDPOINT = "https://maps.googleapis.com/maps/api/distancematrix/json"
MAPS_TIMEOUT_SECONDS = 6

# Minutes on site to draw the sample - used to work out when a candidate would
# be free again, and therefore which of their own visits this would push.
COLLECTION_MINUTES = 15

# Inside this window, where someone is standing now predicts when they arrive,
# so the ranking is worth a traffic call. Beyond it they will have moved several
# times before the slot opens and the position means nothing.
IMMINENT_WINDOW_MINUTES = 120

# Order creation is a synchronous flow with a customer waiting on it, so only
# the nearest few candidates are worth a billable round-trip to Google.
MAX_ETA_CANDIDATES = 5

# Statuses where the visit is finished with - it occupies nobody's slot and
# counts towards nobody's load. Same set the rest of this repo treats as done.
TERMINAL_STATUSES = {
    'sample collected', 'completed', 'enroute to lab', 'enroute to ship',
    'enroute to runner', 'dropped in lab', 'dropped in courier',
    'sample delivered to lab', 'collected', 'collecting', 'cancelled',
}


def haversine_meters(a, b):
    R = 6371000
    lat1, lon1 = a
    lat2, lon2 = b

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    x = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    return int(2 * R * math.atan2(math.sqrt(x), math.sqrt(1 - x)))


def parse_int_list(value):
    if isinstance(value, list):
        return [int(x) for x in value if str(x).strip() not in ('', 'None')]

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [int(x) for x in parsed if str(x).strip() not in ('', 'None')]
        except (ValueError, TypeError):
            pass

    return []


def coerce_point(lat, lng):
    """(lat, lng) as floats, or None. 0/0 is rejected as well as null - in this
    data it always means "never captured", never a real location."""
    if lat in (None, '', 0, '0') or lng in (None, '', 0, '0'):
        return None
    try:
        point = (float(lat), float(lng))
    except (TypeError, ValueError):
        return None

    if point == (0.0, 0.0) or not (-90 <= point[0] <= 90) or not (-180 <= point[1] <= 180):
        return None

    return point


def slot_bounds(slot_date, slot_time):
    """("2026-09-14", "14:00 - 15:00") -> (start, end). A bare start time means a
    30-minute window, the same rule the rest of the platform uses."""
    if not slot_date or not slot_time:
        return None, None

    date_str = slot_date.strftime('%Y-%m-%d') if hasattr(slot_date, 'strftime') else str(slot_date)
    parts = str(slot_time).split('-')

    try:
        start = datetime.strptime(f"{date_str} {parts[0].strip()}", '%Y-%m-%d %H:%M')
    except ValueError:
        return None, None

    if len(parts) == 2:
        try:
            end = datetime.strptime(f"{date_str} {parts[1].strip()}", '%Y-%m-%d %H:%M')
            if end > start:
                return start, end
        except ValueError:
            pass

    return start, start + timedelta(minutes=30)


def db_connection():
    return pymysql.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", 3306)),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
        cursorclass=pymysql.cursors.DictCursor,
    )


def customer_id_from_token():
    """customer_id and country_id out of the caller's JWT - the same token
    tymon issues on the PHP side. customer_id is the only source for it, and
    here it is load-bearing twice over: it scopes the row that gets read *and*
    the row that gets written. country_id decides the phlebo-matching rule in
    country_settings (radius-from-home-base vs area/city) - a token without
    one just means that lookup is skipped and the area/city match is used, the
    same as it always was.

    Returns (customer_id, country_id, error_message)."""
    secret = os.environ.get('JWT_SECRET')
    if not secret:
        return None, None, "JWT_SECRET is not configured on this service"

    header = request.headers.get('Authorization', '')
    # The scheme is case-insensitive per RFC 7235, and callers do send "bearer"
    # lowercase - matching only "Bearer " rejected valid tokens.
    if not header[:7].lower() == 'bearer ':
        return None, None, "Token is required"

    try:
        claims = jwt.decode(header[7:].strip(), secret, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return None, None, "Token has expired"
    except jwt.InvalidTokenError:
        return None, None, "Token is invalid"

    customer_id = claims.get('customer_id')
    if not customer_id:
        return None, None, "Token carries no customer_id"

    return customer_id, claims.get('country_id'), None


def fetch_visit(conn, visit_id, customer_id):
    """The visit to staff. Scoped by customer_id as well as id - without that,
    a caller could staff, and so learn about, any customer's booking."""
    with conn.cursor() as cur:
        cur.execute("""
            select visits.id, visits.order_id, visits.task_id, visits.order_number,
                   visits.main_patient_name, visits.status, visits.slot_date,
                   visits.slot_time, visits.phlebo_id, visits.lat, visits.lng,
                   visits.city_id, visits.patient_pincode,
                   (select pincodes.area_code_id
                      from pincodes
                      where pincodes.pincode = visits.patient_pincode
                            and pincodes.deleted_at is null
                      limit 1) as areacode_id
            from visits
            where visits.id = %s and visits.customer_id = %s
              and visits.is_active = 1 and visits.deleted_at is null
        """, (visit_id, customer_id))
        return cur.fetchone()


def fetch_day_board(conn, customer_id, slot_date, exclude_visit_id):
    """Everything else live on the same day, so a candidate can be judged on
    what they are already holding: slot conflicts, load, and what this job
    would push."""
    with conn.cursor() as cur:
        cur.execute("""
            select id, phlebo_id, slot_date, slot_time, status
            from visits
            where customer_id = %s and slot_date = %s and id <> %s
              and is_active = 1 and deleted_at is null and phlebo_id is not null
        """, (customer_id, slot_date, exclude_visit_id))

        holdings = {}
        for row in cur.fetchall():
            if (row['status'] or '').strip().lower() in TERMINAL_STATUSES:
                continue
            holdings.setdefault(int(row['phlebo_id']), []).append(row)

        return holdings


def phlebo_directory(conn, phlebo_ids):
    """Names, phones, and everything eligible_phlebos() needs to judge a
    candidate: serving_areacode_id/serving_city_id (aliased from the actual
    serving_area_code/serving_city columns) for the area/city match, plus
    use_address_as_home_base, start_lat/start_long, homebase_id and
    serving_radius for the radius-from-home-base match country_settings can
    switch on instead. All optional: if this deployment's schema lacks them
    the query is retried without, and everyone is then allowed everywhere -
    the same fallback /auto-assign already applies when a visit matches
    nobody's area."""
    if not phlebo_ids:
        return {}

    placeholders = ",".join(["%s"] * len(phlebo_ids))
    base = f"""
        select id, concat(coalesce(first_name, ''), ' ', coalesce(last_name, '')) as name,
               phone{{extra}}
        from phlebos
        where id in ({placeholders}) and deleted_at is null
    """
    extra_cols = (
        ", serving_area_code as serving_areacode_id, serving_city as serving_city_id, "
        "use_address_as_home_base, start_lat, start_long, homebase_id, serving_radius"
    )

    for extra in (extra_cols, ""):
        try:
            with conn.cursor() as cur:
                cur.execute(base.format(extra=extra), list(phlebo_ids))
                return {int(r['id']): r for r in cur.fetchall()}
        except pymysql.err.MySQLError:
            continue

    return {}


def live_locations(customer_id):
    """Where every phlebotomist is, from the path the phlebo app writes to."""
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
        point = coerce_point(loc.get('latitude', loc.get('lat')),
                             loc.get('longitude', loc.get('lng')))
        if point is None:
            continue
        try:
            locations[int(phlebo_id)] = {
                "point": point, "status": loc.get('status'),
            }
        except (TypeError, ValueError):
            continue

    return locations


def traffic_eta(origin, destination):
    """Traffic-aware driving minutes, or None. A missing answer means we do not
    know, never that they are already there."""
    key = os.environ.get('GOOGLE_MAPS_API_KEY')
    if not key:
        return None

    try:
        resp = requests.get(MAPS_ENDPOINT, params={
            "origins": f"{origin[0]},{origin[1]}",
            "destinations": f"{destination[0]},{destination[1]}",
            "mode": "driving",
            # departure_time is what makes Google return duration_in_traffic;
            # without it the answer is free-flow and ignores congestion.
            "departure_time": "now",
            "traffic_model": "best_guess",
            "key": key,
        }, timeout=MAPS_TIMEOUT_SECONDS)
    except requests.RequestException:
        return None

    if not resp.ok:
        return None

    try:
        element = (resp.json().get('rows') or [{}])[0].get('elements', [{}])[0]
    except (ValueError, IndexError):
        return None

    if element.get('status') != 'OK':
        return None

    duration = element.get('duration_in_traffic') or element.get('duration')
    if not duration:
        return None

    return {
        "minutes": math.ceil(duration['value'] / 60),
        "km": round((element.get('distance') or {}).get('value', 0) / 1000, 1),
    }


def serves_area(info, areacode_id, city_id):
    areas = parse_int_list(info.get('serving_areacode_id', []))
    cities = parse_int_list(info.get('serving_city_id', []))

    if areacode_id and areacode_id in areas:
        return True
    if city_id and city_id in cities:
        return True

    return False


# ---------------------------------------------------------------------------
# Phlebo-matching rule: radius from home base, or area/city.
#
# country_settings.phlebo_assign_via_radius (per country_id, from the caller's
# JWT) picks which one applies here too - same rule dynamic_reallocation.py
# uses for /reallocate and /reallocation-suggestions, duplicated locally so
# this file keeps no import-time dependency on that one. Radius mode checks
# the straight-line distance from each phlebo's home base to the visit, in
# the country's own unit (country_settings.distance_value: km or mile) - not
# their live position, this is about coverage, not who is closest right now.
# ---------------------------------------------------------------------------

def country_settings(conn, country_id):
    """The per-country phlebo-matching rule. None (missing country_id, no
    matching row, or a DB error) means "no rule configured" - callers treat
    that exactly like phlebo_assign_via_radius being off, i.e. the original
    area/city match, so an unconfigured country never leaves a visit
    unstaffed over this."""
    if not country_id:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                select phlebo_assign_via_radius, phlebo_assign_via_serving_city, distance_value
                from country_settings where country_id = %s
            """, (country_id,))
            return cur.fetchone()
    except pymysql.err.MySQLError:
        return None


def radius_meters(value, unit):
    """serving_radius, in the country's own distance_value unit, to meters -
    or None if it isn't a usable positive number."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if (unit or '').strip().lower().startswith('mile'):
        return value * 1609.344
    return value * 1000.0  # km is the only other value distance_value takes


def home_base_point(phlebo, homebase_locations):
    """(lat, lng) of wherever this phlebo is based from - their own start
    point if use_address_as_home_base is set, else the phlebo_location_master
    row homebase_id points at. None if neither resolves to a real point."""
    if phlebo.get('use_address_as_home_base'):
        return coerce_point(phlebo.get('start_lat'), phlebo.get('start_long'))

    base = None
    if phlebo.get('homebase_id'):
        try:
            base = homebase_locations.get(int(phlebo['homebase_id']))
        except (TypeError, ValueError):
            base = None
    if not base:
        return None
    return coerce_point(base.get('latitude'), base.get('longitude'))


def homebase_locations_for(conn, phlebos):
    """phlebo_location_master rows for every homebase_id these phlebos use -
    only the ones with use_address_as_home_base off actually need one."""
    ids = set()
    for p in phlebos:
        if p.get('use_address_as_home_base') or not p.get('homebase_id'):
            continue
        try:
            ids.add(int(p['homebase_id']))
        except (TypeError, ValueError):
            continue

    if not ids:
        return {}

    placeholders = ",".join(["%s"] * len(ids))
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                select id, latitude, longitude from phlebo_location_master
                where id in ({placeholders}) and deleted_at is null
            """, list(ids))
            return {int(r['id']): r for r in cur.fetchall()}
    except pymysql.err.MySQLError:
        return {}


def eligible_by_radius(phlebo, visit_point, homebase_locations, unit):
    """Visit falls inside this phlebo's serving_radius around their home
    base."""
    home = home_base_point(phlebo, homebase_locations)
    if home is None or visit_point is None:
        return False
    limit = radius_meters(phlebo.get('serving_radius'), unit)
    if limit is None:
        return False
    return haversine_meters(home, visit_point) <= limit


def eligible_for_visit(info, visit, visit_point, settings, homebase_locations):
    """Whether this phlebo may be considered for this visit at all - the
    radius check above when the visit's country has phlebo_assign_via_radius
    on, else the original area-then-city match (serves_area)."""
    if settings and settings.get('phlebo_assign_via_radius'):
        return eligible_by_radius(info, visit_point, homebase_locations, settings.get('distance_value'))

    areacode_id = int(visit.get('areacode_id') or 0)
    city_id = int(visit.get('city_id') or 0)
    return serves_area(info, areacode_id, city_id)


def eligible_phlebos(visit, locations, holdings, directory, capacity,
                      settings=None, homebase_locations=None):
    """Everyone who could take this at all, with why the others could not.

    Radius-from-home-base or area-then-city (per country_settings), then the
    all-allowed fallback when nobody qualifies - the same precedence
    app.py's /auto-assign uses. The rejection reasons are kept because
    "nobody covers this pincode" and "everyone was already full" are
    different problems for whoever reads the queue."""
    visit_point = coerce_point(visit.get('lat'), visit.get('lng'))

    area_matched = [
        pid for pid in locations
        if eligible_for_visit(directory.get(pid, {}), visit, visit_point, settings, homebase_locations)
    ]
    # Nobody declares coverage here - fall back to everyone rather than leaving
    # the booking unstaffed over a missing configuration row.
    pool = area_matched if area_matched else list(locations)

    eligible = []
    rejected = {"clocked_out": 0, "out_of_area": len(locations) - len(pool),
                "busy_in_slot": 0, "at_capacity": 0}

    for phlebo_id in pool:
        if locations[phlebo_id].get('status') == 'clock_out':
            rejected['clocked_out'] += 1
            continue

        held = holdings.get(phlebo_id, [])
        # Nobody can be in two places at once.
        if any((h['slot_time'] or '') == (visit['slot_time'] or '') for h in held):
            rejected['busy_in_slot'] += 1
            continue

        if capacity and len(held) >= capacity:
            rejected['at_capacity'] += 1
            continue

        eligible.append(phlebo_id)

    return eligible, rejected


def would_push(held, free_again):
    """Which of a candidate's own visits this job would delay - anything whose
    window opens before they could get there, draw, and move on."""
    pushed = []
    for h in held:
        start, _ = slot_bounds(h['slot_date'], h['slot_time'])
        if start and start < free_again:
            pushed.append({"visitId": int(h['id']), "slotTime": h['slot_time']})

    return pushed


def rank_by_traffic(visit, point, now, slot_end, eligible, locations, holdings):
    """For a slot close enough that a current position means something.

    Only candidates who can actually arrive before the window closes are
    returned - an on-demand order handed to someone who cannot make it is a
    broken promise dressed up as a decision."""
    by_distance = sorted(
        eligible,
        key=lambda pid: haversine_meters(locations[pid]['point'], point))

    scored = []
    for phlebo_id in by_distance[:MAX_ETA_CANDIDATES]:
        eta = traffic_eta(locations[phlebo_id]['point'], point)
        if eta is None:
            continue

        arrival = now + timedelta(minutes=eta['minutes'])
        if arrival > slot_end:
            continue

        pushed = would_push(
            holdings.get(phlebo_id, []),
            arrival + timedelta(minutes=COLLECTION_MINUTES))

        scored.append({
            "id": phlebo_id,
            "etaMinutes": eta['minutes'],
            "distanceKm": eta['km'],
            "arrival": arrival,
            "pushes": pushed,
        })

    # Fewest of their own visits disrupted first, then soonest to arrive.
    scored.sort(key=lambda c: (len(c['pushes']), c['etaMinutes']))
    return scored


def rank_by_workload(visit, point, eligible, locations, holdings):
    """For a slot far enough out that where anyone is standing now predicts
    nothing. Whoever is carrying least that day goes first; straight-line
    proximity only breaks ties, and no Google call is bought for a journey that
    does not start for hours."""
    scored = []
    for phlebo_id in eligible:
        metres = haversine_meters(locations[phlebo_id]['point'], point)
        scored.append({
            "id": phlebo_id,
            "etaMinutes": None,
            "distanceKm": round(metres / 1000, 1),
            "arrival": None,
            "pushes": [],
            "currentLoad": len(holdings.get(phlebo_id, [])),
        })

    scored.sort(key=lambda c: (c['currentLoad'], c['distanceKm']))
    return scored


def try_assign(conn, visit_id, customer_id, phlebo_id, slot_date, slot_time):
    """Write phlebo_id, but only if this is still a fair thing to write.

    Two orders created in the same second can pick the same person, so the
    candidate's rows for this slot are locked before the write and re-checked
    inside the lock; the request that loses falls through to its next choice.
    The update is also conditional on the visit still being unstaffed, which
    makes a retry of the same call harmless.

    Returns (True, None) on success, or (False, reason)."""
    conn.begin()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                select id from visits
                where phlebo_id = %s and customer_id = %s and slot_date = %s
                  and slot_time = %s and id <> %s
                  and is_active = 1 and deleted_at is null
                for update
            """, (phlebo_id, customer_id, slot_date, slot_time, visit_id))

            clash = [r for r in cur.fetchall()]
            if clash:
                conn.rollback()
                return False, "taken_by_another_order"

            cur.execute("""
                update visits set phlebo_id = %s, updated_at = now()
                where id = %s and customer_id = %s
                  and (phlebo_id is null or phlebo_id = 0)
            """, (phlebo_id, visit_id, customer_id))

            if cur.rowcount == 0:
                conn.rollback()
                return False, "already_assigned"

        conn.commit()
        return True, None
    except Exception:
        conn.rollback()
        raise


@bp.route('/assign-now', methods=['POST'])
def assign_now_endpoint():
    try:
        customer_id, country_id, auth_error = customer_id_from_token()
        if auth_error:
            return jsonify({"success": False, "message": auth_error}), 401

        body = request.json or {}
        visit_id = body.get('visit_id')
        if not visit_id:
            return jsonify({"success": False, "message": "visit_id is required"}), 400

        try:
            visit_id = int(visit_id)
        except (TypeError, ValueError):
            return jsonify({"success": False, "message": "visit_id must be a number"}), 400

        capacity = body.get('capacity_per_phlebo')
        capacity = int(capacity) if capacity else None
        dry_run = bool(body.get('dry_run'))
        now = datetime.now()

        conn = db_connection()
        try:
            visit = fetch_visit(conn, visit_id, customer_id)
            if not visit:
                return jsonify({"success": False, "message": "Visit not found"}), 404

            if visit['phlebo_id']:
                return jsonify({"success": True, "data": {
                    "visitId": visit_id, "assigned": False,
                    "outcome": "already_assigned",
                    "phlebo": {"id": int(visit['phlebo_id'])},
                    "reason": "This visit already has a phlebotomist",
                }})

            point = coerce_point(visit['lat'], visit['lng'])
            if point is None:
                return jsonify({"success": True, "data": {
                    "visitId": visit_id, "assigned": False, "outcome": "unroutable",
                    "reason": "The visit has no usable latitude/longitude",
                }})

            slot_start, slot_end = slot_bounds(visit['slot_date'], visit['slot_time'])
            if slot_end is None:
                return jsonify({"success": True, "data": {
                    "visitId": visit_id, "assigned": False, "outcome": "no_slot",
                    "reason": f"Slot '{visit['slot_time']}' could not be read as a time window",
                }})

            locations = live_locations(customer_id)
            if not locations:
                return jsonify({"success": True, "data": {
                    "visitId": visit_id, "assigned": False, "outcome": "no_candidates",
                    "reason": "No phlebotomist is reporting a live position right now",
                }})

            holdings = fetch_day_board(conn, customer_id, visit['slot_date'], visit_id)
            directory = phlebo_directory(conn, set(locations))

            # country_settings decides which eligibility rule applies below;
            # home base locations are only fetched when radius mode needs them.
            settings = country_settings(conn, country_id)
            homebase_locations = (
                homebase_locations_for(conn, directory.values())
                if settings and settings.get('phlebo_assign_via_radius') else {}
            )

            eligible, rejected = eligible_phlebos(
                visit, locations, holdings, directory, capacity, settings, homebase_locations)

            if not eligible:
                return jsonify({"success": True, "data": {
                    "visitId": visit_id, "assigned": False, "outcome": "no_candidates",
                    "reason": "Every tracked phlebotomist is clocked out, out of area, "
                              "already booked in this slot, or at capacity",
                    "rejected": rejected,
                }})

            minutes_to_slot = (slot_start - now).total_seconds() / 60
            imminent = minutes_to_slot <= IMMINENT_WINDOW_MINUTES

            if imminent:
                ranked = rank_by_traffic(visit, point, now, slot_end, eligible,
                                         locations, holdings)
                basis = "traffic_eta"
            else:
                ranked = rank_by_workload(visit, point, eligible, locations, holdings)
                basis = "workload_balance"

            if not ranked:
                return jsonify({"success": True, "data": {
                    "visitId": visit_id, "assigned": False, "outcome": "cannot_arrive_in_time",
                    "reason": f"None of the {len(eligible)} available phlebotomists can reach "
                              f"this address before the {visit['slot_time']} slot ends",
                    "candidatesConsidered": len(eligible),
                    "rejected": rejected,
                }})

            if dry_run:
                top = ranked[0]
                info = directory.get(top['id'], {})
                return jsonify({"success": True, "data": {
                    "visitId": visit_id, "assigned": False, "outcome": "dry_run",
                    "basis": basis,
                    "wouldAssign": {
                        "id": top['id'], "name": (info.get('name') or '').strip() or None,
                        "etaMinutes": top['etaMinutes'], "distanceKm": top['distanceKm'],
                    },
                    "candidatesConsidered": len(ranked),
                }})

            # Walk the ranking: a candidate can be lost to a concurrent order
            # between being chosen and being written, and the next one down is
            # a better answer than failing the request.
            chosen, write_failures = None, []
            for candidate in ranked:
                ok, why = try_assign(conn, visit_id, customer_id, candidate['id'],
                                     visit['slot_date'], visit['slot_time'])
                if ok:
                    chosen = candidate
                    break
                if why == 'already_assigned':
                    return jsonify({"success": True, "data": {
                        "visitId": visit_id, "assigned": False,
                        "outcome": "already_assigned",
                        "reason": "Another request staffed this visit first",
                    }})
                write_failures.append(candidate['id'])

            if chosen is None:
                return jsonify({"success": True, "data": {
                    "visitId": visit_id, "assigned": False, "outcome": "no_candidates",
                    "reason": "Every candidate was taken by another order while assigning",
                    "contendedWith": write_failures,
                }})
        finally:
            conn.close()

        info = directory.get(chosen['id'], {})
        name = (info.get('name') or '').strip()

        return jsonify({
            "success": True,
            "data": {
                "visitId": visit_id,
                "assigned": True,
                "outcome": "assigned",
                "basis": basis,
                "slotDate": str(visit['slot_date']),
                "slotTime": visit['slot_time'],
                "phlebo": {
                    "id": chosen['id'],
                    "name": name or None,
                    "phone": info.get('phone'),
                    "etaMinutes": chosen['etaMinutes'],
                    "distanceKm": chosen['distanceKm'],
                },
                "arrivesBy": chosen['arrival'].strftime('%Y-%m-%d %H:%M') if chosen['arrival'] else None,
                "impact": {
                    "visitsPushed": len(chosen['pushes']),
                    "visits": chosen['pushes'],
                },
                "candidatesConsidered": len(ranked),
                "rejected": rejected,
                "reason": _reason(name, chosen, basis, visit),
            },
        })
    except KeyError as e:
        return jsonify({"success": False, "message": f"Missing configuration: {e}"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


def _reason(name, chosen, basis, visit):
    who = name or f"Phlebotomist {chosen['id']}"
    if basis == "traffic_eta":
        head = (f"{who} is {chosen['etaMinutes']} min away and can reach the patient "
                f"inside the {visit['slot_time']} slot")
    else:
        head = (f"{who} has the lightest round today and covers this area; the "
                f"{visit['slot_time']} slot is far enough out that a live ETA would "
                f"mean nothing yet")

    if chosen['pushes']:
        return f"{head}, but taking it would push {len(chosen['pushes'])} of their own visits."

    return f"{head}, with none of their own visits affected."


if __name__ == '__main__':
    # Standalone mode - app.py mounts this blueprint instead when run together.
    app = Flask(__name__)
    app.register_blueprint(bp)
    port = int(os.environ.get("PORT", 5005))
    app.run(host='0.0.0.0', port=port)
