"""
Dynamic Reallocation via the Hungarian algorithm (scipy.optimize.linear_sum_assignment).

app.py's /auto-assign is greedy: it walks the visit list once, and each visit
picks whichever allowed phlebo looks best *at that moment*. That can lock in a
locally-good pick that blocks a better global arrangement for visits considered
later. This solves for every visit at once, minimizing the *total* distance
across all phlebos - the same class of problem, solved optimally instead of
greedily.

A plain Hungarian match is one-to-one (one visit per phlebo). Since a phlebo can
take several visits in a day, each phlebo is expanded into `capacity` columns
("slots") before solving, so the algorithm can hand it more than one.

This does NOT belong here if hard constraints - required skills, a fixed max
capacity that must never be exceeded, multi-stop time windows - need enforcing.
Those turn this into a constrained VRP, which is what OR-Tools (already used in
this repo's /optimize) is built for; a plain assignment matrix has no way to
express "no more than 3" as a rule rather than a soft preference.

Two endpoints live here. /reallocate is the solver above: hand it phlebos and
visits, get back an optimal arrangement. /reallocation-suggestions is the part
that decides there is a problem worth solving - it reads today's board from the
database itself, finds visits their holder is not going to make, and names a
replacement with the reason and what the swap would cost. See further down.

Run standalone: `python dynamic_reallocation.py` (needs `pip install flask
numpy scipy pymysql requests python-dotenv PyJWT` - not yet in requirements.txt,
since this runs independently of app.py's deployment).
"""

import math
import json
import os
from datetime import datetime, timedelta

import jwt
import numpy as np
import pymysql
import requests
from dotenv import load_dotenv
from scipy.optimize import linear_sum_assignment
from flask import Blueprint, Flask, request, jsonify

load_dotenv()

bp = Blueprint('reallocation', __name__)

# Disallowed visit-phlebo pairs get this cost instead of being removed from the
# matrix outright - linear_sum_assignment needs a fixed-size matrix, so "not
# allowed" has to be expressed as "prohibitively expensive" rather than absent.
DISALLOWED_COST = 1e9


def distance(a, b):
    """Haversine distance in meters - same formula as app.py's, kept local so this file has no import-time dependency on it."""
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
        return list(map(int, value))

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return list(map(int, parsed))
        except (ValueError, TypeError):
            pass

    return []


# ---------------------------------------------------------------------------
# Phlebo-matching rule: radius from home base, or area/city.
#
# country_settings.phlebo_assign_via_radius (per country_id, from the caller's
# JWT) picks which one applies. Radius mode checks the straight-line distance
# from each phlebo's home base to the visit, in the country's own unit
# (country_settings.distance_value: km or mile) - it never looks at live
# location, this is about coverage, not who is closest right now. Area/city
# mode is the original allowed_phlebos_for() match, unchanged.
#
# A phlebo's home base is their own start point (start_lat/start_long) when
# use_address_as_home_base is set, else the phlebo_location_master row their
# homebase_id points at.
# ---------------------------------------------------------------------------

def country_settings(conn, country_id):
    """The per-country phlebo-matching rule. None (missing country_id, no
    matching row, or a DB error) means "no rule configured" - callers treat
    that exactly like phlebo_assign_via_radius being off, i.e. the original
    area/city match, so an unconfigured country never leaves a visit
    unmatched over this."""
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
    row homebase_id points at. None if neither resolves to a real point, so
    the caller can treat them as ineligible rather than guessing."""
    if phlebo.get('use_address_as_home_base'):
        lat, lng = phlebo.get('start_lat'), phlebo.get('start_long')
    else:
        base = None
        if phlebo.get('homebase_id'):
            try:
                base = homebase_locations.get(int(phlebo['homebase_id']))
            except (TypeError, ValueError):
                base = None
        if not base:
            return None
        lat, lng = base.get('latitude'), base.get('longitude')

    if lat in (None, '', 0) or lng in (None, '', 0):
        return None
    try:
        return (float(lat), float(lng))
    except (TypeError, ValueError):
        return None


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


def eligible_by_radius(phlebo, visit, homebase_locations, unit):
    """Visit falls inside this phlebo's serving_radius around their home
    base."""
    home = home_base_point(phlebo, homebase_locations)
    if home is None or not visit.get('lat') or not visit.get('lng'):
        return False
    limit = radius_meters(phlebo.get('serving_radius'), unit)
    if limit is None:
        return False
    return distance(home, (float(visit['lat']), float(visit['lng']))) <= limit


def allowed_phlebos_for(visit, phlebos, settings=None, homebase_locations=None):
    """Same area-then-city match as app.py's /auto-assign, with the same
    all-phlebos fallback when nothing matches - unless settings says this
    country runs on phlebo_assign_via_radius, in which case eligibility is
    the home-base radius check above instead."""
    if settings and settings.get('phlebo_assign_via_radius'):
        allowed = [
            p for p in phlebos
            if eligible_by_radius(p, visit, homebase_locations or {}, settings.get('distance_value'))
        ]
        return allowed if allowed else phlebos

    visit_area = int(visit.get('areacode_id') or 0)
    visit_city = int(visit.get('city_id') or 0)

    allowed = []
    for p in phlebos:
        phlebo_areas = parse_int_list(p.get('serving_areacode_id', []))
        phlebo_cities = parse_int_list(p.get('serving_city_id', []))

        if visit_area != 0 and visit_area in phlebo_areas:
            allowed.append(p)
        elif visit_city in phlebo_cities:
            allowed.append(p)

    return allowed if allowed else phlebos


def nearest_route(start, visits):
    """Nearest-neighbor stop order, used only to report a realistic total_distance per phlebo - the assignment itself doesn't depend on stop order."""
    route = []
    current = (float(start['lat']), float(start['lng']))
    remaining = visits.copy()

    while remaining:
        nxt = min(remaining, key=lambda v: distance(current, (float(v['lat']), float(v['lng']))))
        route.append(nxt)
        current = (float(nxt['lat']), float(nxt['lng']))
        remaining.remove(nxt)

    return route


def solve_reallocation(phlebos, visits, capacity_per_phlebo=None, settings=None, homebase_locations=None):
    valid_phlebos = [
        p for p in phlebos
        if p.get('lat') not in (None, '', 0) and p.get('lng') not in (None, '', 0)
    ]

    if not valid_phlebos:
        raise ValueError("No valid phlebos with location")
    if not visits:
        raise ValueError("No visits provided")

    n_visits = len(visits)
    n_phlebos = len(valid_phlebos)

    # Default capacity covers every visit even in the worst case (all visits
    # landing on one phlebo would still fit) - callers who want a hard per-phlebo
    # cap should pass capacity_per_phlebo explicitly; visits that don't fit within
    # it come back in `unassigned` rather than raising.
    capacity = max(1, capacity_per_phlebo or math.ceil(n_visits / n_phlebos))

    slot_owner = []
    for p_idx in range(n_phlebos):
        slot_owner.extend([p_idx] * capacity)
    n_slots = len(slot_owner)

    cost = np.full((n_visits, n_slots), DISALLOWED_COST)

    for v_idx, v in enumerate(visits):
        allowed_ids = {p['id'] for p in allowed_phlebos_for(v, valid_phlebos, settings, homebase_locations)}
        v_point = (float(v['lat']), float(v['lng']))

        for slot_idx, p_idx in enumerate(slot_owner):
            p = valid_phlebos[p_idx]
            if p['id'] not in allowed_ids:
                continue
            cost[v_idx, slot_idx] = distance((float(p['lat']), float(p['lng'])), v_point)

    row_ind, col_ind = linear_sum_assignment(cost)

    phlebo_visits = {p['id']: [] for p in valid_phlebos}
    visit_results = []
    assigned_visit_indices = set()

    for v_idx, slot_idx in zip(row_ind, col_ind):
        if cost[v_idx, slot_idx] >= DISALLOWED_COST:
            continue

        assigned_visit_indices.add(v_idx)
        v = visits[v_idx]
        p = valid_phlebos[slot_owner[slot_idx]]
        phlebo_visits[p['id']].append(v)

        visit_results.append({
            "visit_id": v['id'],
            "patient_name": v.get('patient_name', ''),
            "address": v.get('address', ''),
            "slot": v.get('slot', ''),
            "order_id": v.get('order_id', ''),
            "assigned_phlebo_id": p['id'],
            "allowed_phlebos": [
                {"id": ap['id'], "name": ap.get('name', '')}
                for ap in allowed_phlebos_for(v, valid_phlebos, settings, homebase_locations)
            ],
        })

    unassigned = [
        visits[i]['id'] for i in range(n_visits) if i not in assigned_visit_indices
    ]

    phlebo_summary = []
    for p in valid_phlebos:
        assigned = phlebo_visits[p['id']]
        sequenced = nearest_route(p, assigned) if assigned else []

        total_distance = 0
        current = (float(p['lat']), float(p['lng']))
        for v in sequenced:
            total_distance += distance(current, (float(v['lat']), float(v['lng'])))
            current = (float(v['lat']), float(v['lng']))

        phlebo_summary.append({
            "id": p['id'],
            "name": p.get('name', ''),
            "total_orders": len(assigned),
            "total_distance": f"{round(total_distance / 1000, 2)} KM",
        })

    return {
        "success": True,
        "phlebos": phlebo_summary,
        "visits": visit_results,
        "unassigned_visit_ids": unassigned,
        "capacity_per_phlebo": capacity,
    }


@bp.route('/reallocate', methods=['POST'])
def reallocate_endpoint():
    try:
        # country_id off the caller's JWT decides the matching rule below -
        # this endpoint took no auth before, but the frontend already sends
        # Authorization on every request (token.interceptor.ts), so this
        # only starts being enforced, not newly required in practice.
        _, country_id, auth_error = customer_id_from_token()
        if auth_error:
            return jsonify({"error": auth_error}), 401

        data = request.json
        phlebos = data['phlebos']
        visits = data['visits']
        capacity = data.get('capacity_per_phlebo')

        settings = None
        homebase_locations = {}
        conn = db_connection()
        try:
            settings = country_settings(conn, country_id)
            if settings and settings.get('phlebo_assign_via_radius'):
                homebase_locations = homebase_locations_for(conn, phlebos)
        finally:
            conn.close()

        result = solve_reallocation(phlebos, visits, capacity, settings, homebase_locations)
        return jsonify(result)
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# /reallocation-suggestions
#
# /reallocate above is a solver: hand it phlebos and visits, get back an optimal
# arrangement. This one is the thing that decides there is a problem worth
# solving at all. Give it only the caller's token and it reads today's board
# itself, finds the visits that are not going to be met by whoever holds them,
# and names a better phlebo for each - with the reason and what it would cost.
#
# It only ever suggests. Nothing is written back, because the patient has
# already been told who is coming; a wrong silent swap is worse than no swap.
# PHP performs the reassignment if a human accepts it.
#
# Only visits where nobody has arrived yet are candidates. Once a phlebo is on
# site, moving the job to someone else cannot make it any faster.
# ---------------------------------------------------------------------------

FIREBASE_LOCATIONS = "https://hsc-phlebotomist-app-default-rtdb.firebaseio.com/phlebo_locations"
MAPS_ENDPOINT = "https://maps.googleapis.com/maps/api/distancematrix/json"

# Statuses meaning the visit is finished with - nothing left to reallocate.
TERMINAL_STATUSES = {
    'sample collected', 'completed', 'enroute to lab', 'enroute to ship',
    'enroute to runner', 'dropped in lab', 'dropped in courier',
    'sample delivered to lab', 'collected', 'collecting', 'cancelled',
}

# Minutes on site to draw the sample - used when judging whether handing a
# candidate this visit would push the visits they already hold.
COLLECTION_MINUTES = 15

# A slot further out than this can still resolve itself; nothing to do yet.
WINDOW_AHEAD_MINUTES = 120

# find_replacement() below only spends a live traffic call (Google Maps) on
# the closest few candidates by straight-line distance, not every phlebo who
# clears the slot/capacity filters - with a large team that's one blocking
# call per candidate per at-risk visit, and this page polls every 30s. Same
# cap and reasoning as ondemand_assignment.py's MAX_ETA_CANDIDATES.
MAX_ETA_CANDIDATES = 5


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
    tymon issues on the PHP side. customer_id is the only source for it: these
    rows carry patient names and phone numbers, so a body parameter would let
    anyone read any customer. country_id decides the phlebo-matching rule in
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


def slot_bounds(slot_date, slot_time):
    """("2026-09-09", "19:30 - 20:00") -> (start, end). A bare start time means a
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
            return start, datetime.strptime(f"{date_str} {parts[1].strip()}", '%Y-%m-%d %H:%M')
        except ValueError:
            pass

    return start, start + timedelta(minutes=30)


def todays_visits(conn, customer_id):
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
                      limit 1) as areacode_id,
                   cities.city_name as city_name,
                   concat(coalesce(phlebos.first_name, ''), ' ',
                          coalesce(phlebos.last_name, '')) as phlebo_name,
                   phlebos.phone as phlebo_phone
            from visits
            left join cities on cities.id = visits.city_id and cities.deleted_at is null
            left join phlebos on phlebos.id = visits.phlebo_id and phlebos.deleted_at is null
            where visits.customer_id = %s and visits.is_active = 1
              and visits.deleted_at is null and visits.slot_date = curdate()
        """, (customer_id,))
        return cur.fetchall()


def phlebo_directory(conn, phlebo_ids):
    """Names and phones for the phlebos being suggested - they aren't holding
    any of today's visits, so they never appear in the visit join above.

    Also carries the columns eligible_by_radius()/allowed_phlebos_for() need
    to judge a candidate: serving_areacode_id/serving_city_id (aliased from
    the actual serving_area_code/serving_city columns, so the same helper
    functions /reallocate uses work unchanged here), use_address_as_home_base,
    start_lat/start_long, homebase_id, serving_radius. These are optional: a
    schema missing any of them just falls back to the original
    "everyone allowed" behavior, same as ondemand_assignment.py's version of
    this query."""
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
    """Where every phlebo actually is, from the path the phlebo app writes to."""
    try:
        resp = requests.get(f"{FIREBASE_LOCATIONS}/{int(customer_id)}.json", timeout=5)
        payload = resp.json() if resp.ok else None
    except (requests.RequestException, ValueError):
        return {}

    if not isinstance(payload, dict):
        return {}

    locations = {}
    for phlebo_id, loc in payload.items():
        if not isinstance(loc, dict):
            continue
        lat = loc.get('latitude', loc.get('lat'))
        lng = loc.get('longitude', loc.get('lng'))
        if lat is None or lng is None:
            continue
        locations[int(phlebo_id)] = {
            "lat": float(lat), "lng": float(lng), "status": loc.get('status'),
        }

    return locations


def traffic_eta(origin, destination):
    """Traffic-aware driving minutes, or None. Never treat a missing answer as
    zero - it means we don't know, not that they are already there."""
    key = os.environ.get('GOOGLE_MAPS_API_KEY')
    if not key:
        return None

    try:
        resp = requests.get(MAPS_ENDPOINT, params={
            "origins": f"{origin[0]},{origin[1]}",
            "destinations": f"{destination[0]},{destination[1]}",
            "mode": "driving", "departure_time": "now",
            "traffic_model": "best_guess", "key": key,
        }, timeout=6)
    except requests.RequestException:
        return None

    if not resp.ok:
        return None

    element = (resp.json().get('rows') or [{}])[0].get('elements', [{}])[0]
    if element.get('status') != 'OK':
        return None

    duration = element.get('duration_in_traffic') or element.get('duration')
    if not duration:
        return None

    return {
        "minutes": math.ceil(duration['value'] / 60),
        "km": round((element.get('distance') or {}).get('value', 0) / 1000, 1),
    }


def diagnose(visit, now, locations):
    """Why this visit is in trouble, or None if it isn't.

    Deliberately blind to the trained delay model: a booking that merely *looks*
    risky by historical pattern is not grounds for taking it off someone. What
    justifies a swap is the present tense - the slot is gone and nobody left, or
    they are moving but cannot arrive in time."""
    status = (visit['status'] or '').strip().lower()
    if status in TERMINAL_STATUSES or status == 'phlebo reached':
        return None

    _, slot_end = slot_bounds(visit['slot_date'], visit['slot_time'])
    if slot_end is None:
        return None

    minutes_left = (slot_end - now).total_seconds() / 60
    if minutes_left > WINDOW_AHEAD_MINUTES:
        return None

    phlebo_id = int(visit['phlebo_id']) if visit['phlebo_id'] else 0
    if not phlebo_id:
        return {"problem": "unassigned", "minutesLeft": minutes_left,
                "detail": "No phlebotomist assigned"}

    if minutes_left < 0 and status == 'phlebo assigned':
        return {"problem": "never_departed", "minutesLeft": minutes_left,
                "detail": f"still not departed, {round(-minutes_left)} min past the slot"}

    origin = locations.get(phlebo_id)
    if not origin or not visit['lat'] or not visit['lng']:
        # Nothing to measure against - an unreachable phlebo past their slot is
        # still a problem, one with time left simply can't be judged yet.
        if minutes_left < 0:
            return {"problem": "no_tracking", "minutesLeft": minutes_left,
                    "detail": f"no live location, {round(-minutes_left)} min past the slot"}
        return None

    eta = traffic_eta((origin['lat'], origin['lng']), (visit['lat'], visit['lng']))
    if eta is None:
        return None

    if eta['minutes'] > minutes_left:
        return {"problem": "cannot_arrive_in_time", "minutesLeft": minutes_left,
                "currentEtaMinutes": eta['minutes'],
                "detail": f"is {eta['minutes']} min away with {round(minutes_left)} min left"}

    return None


def eligible_for_visit(phlebo, visit, settings, homebase_locations):
    """Whether this phlebo may be suggested for this visit at all - the same
    radius-or-area/city rule allowed_phlebos_for() applies for /reallocate.
    An empty/missing directory row (schema without the columns, or a phlebo
    find_replacement() has no directory data for) reads as False here; the
    caller falls back to "everyone eligible" the same way allowed_phlebos_for
    does when its list comes back empty, rather than doing that per-call."""
    if settings and settings.get('phlebo_assign_via_radius'):
        return eligible_by_radius(phlebo, visit, homebase_locations, settings.get('distance_value'))

    visit_area = int(visit.get('areacode_id') or 0)
    visit_city = int(visit.get('city_id') or 0)
    phlebo_areas = parse_int_list(phlebo.get('serving_areacode_id', []))
    phlebo_cities = parse_int_list(phlebo.get('serving_city_id', []))

    if visit_area != 0 and visit_area in phlebo_areas:
        return True
    return visit_city in phlebo_cities


def find_replacement(visit, now, locations, holdings, capacity, exclude_id,
                      directory=None, settings=None, homebase_locations=None):
    """The closest phlebo who is free in this slot and has room, with what
    taking it would cost the visits they already hold.

    Eligibility (radius-from-home-base or area/city, per country_settings) is
    preferred but not absolute: if nobody eligible qualifies, the closest
    otherwise-workable candidate is used instead - same "don't leave a visit
    unstaffed over a missing configuration row" fallback allowed_phlebos_for()
    applies for /reallocate.

    Only the MAX_ETA_CANDIDATES closest survivors (straight-line) get a live
    traffic call - with a large team this is called once per at-risk visit,
    and a Maps round-trip per phlebo per visit does not scale."""
    directory = directory or {}
    slot_start, slot_end = slot_bounds(visit['slot_date'], visit['slot_time'])
    # DB rows carry lat/lng as strings - distance() (math.radians) needs floats,
    # unlike traffic_eta() below which just interpolates them into a URL.
    visit_point = (float(visit['lat']), float(visit['lng']))
    best = None
    best_any = None

    survivors = []
    for phlebo_id, loc in locations.items():
        if phlebo_id == exclude_id or loc.get('status') == 'clock_out':
            continue

        held = holdings.get(phlebo_id, [])
        # Free in this slot: the whole point - nobody can be in two places at once.
        if any(h['slot_time'] == visit['slot_time'] for h in held):
            continue
        if len(held) >= capacity:
            continue

        survivors.append((phlebo_id, loc, held))

    survivors.sort(key=lambda s: distance((s[1]['lat'], s[1]['lng']), visit_point))

    for phlebo_id, loc, held in survivors[:MAX_ETA_CANDIDATES]:
        eta = traffic_eta((loc['lat'], loc['lng']), (visit['lat'], visit['lng']))
        if eta is None:
            continue

        # Impact: which of their own visits would this push? Anything whose slot
        # starts before they could get there, collect, and move on.
        free_again = now + timedelta(minutes=eta['minutes'] + COLLECTION_MINUTES)
        affected = []
        for h in held:
            h_start, _ = slot_bounds(h['slot_date'], h['slot_time'])
            if h_start and h_start < free_again:
                affected.append({"visitId": h['id'], "slotTime": h['slot_time']})

        candidate = {
            "id": phlebo_id, "etaMinutes": eta['minutes'], "distanceKm": eta['km'],
            "currentLoad": len(held), "affected": affected,
        }

        # Fewest people hurt first, then soonest to arrive.
        rank = (len(affected), eta['minutes'])
        if best_any is None or rank < (len(best_any['affected']), best_any['etaMinutes']):
            best_any = candidate

        if eligible_for_visit(directory.get(phlebo_id, {}), visit, settings, homebase_locations):
            if best is None or rank < (len(best['affected']), best['etaMinutes']):
                best = candidate

    return best if best is not None else best_any


def suggestion_reason(visit, issue, replacement, replacement_name):
    who = (visit['phlebo_name'] or '').strip()
    if issue['problem'] == 'unassigned':
        head = f"{visit['slot_time']} has no phlebotomist"
    else:
        head = f"{who or 'The assigned phlebotomist'} {issue['detail']}"

    tail = (f"{replacement_name} is {replacement['etaMinutes']} min away "
            f"and free in this slot")
    if replacement['affected']:
        tail += f", but taking it would push {len(replacement['affected'])} of their own visits"

    return f"{head}. {tail}."


@bp.route('/reallocation-suggestions', methods=['POST'])
def reallocation_suggestions_endpoint():
    try:
        customer_id, country_id, auth_error = customer_id_from_token()
        if auth_error:
            return jsonify({"success": False, "message": auth_error}), 401

        body = request.json or {}
        limit = min(int(body.get('limit', 25)), 50)
        now = datetime.now()

        conn = db_connection()
        try:
            visits = todays_visits(conn, customer_id)

            # What each phlebo is already holding today - this is both the slot
            # occupancy check and the workload figure.
            holdings = {}
            live_visits = []
            for v in visits:
                if (v['status'] or '').strip().lower() in TERMINAL_STATUSES:
                    continue
                live_visits.append(v)
                if v['phlebo_id']:
                    holdings.setdefault(int(v['phlebo_id']), []).append(v)

            locations = live_locations(customer_id)

            # Directory (names + the radius-or-area/city eligibility columns)
            # for every phlebo who could be suggested - everyone with a live
            # location. Doubles as the name/phone lookup below, since a
            # replacement can only ever be one of these.
            directory = phlebo_directory(conn, locations.keys())

            # country_settings decides which eligibility rule find_replacement
            # applies; home base locations are only fetched when radius mode
            # actually needs them.
            settings = country_settings(conn, country_id)
            homebase_locations = (
                homebase_locations_for(conn, directory.values())
                if settings and settings.get('phlebo_assign_via_radius') else {}
            )

            # Capacity straight from the board: how many visits there are to go
            # round how many phlebos are actually out there.
            capacity = max(1, math.ceil(len(live_visits) / len(locations))) if locations else 1

            suggestions = []
            for visit in live_visits:
                issue = diagnose(visit, now, locations)
                if not issue:
                    continue

                current_id = int(visit['phlebo_id']) if visit['phlebo_id'] else None
                replacement = find_replacement(visit, now, locations, holdings, capacity,
                                               current_id, directory, settings, homebase_locations)
                if not replacement:
                    continue
                suggestions.append((visit, issue, replacement))

            names = directory
        finally:
            conn.close()

        out = []
        for visit, issue, replacement in suggestions:
            row = names.get(replacement['id'], {})
            new_name = (row.get('name') or '').strip()
            city = (visit['city_name'] or '').strip()
            current_name = (visit['phlebo_name'] or '').strip()

            out.append({
                "visitId": int(visit['id']),
                "orderId": visit['order_id'],
                "bookingId": visit['task_id'] or visit['order_number'],
                "patient": (visit['main_patient_name'] or '').strip() or None,
                "city": city or None,
                "slotDate": str(visit['slot_date']),
                "slotTime": visit['slot_time'],
                "status": visit['status'],
                "problem": issue['problem'],
                "minutesLate": round(-issue['minutesLeft']) if issue['minutesLeft'] < 0 else 0,
                "currentPhlebo": None if not visit['phlebo_id'] else {
                    "id": int(visit['phlebo_id']),
                    "name": current_name or None,
                    "phone": visit['phlebo_phone'],
                    "etaMinutes": issue.get('currentEtaMinutes'),
                },
                "suggestedPhlebo": {
                    "id": replacement['id'],
                    "name": new_name or None,
                    "phone": row.get('phone'),
                    "etaMinutes": replacement['etaMinutes'],
                    "distanceKm": replacement['distanceKm'],
                    "currentLoad": replacement['currentLoad'],
                    "freeInSlot": True,
                },
                "impact": {
                    "delaysCaused": len(replacement['affected']),
                    "phleboVisitsAffected": replacement['affected'],
                },
                "reason": suggestion_reason(visit, issue, replacement, new_name or 'The suggested phlebotomist'),
            })

        # Cleanest swaps first: nobody else hurt, and the most overdue up top.
        out.sort(key=lambda r: (r['impact']['delaysCaused'], -r['minutesLate']))
        out = out[:limit]

        return jsonify({
            "success": True,
            "data": {
                "date": now.strftime('%Y-%m-%d'),
                "generatedAt": now.strftime('%Y-%m-%d %H:%M:%S'),
                "capacityPerPhlebo": capacity,
                "counts": {
                    "suggested": len(out),
                    "zeroImpact": sum(1 for r in out if r['impact']['delaysCaused'] == 0),
                },
                "suggestions": out,
            },
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == '__main__':
    # Standalone mode - app.py mounts this blueprint instead when run together.
    app = Flask(__name__)
    app.register_blueprint(bp)
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)
