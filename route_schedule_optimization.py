"""
Route & Schedule Optimization - multi-stop schedules that respect the slot.

What this adds over what already exists
---------------------------------------
app.py's /optimize is a single-vehicle TSP: one phlebo, a list of points, the
shortest tour through them. It has no idea that a stop is promised for
11:00-12:00, and it cannot split work across people.

dynamic_reallocation.py's /reallocate is the mirror image: it decides *who*
takes each visit and nothing about *when* or *in what order*. Its own docstring
says the missing piece - hard capacity and time windows - "turn this into a
constrained VRP, which is what OR-Tools is built for". This file is that VRP.

So the three are complementary, and none of them replaces another:

    /optimize                  one phlebo, order only, no clock
    /reallocate                who takes what, no order, no clock
    /optimize-schedule (here)  who takes what, in what order, against the clock

Nothing in app.py or dynamic_reallocation.py is modified or imported - the
shared helpers below are deliberately re-declared, so this runs and deploys on
its own the same way the other services do.

The model
---------
A vehicle routing problem with time windows (VRPTW), solved with OR-Tools:

  travel time   the arc cost. Straight-line distance inflated by a road factor
                and divided by a city driving speed - or, on request, a real
                traffic-aware matrix from Google.
  SLA           each visit carries its slot as a time window. The window's
                *end* is the deadline, because a visit is not late until the
                promised window has elapsed - arriving at 11:30 into an
                11:00-12:00 slot is on time. Missing it is penalised, not
                forbidden: a real board always contains slots that have already
                passed, and a hard constraint would make the whole day
                unsolvable instead of returning the least-bad plan.
  workload      a per-phlebo cap on stops (hard) plus a price on the busiest
                round, which pulls work off whoever is carrying most (soft).
                Measured in stops rather than hours on purpose: charging for
                the length of the working day rewards setting off late, since
                a round that starts later has a shorter span.

A visit that cannot be served by anyone - out of area, no one with room, no way
to reach it in the day - is dropped into `unassigned` with a reason rather than
making the model infeasible.

Two endpoints
-------------
/optimize-schedule is the solver. Hand it phlebos and visits, get back an
ordered plan with projected arrival times. No database, no token - the same
stateless shape as /optimize and /reallocate.

/schedule-plan is the "as conditions change" half. Give it only the caller's
token and it reads today's board itself, takes live phlebo positions from
Firebase, freezes the work that is already underway, re-optimises everything
still open, and reports the new plan against what the board would do if left
alone. It never writes: the patient has already been told who is coming, so a
silent re-shuffle is worse than none. PHP applies a plan a human accepted.

Data access
-----------
/schedule-plan reads visits/phlebos/visit_status read-only via PyMySQL, using
DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD from .env - never hardcode
credentials here. Which customer's rows come back is decided by the customer_id
claim inside the caller's JWT and nothing else (see customer_id_from_token):
the rows carry patient names and phone numbers, so a body parameter would let
anyone read any customer's board.

GOOGLE_MAPS_API_KEY is optional. Without it - or for a board too large to price
a full matrix - travel times fall back to the geometric estimate, and the
response says which model was used in `travelModel`.

Usage
-----
    python route_schedule_optimization.py

Needs `pip install flask ortools pymysql requests python-dotenv PyJWT` (ortools
and flask are already in requirements.txt; the rest arrive with the other
services in this repo).
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
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

load_dotenv()

bp = Blueprint('schedule', __name__)

# --- travel model ----------------------------------------------------------

# Straight-line distance understates a drive. Roads bend, rivers have bridges
# in only some places. 1.3 is the usual planning multiplier for dense Indian
# cities; it matters because every SLA decision below is made on these numbers.
ROAD_FACTOR = 1.3
DEFAULT_SPEED_KMPH = 20

# A full n x n traffic matrix costs n^2 billable elements. Past this the price
# stops being worth the accuracy and the geometric estimate is used instead.
MAX_TRAFFIC_ELEMENTS = 400
MAPS_ENDPOINT = "https://maps.googleapis.com/maps/api/distancematrix/json"
MAPS_TIMEOUT_SECONDS = 10
# Google caps a single Distance Matrix request at 25 origins, 25 destinations
# and 100 elements, so the matrix is fetched in 10x10 blocks.
MAPS_BLOCK = 10

FIREBASE_LOCATIONS = "https://hsc-phlebotomist-app-default-rtdb.firebaseio.com/phlebo_locations"
FIREBASE_TIMEOUT_SECONDS = 5

# --- scheduling ------------------------------------------------------------

DEFAULT_COLLECTION_MINUTES = 15

# Seconds of extra driving the solver will accept to save one second of
# lateness. Above 1 the plan protects the promised slot at the cost of fuel,
# which is the trade a diagnostics business actually wants.
DEFAULT_LATENESS_PENALTY = 3

# Leaving a visit unserved must always cost more than any detour that could
# serve it - otherwise the solver quietly drops the awkward ones.
DEFAULT_DROP_PENALTY = 10_000_000

# Pressure to even out stop counts, priced in seconds of driving: at 600 the
# solver will accept ten extra minutes on the road to take one stop off
# whoever is carrying the most. Balance is deliberately measured in stops and
# not in hours worked - an hours-based span cost on a dimension that contains
# waiting rewards setting off late, which is the opposite of the point.
DEFAULT_BALANCE_COEFFICIENT = 600

# Headroom over the tight ceil(visits / phlebos) share. A cap set exactly at
# the average leaves no room for area constraints and forces avoidable drops.
CAPACITY_SLACK = 2

# Guided local search never converges early - it improves until the clock runs
# out - so the budget is scaled to the board instead of being a flat number a
# three-visit morning would sit through for no gain.
MIN_TIME_LIMIT_SECONDS = 2
MAX_TIME_LIMIT_SECONDS = 20
VISITS_PER_SECOND_OF_SEARCH = 5

# One day ahead of "now" bounds every cumulative time variable in the model.
HORIZON_SECONDS = 24 * 3600

# Statuses where the visit is finished with - nothing left to schedule. Same
# set the other services in this repo treat as terminal.
TERMINAL_STATUSES = {
    'sample collected', 'completed', 'enroute to lab', 'enroute to ship',
    'enroute to runner', 'dropped in lab', 'dropped in courier',
    'sample delivered to lab', 'collected', 'collecting', 'cancelled',
}


def haversine_meters(a, b):
    """Great-circle distance in metres. Integer, because OR-Tools costs are."""
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
    """(lat, lng) as floats, or None when either is missing or unusable.

    0/0 is rejected as well as null: it is the Gulf of Guinea, and in this data
    it always means "never captured", never a real location."""
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
    """("2026-09-11", "11:00 - 12:00") -> (start, end) datetimes.

    A bare start time means a 30-minute window, the same rule the rest of the
    platform uses. Returns (None, None) for anything unparseable - the caller
    treats that as "no promised window" rather than dropping the visit."""
    if not slot_date or not slot_time:
        return None, None

    date_str = slot_date.strftime('%Y-%m-%d') if hasattr(slot_date, 'strftime') else str(slot_date)[:10]
    parts = str(slot_time).split('-')

    # Slots are stored both as "16:00 - 17:00" and as "4:00 PM - 5:00 PM".
    def at(clock):
        clock = ' '.join(clock.strip().upper().split())
        for fmt in ('%H:%M', '%I:%M %p', '%I:%M%p', '%H:%M:%S'):
            try:
                return datetime.strptime(f"{date_str} {clock}", f'%Y-%m-%d {fmt}')
            except ValueError:
                continue
        return None

    start = at(parts[0])
    if start is None:
        return None, None

    if len(parts) == 2:
        end = at(parts[1])
        if end is not None and end > start:
            return start, end

    return start, start + timedelta(minutes=30)


def parse_datetime(value):
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M'):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            continue
    return None


def geometric_matrices(points, speed_kmph):
    """(metres, seconds) matrices from straight-line distance.

    `points` may contain None for the synthetic route-end nodes; every arc
    touching one costs nothing, which is what makes routes open-ended."""
    size = len(points)
    metres = [[0] * size for _ in range(size)]
    seconds = [[0] * size for _ in range(size)]
    mps = max(speed_kmph, 1) * 1000 / 3600

    for i in range(size):
        if points[i] is None:
            continue
        for j in range(size):
            if i == j or points[j] is None:
                continue
            road = int(haversine_meters(points[i], points[j]) * ROAD_FACTOR)
            metres[i][j] = road
            seconds[i][j] = int(road / mps)

    return metres, seconds


def google_matrices(points):
    """Traffic-aware (metres, seconds) from Google, or None to fall back.

    None means "could not price this" - no key, too many elements, or the API
    refused. It never means "everything is zero distance apart"; a silent zero
    would read as every patient living at the depot."""
    key = os.environ.get('GOOGLE_MAPS_API_KEY')
    if not key:
        return None

    real = [i for i, p in enumerate(points) if p is not None]
    if not real or len(real) ** 2 > MAX_TRAFFIC_ELEMENTS:
        return None

    size = len(points)
    metres = [[0] * size for _ in range(size)]
    seconds = [[0] * size for _ in range(size)]

    for o_start in range(0, len(real), MAPS_BLOCK):
        origins = real[o_start:o_start + MAPS_BLOCK]
        for d_start in range(0, len(real), MAPS_BLOCK):
            destinations = real[d_start:d_start + MAPS_BLOCK]

            try:
                resp = requests.get(MAPS_ENDPOINT, params={
                    "origins": "|".join(f"{points[i][0]},{points[i][1]}" for i in origins),
                    "destinations": "|".join(f"{points[j][0]},{points[j][1]}" for j in destinations),
                    "mode": "driving",
                    # Without departure_time Google answers free-flow and
                    # ignores congestion entirely.
                    "departure_time": "now",
                    "traffic_model": "best_guess",
                    "key": key,
                }, timeout=MAPS_TIMEOUT_SECONDS)
            except requests.RequestException:
                return None

            if not resp.ok:
                return None

            try:
                rows = resp.json().get('rows') or []
            except ValueError:
                return None

            if len(rows) != len(origins):
                return None

            for oi, row in zip(origins, rows):
                elements = row.get('elements') or []
                if len(elements) != len(destinations):
                    return None
                for dj, element in zip(destinations, elements):
                    if oi == dj:
                        continue
                    if element.get('status') != 'OK':
                        # One unreachable pair is not a reason to throw the
                        # whole matrix away - estimate just that arc.
                        metres[oi][dj], seconds[oi][dj] = _geometric_arc(points[oi], points[dj])
                        continue
                    duration = element.get('duration_in_traffic') or element.get('duration')
                    if not duration:
                        metres[oi][dj], seconds[oi][dj] = _geometric_arc(points[oi], points[dj])
                        continue
                    metres[oi][dj] = int((element.get('distance') or {}).get('value', 0))
                    seconds[oi][dj] = int(duration['value'])

    return metres, seconds


def _geometric_arc(a, b, speed_kmph=DEFAULT_SPEED_KMPH):
    road = int(haversine_meters(a, b) * ROAD_FACTOR)
    return road, int(road / (max(speed_kmph, 1) * 1000 / 3600))


def normalize_phlebo(raw):
    """Solver-shaped phlebo, or None when there is nowhere to start them from.

    `available_from` lets a phlebo who is mid-collection join the plan at the
    moment they actually come free rather than immediately."""
    point = coerce_point(raw.get('lat'), raw.get('lng'))
    if point is None:
        return None

    return {
        "id": raw['id'],
        "name": (raw.get('name') or '').strip(),
        "point": point,
        "serving_areacode_id": parse_int_list(raw.get('serving_areacode_id', [])),
        "serving_city_id": parse_int_list(raw.get('serving_city_id', [])),
        "available_from": parse_datetime(raw.get('available_from')),
        "capacity": raw.get('capacity'),
    }


def normalize_visit(raw, default_service_minutes):
    """Solver-shaped visit, or (None, reason) when it cannot be routed at all.

    The window may come as an explicit slot_start/slot_end pair or as the
    platform's "11:00 - 12:00" slot string; both end up as datetimes, and a
    visit with neither is scheduled with no deadline rather than discarded."""
    point = coerce_point(raw.get('lat'), raw.get('lng'))
    if point is None:
        return None, "no usable latitude/longitude"

    slot_start = parse_datetime(raw.get('slot_start'))
    slot_end = parse_datetime(raw.get('slot_end'))
    if slot_start is None or slot_end is None:
        slot_start, slot_end = slot_bounds(
            raw.get('slot_date'), raw.get('slot_time') or raw.get('slot'))

    service = raw.get('service_minutes')
    try:
        service = int(service) if service is not None else default_service_minutes
    except (TypeError, ValueError):
        service = default_service_minutes

    pinned = raw.get('pinned_phlebo_id')
    # Optional explicit eligibility from the caller (e.g. /batch-plan's radius
    # rule). Absent means the area/city match below decides, as before.
    explicit = parse_int_list(raw.get('allowed_phlebo_ids')) if raw.get('allowed_phlebo_ids') is not None else None

    return {
        "id": raw['id'],
        "point": point,
        "slot_start": slot_start,
        "slot_end": slot_end,
        "slot_time": raw.get('slot_time') or raw.get('slot') or '',
        "areacode_id": int(raw.get('areacode_id') or 0),
        "city_id": int(raw.get('city_id') or 0),
        "service_seconds": max(0, service) * 60,
        "pinned_phlebo_id": int(pinned) if pinned else None,
        "allowed_ids": explicit,
        "raw": raw,
    }, None


def allowed_phlebo_ids(visit, phlebos):
    """Area first, then city - the same precedence app.py's /auto-assign uses,
    including its fall back to everybody when nothing matches.

    Unlike /auto-assign this really is a constraint in the solved model, not a
    preference applied while scanning a list, so the fallback matters more: an
    empty allow-list here would drop the visit outright."""
    if visit['pinned_phlebo_id']:
        pinned = [p['id'] for p in phlebos if p['id'] == visit['pinned_phlebo_id']]
        if pinned:
            return pinned

    if visit.get('allowed_ids'):
        explicit = [p['id'] for p in phlebos if int(p['id']) in visit['allowed_ids']]
        if explicit:
            return explicit

    allowed = []
    for p in phlebos:
        if visit['areacode_id'] and visit['areacode_id'] in p['serving_areacode_id']:
            allowed.append(p['id'])
        elif visit['city_id'] and visit['city_id'] in p['serving_city_id']:
            allowed.append(p['id'])

    return allowed if allowed else [p['id'] for p in phlebos]


def solve_schedule(raw_phlebos, raw_visits, now=None, options=None):
    """Plan every open visit across every available phlebo.

    Raises ValueError for input that has no plan in it at all (nobody to send,
    nothing to send them to). Anything narrower - one unroutable visit, one
    phlebo without a location - is reported in the result, not raised."""
    options = options or {}
    now = now or datetime.now()

    speed = float(options.get('speed_kmph') or DEFAULT_SPEED_KMPH)
    service_minutes = int(options.get('collection_minutes') or DEFAULT_COLLECTION_MINUTES)
    lateness_penalty = int(options.get('lateness_penalty') or DEFAULT_LATENESS_PENALTY)
    drop_penalty = int(options.get('drop_penalty') or DEFAULT_DROP_PENALTY)
    balance = int(options.get('balance_coefficient', DEFAULT_BALANCE_COEFFICIENT))
    use_traffic = bool(options.get('use_live_traffic'))

    phlebos = []
    skipped_phlebos = []
    for raw in raw_phlebos or []:
        p = normalize_phlebo(raw)
        if p is None:
            skipped_phlebos.append({"id": raw.get('id'), "reason": "no usable latitude/longitude"})
        else:
            phlebos.append(p)

    if not phlebos:
        raise ValueError("No phlebotomists with a usable location")

    visits = []
    unassigned = []
    for raw in raw_visits or []:
        v, reason = normalize_visit(raw, service_minutes)
        if v is None:
            unassigned.append({"visitId": raw.get('id'), "reason": reason})
        else:
            visits.append(v)

    if not raw_visits:
        raise ValueError("No visits provided")

    if not visits:
        return _empty_plan(phlebos, unassigned, skipped_phlebos, now, "geometric")

    n_vehicles = len(phlebos)
    n_visits = len(visits)

    # Node layout: one start per phlebo, then the visits, then one synthetic
    # end per phlebo. Distinct start and end nodes per vehicle is the safe
    # OR-Tools shape, and zero-cost ends leave the routes open - a phlebo
    # stops when the work stops rather than driving back to where they woke up.
    starts = list(range(n_vehicles))
    visit_nodes = list(range(n_vehicles, n_vehicles + n_visits))
    visit_node_set = set(visit_nodes)
    ends = list(range(n_vehicles + n_visits, n_vehicles + n_visits + n_vehicles))

    points = [p['point'] for p in phlebos] + [v['point'] for v in visits] + [None] * n_vehicles

    travel_model = "geometric"
    matrices = google_matrices(points) if use_traffic else None
    if matrices is not None:
        travel_model = "google_traffic"
    else:
        matrices = geometric_matrices(points, speed)
    metres, seconds = matrices

    manager = pywrapcp.RoutingIndexManager(len(points), n_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    def travel_time(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        # Service is charged on leaving a visit, so the arrival time read off
        # the dimension is a true arrival and not arrival-plus-collection.
        service = visits[i - n_vehicles]['service_seconds'] if i in visit_node_set else 0
        return seconds[i][j] + service

    transit_index = routing.RegisterTransitCallback(travel_time)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    routing.AddDimension(transit_index, HORIZON_SECONDS, HORIZON_SECONDS, False, 'Time')
    time_dim = routing.GetDimensionOrDie('Time')

    def demand(from_index):
        return 1 if manager.IndexToNode(from_index) in visit_node_set else 0

    demand_index = routing.RegisterUnaryTransitCallback(demand)
    default_capacity = max(1, math.ceil(n_visits / n_vehicles) + CAPACITY_SLACK)
    requested_capacity = options.get('capacity_per_phlebo')
    capacities = []
    for p in phlebos:
        cap = p['capacity'] or requested_capacity or default_capacity
        capacities.append(max(1, int(cap)))

    routing.AddDimensionWithVehicleCapacity(demand_index, 0, capacities, True, 'Load')
    if balance > 0:
        # Every route's load starts at zero, so the span of this dimension is
        # simply the busiest phlebo's stop count. Pricing that pulls work off
        # whoever is most loaded without touching when anyone sets off.
        routing.GetDimensionOrDie('Load').SetGlobalSpanCostCoefficient(balance)

    for v_idx, phlebo in enumerate(phlebos):
        available = 0
        if phlebo['available_from']:
            available = max(0, int((phlebo['available_from'] - now).total_seconds()))
        time_dim.CumulVar(routing.Start(v_idx)).SetRange(available, HORIZON_SECONDS)
        # Without this the solver is free to idle at the depot for free; pulling
        # both ends of every route earlier is what stops it inventing plans that
        # technically make the slot but leave people parked for an hour.
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v_idx)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v_idx)))

    id_to_vehicle = {p['id']: i for i, p in enumerate(phlebos)}

    for offset, visit in enumerate(visits):
        node = visit_nodes[offset]
        index = manager.NodeToIndex(node)
        cumul = time_dim.CumulVar(index)

        if visit['slot_start'] and visit['slot_end']:
            open_at = max(0, int((visit['slot_start'] - now).total_seconds()))
            deadline = int((visit['slot_end'] - now).total_seconds())

            # A slot whose start has passed opens immediately; one whose end has
            # passed gets a deadline of zero, so every further second counts as
            # lateness and the solver races to it instead of treating a dead
            # deadline as one it can no longer miss.
            cumul.SetRange(min(open_at, HORIZON_SECONDS), HORIZON_SECONDS)
            time_dim.SetCumulVarSoftUpperBound(
                index, max(0, min(deadline, HORIZON_SECONDS)), lateness_penalty)
        else:
            cumul.SetRange(0, HORIZON_SECONDS)

        allowed = allowed_phlebo_ids(visit, phlebos)
        vehicles = [id_to_vehicle[pid] for pid in allowed if pid in id_to_vehicle]
        if vehicles and len(vehicles) < n_vehicles:
            # Restricting the node's VehicleVar rather than calling
            # SetAllowedVehiclesForIndex: that method's Python binding rejects
            # every sequence type in ortools 9.15. -1 is the var's "unperformed"
            # value and has to stay in the domain, or the disjunction below
            # could never drop the node and the model turns infeasible again.
            routing.VehicleVar(index).SetValues([-1] + vehicles)

        # Every visit is droppable. With hard area rules and hard capacity an
        # unservable visit would otherwise make the entire board infeasible and
        # return no plan at all, which is the one outcome a dispatcher cannot use.
        routing.AddDisjunction([index], drop_penalty)

    time_limit = options.get('time_limit_seconds')
    time_limit = int(time_limit) if time_limit else max(
        MIN_TIME_LIMIT_SECONDS,
        min(MAX_TIME_LIMIT_SECONDS, math.ceil(n_visits / VISITS_PER_SECOND_OF_SEARCH)))
    time_limit = max(1, min(time_limit, MAX_TIME_LIMIT_SECONDS))

    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search.time_limit.FromSeconds(time_limit)

    solution = routing.SolveWithParameters(search)
    if solution is None:
        raise ValueError("No feasible schedule found for this board")

    routes = []
    planned_visit_ids = set()
    total_metres = 0
    total_late = 0

    for v_idx, phlebo in enumerate(phlebos):
        index = routing.Start(v_idx)
        node = manager.IndexToNode(index)
        stops = []
        route_metres = 0
        depart = solution.Min(time_dim.CumulVar(index))

        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)

            if next_node in visit_node_set:
                visit = visits[next_node - n_vehicles]
                arrival_seconds = solution.Min(time_dim.CumulVar(next_index))
                arrival = now + timedelta(seconds=arrival_seconds)
                leg = metres[node][next_node]
                route_metres += leg

                late_minutes = 0
                if visit['slot_end']:
                    late_minutes = max(0, round((arrival - visit['slot_end']).total_seconds() / 60))

                planned_visit_ids.add(visit['id'])
                if late_minutes:
                    total_late += 1

                stops.append({
                    "sequence": len(stops) + 1,
                    "visitId": visit['id'],
                    "slotTime": visit['slot_time'] or None,
                    "slotStart": visit['slot_start'].strftime('%H:%M') if visit['slot_start'] else None,
                    "slotEnd": visit['slot_end'].strftime('%H:%M') if visit['slot_end'] else None,
                    "projectedArrival": arrival.strftime('%Y-%m-%d %H:%M'),
                    "projectedDeparture": (arrival + timedelta(seconds=visit['service_seconds'])).strftime('%Y-%m-%d %H:%M'),
                    "legKm": round(leg / 1000, 2),
                    "withinSlot": late_minutes == 0,
                    "minutesLate": late_minutes,
                })

            index = next_index
            node = next_node

        finish = solution.Min(time_dim.CumulVar(index))
        total_metres += route_metres

        routes.append({
            "phleboId": phlebo['id'],
            "phleboName": phlebo['name'] or None,
            "stops": stops,
            "totalStops": len(stops),
            "totalKm": round(route_metres / 1000, 2),
            "startsAt": (now + timedelta(seconds=depart)).strftime('%Y-%m-%d %H:%M') if stops else None,
            "endsAt": (now + timedelta(seconds=finish)).strftime('%Y-%m-%d %H:%M') if stops else None,
            "workingMinutes": round((finish - depart) / 60) if stops else 0,
            "capacity": capacities[v_idx],
            "lateStops": sum(1 for s in stops if not s['withinSlot']),
        })

    for visit in visits:
        if visit['id'] in planned_visit_ids:
            continue
        unassigned.append({
            "visitId": visit['id'],
            "slotTime": visit['slot_time'] or None,
            "reason": _drop_reason(visit, phlebos, capacities),
        })

    loads = [r['totalStops'] for r in routes]
    working = [r['workingMinutes'] for r in routes if r['totalStops']]

    return {
        "success": True,
        "generatedAt": now.strftime('%Y-%m-%d %H:%M:%S'),
        "travelModel": travel_model,
        "routes": routes,
        "unassigned": unassigned,
        "skippedPhlebos": skipped_phlebos,
        "summary": {
            "phlebosUsed": sum(1 for r in routes if r['totalStops']),
            "phlebosAvailable": len(routes),
            "visitsPlanned": len(planned_visit_ids),
            "visitsUnassigned": len(unassigned),
            "totalKm": round(total_metres / 1000, 2),
            "stopsOutsideSlot": total_late,
            "busiestPhleboStops": max(loads) if loads else 0,
            "quietestPhleboStops": min(loads) if loads else 0,
            "longestDayMinutes": max(working) if working else 0,
        },
        "settings": {
            "speedKmph": speed,
            "collectionMinutes": service_minutes,
            "latenessPenalty": lateness_penalty,
            "balanceCoefficient": balance,
            "capacityPerPhlebo": capacities[0] if len(set(capacities)) == 1 else capacities,
            "timeLimitSeconds": time_limit,
        },
    }


def _empty_plan(phlebos, unassigned, skipped, now, travel_model):
    """Every visit was unroutable. Still a successful answer - the caller asked
    what the day looks like, and the honest reply is "nothing can be planned"
    with a reason per visit, not an error."""
    return {
        "success": True,
        "generatedAt": now.strftime('%Y-%m-%d %H:%M:%S'),
        "travelModel": travel_model,
        "routes": [{
            "phleboId": p['id'], "phleboName": p['name'] or None, "stops": [],
            "totalStops": 0, "totalKm": 0, "startsAt": None, "endsAt": None,
            "workingMinutes": 0, "capacity": 0, "lateStops": 0,
        } for p in phlebos],
        "unassigned": unassigned,
        "skippedPhlebos": skipped,
        "summary": {
            "phlebosUsed": 0, "phlebosAvailable": len(phlebos), "visitsPlanned": 0,
            "visitsUnassigned": len(unassigned), "totalKm": 0, "stopsOutsideSlot": 0,
            "busiestPhleboStops": 0, "quietestPhleboStops": 0, "longestDayMinutes": 0,
        },
        "settings": {},
    }


def _drop_reason(visit, phlebos, capacities):
    """Why the solver left this one out. Distinguishing "nobody covers this
    area" from "everyone was full" is the difference between a hiring problem
    and a rostering problem, so it is worth naming."""
    allowed = allowed_phlebo_ids(visit, phlebos)
    if len(allowed) < len(phlebos):
        return f"only {len(allowed)} phlebotomist(s) cover this area, and none had room in time"
    if sum(capacities) < 1:
        return "no capacity left on any route"
    return "no route could reach it without pushing other visits further past their slots"


def simulate_current_plan(assignments, now, speed_kmph, service_minutes):
    """What the board does if nobody intervenes.

    Each phlebo works their own visits in slot order - what a human dispatcher
    does by hand - measured with the same travel model as the optimised plan so
    the two numbers are comparable. Anything nobody holds counts as unserved,
    because on an untouched board it is exactly that."""
    total_metres = 0
    late_stops = 0
    planned = 0

    for phlebo, visits in assignments:
        ordered = sorted(
            visits,
            key=lambda v: v['slot_start'] or (now + timedelta(seconds=HORIZON_SECONDS)))
        clock = max(now, phlebo['available_from'] or now)
        current = phlebo['point']

        for visit in ordered:
            metres, secs = _geometric_arc(current, visit['point'], speed_kmph)
            arrival = clock + timedelta(seconds=secs)
            if visit['slot_start'] and arrival < visit['slot_start']:
                arrival = visit['slot_start']

            total_metres += metres
            planned += 1
            if visit['slot_end'] and arrival > visit['slot_end']:
                late_stops += 1

            clock = arrival + timedelta(minutes=service_minutes)
            current = visit['point']

    return {
        "visitsPlanned": planned,
        "totalKm": round(total_metres / 1000, 2),
        "stopsOutsideSlot": late_stops,
    }


@bp.route('/optimize-schedule', methods=['POST'])
def optimize_schedule_endpoint():
    """Stateless solver. Same contract style as /optimize and /reallocate:
    everything needed arrives in the body, nothing is read or written."""
    try:
        data = request.json or {}
        result = solve_schedule(
            data.get('phlebos'),
            data.get('visits'),
            now=parse_datetime(data.get('now')),
            options=data,
        )
        return jsonify(result)
    except (KeyError, TypeError) as e:
        return jsonify({"success": False, "message": f"Malformed phlebos/visits payload: {e}"}), 400
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ---------------------------------------------------------------------------
# /schedule-plan
#
# The solver above is told what the world looks like. This one goes and finds
# out: today's open board from the database, live phlebo positions from
# Firebase, and which jobs are already too far along to be moved.
#
# It only proposes. Nothing is written back, for the same reason
# /reallocation-suggestions writes nothing - the patient has been told who is
# coming, and a silent re-shuffle is worse than none.
# ---------------------------------------------------------------------------


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
    """customer_id out of the caller's JWT - the same token tymon issues on the
    PHP side. It is the only source for it: these rows carry patient names and
    phone numbers, so a body parameter would let anyone read any customer."""
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


def todays_visits(conn, customer_id, city_id=None):
    where = ["visits.customer_id = %s", "visits.is_active = 1",
             "visits.deleted_at is null", "visits.slot_date = curdate()"]
    params = [customer_id]

    if city_id:
        where.append("visits.city_id = %s")
        params.append(city_id)

    with conn.cursor() as cur:
        cur.execute(f"""
            select visits.id, visits.order_id, visits.task_id, visits.order_number,
                   visits.main_patient_name, visits.status, visits.slot_date,
                   visits.slot_time, visits.phlebo_id, visits.lat, visits.lng,
                   visits.city_id, visits.patient_pincode,
                   cities.city_name as city_name,
                   (select pincodes.area_code_id
                      from pincodes
                      where pincodes.pincode = visits.patient_pincode
                            and pincodes.deleted_at is null
                      limit 1) as areacode_id,
                   concat(coalesce(phlebos.first_name, ''), ' ',
                          coalesce(phlebos.last_name, '')) as phlebo_name
            from visits
            left join cities on cities.id = visits.city_id and cities.deleted_at is null
            left join phlebos on phlebos.id = visits.phlebo_id and phlebos.deleted_at is null
            where {' and '.join(where)}
        """, params)
        return cur.fetchall()


def phlebo_roster(conn, phlebo_ids):
    """Names and serving areas for the phlebos who are actually out there.

    The serving columns are optional on purpose: if this deployment's schema
    does not carry them the query is retried without, and every phlebo is then
    allowed everywhere - the same fallback /auto-assign already applies when a
    visit matches nobody's area."""
    if not phlebo_ids:
        return {}

    placeholders = ",".join(["%s"] * len(phlebo_ids))
    base = f"""
        select id, concat(coalesce(first_name, ''), ' ', coalesce(last_name, '')) as name,
               phone{{extra}}
        from phlebos
        where id in ({placeholders}) and deleted_at is null
    """

    for extra in (", serving_areacode_id, serving_city_id", ""):
        try:
            with conn.cursor() as cur:
                cur.execute(base.format(extra=extra), list(phlebo_ids))
                return {int(r['id']): r for r in cur.fetchall()}
        except pymysql.err.MySQLError:
            continue

    return {}


def reached_times(conn, visit_ids):
    """When each on-site phlebo arrived - so a phlebo mid-collection rejoins the
    plan when they finish, not this instant."""
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


def live_locations(customer_id):
    """Where every phlebo actually is, from the path the phlebo app writes to."""
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
                "lat": point[0], "lng": point[1], "status": loc.get('status'),
            }
        except (TypeError, ValueError):
            continue

    return locations


@bp.route('/schedule-plan', methods=['POST'])
def schedule_plan_endpoint():
    try:
        customer_id, auth_error = customer_id_from_token()
        if auth_error:
            return jsonify({"success": False, "message": auth_error}), 401

        body = request.json or {}
        now = datetime.now()
        service_minutes = int(body.get('collection_minutes') or DEFAULT_COLLECTION_MINUTES)
        speed = float(body.get('speed_kmph') or DEFAULT_SPEED_KMPH)
        # Off by default: re-sequencing someone already driving to a door is the
        # kind of change a dispatcher has to defend to a patient on the phone.
        free_enroute = bool(body.get('reassign_enroute'))

        conn = db_connection()
        try:
            rows = todays_visits(conn, customer_id, body.get('city'))
            locations = live_locations(customer_id)

            open_rows = []
            in_progress = []
            for row in rows:
                status = (row['status'] or '').strip().lower()
                if status in TERMINAL_STATUSES:
                    continue
                if status == 'phlebo reached':
                    # Already standing at the door. Nothing to re-route: the
                    # only thing it changes is where that phlebo is free next.
                    in_progress.append(row)
                    continue
                open_rows.append(row)

            reached = reached_times(conn, [int(r['id']) for r in in_progress])
            roster = phlebo_roster(conn, set(locations) | {
                int(r['phlebo_id']) for r in rows if r['phlebo_id']})
        finally:
            conn.close()

        if not locations:
            return jsonify({
                "success": False,
                "message": "No live phlebotomist locations available - nothing to plan against",
            }), 409

        # A phlebo mid-collection starts from that patient's door, free once the
        # draw is done; everyone else starts from wherever they are right now.
        busy_until = {}
        busy_at = {}
        for row in in_progress:
            if not row['phlebo_id']:
                continue
            pid = int(row['phlebo_id'])
            started = parse_datetime(reached.get(int(row['id'])))
            free_at = now
            if started:
                free_at = max(now, started + timedelta(minutes=service_minutes))
            busy_until[pid] = max(busy_until.get(pid, now), free_at)
            point = coerce_point(row['lat'], row['lng'])
            if point:
                busy_at[pid] = point

        phlebos_payload = []
        for phlebo_id, loc in locations.items():
            if loc.get('status') == 'clock_out':
                continue
            info = roster.get(phlebo_id, {})
            point = busy_at.get(phlebo_id, (loc['lat'], loc['lng']))
            phlebos_payload.append({
                "id": phlebo_id,
                "name": (info.get('name') or '').strip(),
                "lat": point[0],
                "lng": point[1],
                "serving_areacode_id": info.get('serving_areacode_id', []),
                "serving_city_id": info.get('serving_city_id', []),
                "available_from": busy_until.get(phlebo_id),
            })

        if not phlebos_payload:
            return jsonify({
                "success": False,
                "message": "Every tracked phlebotomist is clocked out",
            }), 409

        visits_payload = []
        for row in open_rows:
            status = (row['status'] or '').strip().lower()
            pinned = None
            if not free_enroute and status == 'phlebo enroute' and row['phlebo_id']:
                pinned = int(row['phlebo_id'])

            visits_payload.append({
                "id": int(row['id']),
                "lat": row['lat'],
                "lng": row['lng'],
                "slot_date": row['slot_date'],
                "slot_time": row['slot_time'],
                "city_id": row['city_id'],
                "areacode_id": row['areacode_id'],
                "pinned_phlebo_id": pinned,
            })

        if not visits_payload:
            return jsonify({
                "success": True,
                "data": {
                    "date": now.strftime('%Y-%m-%d'),
                    "generatedAt": now.strftime('%Y-%m-%d %H:%M:%S'),
                    "message": "Nothing open to plan - every visit today is finished or in progress",
                    "inProgress": len(in_progress),
                    "routes": [], "unassigned": [], "changes": [],
                },
            })

        plan = solve_schedule(phlebos_payload, visits_payload, now=now, options={
            **body, "collection_minutes": service_minutes, "speed_kmph": speed,
        })

        # What the board would do untouched, measured the same way, so the
        # saving claimed below is a like-for-like comparison and not a
        # better-model-flatters-itself artefact.
        by_phlebo = {p['id']: [] for p in phlebos_payload}
        normalized = {}
        for raw in visits_payload:
            v, reason = normalize_visit(raw, service_minutes)
            if v:
                normalized[raw['id']] = v
        for row in open_rows:
            pid = int(row['phlebo_id']) if row['phlebo_id'] else None
            v = normalized.get(int(row['id']))
            if pid in by_phlebo and v:
                by_phlebo[pid].append(v)

        assignments = []
        for raw in phlebos_payload:
            p = normalize_phlebo(raw)
            if p:
                assignments.append((p, by_phlebo[raw['id']]))

        baseline = simulate_current_plan(assignments, now, speed, service_minutes)

        current_owner = {int(r['id']): (int(r['phlebo_id']) if r['phlebo_id'] else None)
                         for r in open_rows}
        detail = {int(r['id']): r for r in open_rows}
        names = {p['id']: p['name'] for p in phlebos_payload}
        phones = {pid: r.get('phone') for pid, r in roster.items()}

        changes = []
        for route in plan['routes']:
            for stop in route['stops']:
                visit_id = stop['visitId']
                was = current_owner.get(visit_id)
                if was == route['phleboId']:
                    continue
                row = detail.get(visit_id, {})
                changes.append({
                    "visitId": visit_id,
                    "bookingId": row.get('task_id') or row.get('order_number'),
                    "patient": (row.get('main_patient_name') or '').strip() or None,
                    "slotTime": stop['slotTime'],
                    "from": None if not was else {
                        "id": was, "name": (row.get('phlebo_name') or '').strip() or None},
                    "to": {"id": route['phleboId'], "name": names.get(route['phleboId']),
                           "phone": phones.get(route['phleboId'])},
                    "projectedArrival": stop['projectedArrival'],
                    "withinSlot": stop['withinSlot'],
                })

        optimized = plan['summary']
        return jsonify({
            "success": True,
            "data": {
                "date": now.strftime('%Y-%m-%d'),
                "generatedAt": plan['generatedAt'],
                "travelModel": plan['travelModel'],
                "inProgress": len(in_progress),
                "routes": plan['routes'],
                "unassigned": plan['unassigned'],
                "skippedPhlebos": plan['skippedPhlebos'],
                "changes": changes,
                "summary": optimized,
                "comparison": {
                    "current": baseline,
                    "optimized": {
                        "visitsPlanned": optimized['visitsPlanned'],
                        "totalKm": optimized['totalKm'],
                        "stopsOutsideSlot": optimized['stopsOutsideSlot'],
                    },
                    "kmSaved": round(baseline['totalKm'] - optimized['totalKm'], 2),
                    "breachesAvoided": baseline['stopsOutsideSlot'] - optimized['stopsOutsideSlot'],
                    "visitsMoved": len(changes),
                },
                "settings": plan['settings'],
            },
        })
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except KeyError as e:
        return jsonify({"success": False, "message": f"Missing configuration: {e}"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ---------------------------------------------------------------------------
# /batch-plan
#
# The web "Batch assign" page: the dispatcher picks one day, some phlebos and
# some unassigned visits, and wants who-takes-what in driving order against
# the slots - /optimize-schedule's answer. The one thing that endpoint cannot
# know on its own is the country's matching rule: with
# country_settings.phlebo_assign_via_radius on, eligibility is the phlebo's
# serving_radius around their home base, not area/city. That needs the
# caller's country_id (from the token) and phlebo_location_master, so it is
# resolved here and handed to the solver as per-visit allowed_phlebo_ids -
# the same rule, and the same helpers, /reallocate applies.
#
# Like every other planner here it only proposes. PHP's assign_phlebo_batch
# writes what the dispatcher confirms.
# ---------------------------------------------------------------------------

@bp.route('/batch-plan', methods=['POST'])
def batch_plan_endpoint():
    # Reuse /reallocate's token + radius helpers rather than a third copy of
    # them; imported here so this module still runs standalone without them.
    from dynamic_reallocation import (
        customer_id_from_token as token_claims, country_settings,
        homebase_locations_for, allowed_phlebos_for, db_connection as realloc_db,
    )
    try:
        _, country_id, auth_error = token_claims()
        if auth_error:
            return jsonify({"success": False, "message": auth_error}), 401

        body = request.json or {}
        phlebos = body.get('phlebos') or []
        visits = body.get('visits') or []

        settings = None
        if country_id:
            conn = realloc_db()
            try:
                settings = country_settings(conn, country_id)
                if settings and settings.get('phlebo_assign_via_radius'):
                    homebase = homebase_locations_for(conn, phlebos)
                    located = [p for p in phlebos if coerce_point(p.get('lat'), p.get('lng'))]
                    for v in visits:
                        if v.get('pinned_phlebo_id') or not coerce_point(v.get('lat'), v.get('lng')):
                            continue
                        v['allowed_phlebo_ids'] = [
                            p['id'] for p in allowed_phlebos_for(v, located, settings, homebase)
                        ]
            finally:
                conn.close()

        result = solve_schedule(phlebos, visits, now=parse_datetime(body.get('now')), options=body)
        result['matchRule'] = 'radius' if settings and settings.get('phlebo_assign_via_radius') else 'area_city'
        return jsonify(result)
    except (KeyError, TypeError) as e:
        return jsonify({"success": False, "message": f"Malformed phlebos/visits payload: {e}"}), 400
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == '__main__':
    # Standalone mode - app.py mounts this blueprint instead when run together.
    app = Flask(__name__)
    app.register_blueprint(bp)
    port = int(os.environ.get("PORT", 5004))
    app.run(host='0.0.0.0', port=port)
