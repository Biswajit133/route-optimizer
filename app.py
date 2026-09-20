from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from dotenv import load_dotenv
import math
import json
import os
import traceback

# Must run before reading CORS_ORIGINS below - the blueprint modules also call
# this on their own import, but that happens further down this file, after
# CORS(app, ...) would otherwise have already read an empty env var.
load_dotenv()

app = Flask(__name__)

# Angular runs on its own origin (localhost:4200 in dev), so the browser needs
# an explicit CORS allow before it'll let those calls through - the JWT check
# in each blueprint still does the actual auth. Comma-separated list so prod
# origins can be added via env without another deploy.
CORS_ORIGINS = [o.strip() for o in os.environ.get(
    'CORS_ORIGINS', 'http://localhost:4200'
).split(',') if o.strip()]
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

# ✅ Accurate Haversine distance (meters)
def distance(a, b):
    R = 6371000  # Earth radius in meters

    lat1, lon1 = a
    lat2, lon2 = b

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    x = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    d = 2 * R * math.atan2(math.sqrt(x), math.sqrt(1 - x))

    return int(d)  # 🔥 IMPORTANT: integer for OR-Tools

def parse_int_list(value):
    if isinstance(value, list):
        return list(map(int, value))

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return list(map(int, parsed))
        except:
            pass

    return []

def get_nearest_route(start, visits):

    route = []
    current = (float(start['lat']), float(start['lng']))
    remaining = visits.copy()

    while remaining:

        nearest = min(
            remaining,
            key=lambda v: distance(
                current,
                (float(v['lat']), float(v['lng']))
            )
        )

        route.append(nearest)

        current = (float(nearest['lat']), float(nearest['lng']))
        remaining.remove(nearest)

    return route

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.json

        locations = data['locations']
        start_index = int(data['start_index'])
        end_index = data.get('end_index')

        if end_index is not None:
            end_index = int(end_index)
        # optional

        coords = [(loc['lat'], loc['lng']) for loc in locations]
        size = len(coords)

        if size == 0:
            return jsonify({"error": "No locations provided"}), 400

        # ✅ Distance matrix using Haversine
        distance_matrix = [
            [distance(coords[i], coords[j]) for j in range(size)]
            for i in range(size)
        ]

        # ✅ Routing Manager
        if end_index is not None:
            manager = pywrapcp.RoutingIndexManager(size, 1, [start_index], [end_index])
        else:
            max_dist = -1
            end_index = start_index

            for i in range(size):
                if i != start_index:
                    d = distance(coords[start_index], coords[i])
                    if d > max_dist:
                        max_dist = d
                        end_index = i
            manager = pywrapcp.RoutingIndexManager(size, 1,[start_index], [end_index])

        routing = pywrapcp.RoutingModel(manager)
        routing.AddVariableMinimizedByFinalizer(routing.NextVar(routing.Start(0)))
        # routing.AddVariableMinimizedByFinalizer(routing.NextVar(routing.End(0)))
        # ✅ Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # ✅ 🔥 Force nearest-like behavior
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Optional: improve solution slightly
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 2

        # ✅ Solve
        solution = routing.SolveWithParameters(search_parameters)

        if not solution:
            return jsonify({"error": "No solution found"}), 500

        # ✅ Extract route
        route = []
        index = routing.Start(0)

        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))

        route.append(manager.IndexToNode(index))

        # ✅ Remove return to start if no end_index
        if end_index is None and len(route) > 1 and route[0] == route[-1]:
            route.pop()

        return jsonify({
            "route": route,
            "start_index": start_index,
            "end_index": end_index
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/auto-assign', methods=['POST'])
def auto_assign():
    try:
        data = request.json

        phlebos = data['phlebos']
        visits = data['visits']

        if not phlebos or not visits:
            return jsonify({"error": "Missing data"}), 400

        # 🔥 INIT PHLEBO STATE
        phlebo_state = {}
        valid_phlebos = []
        for p in phlebos:

            lat = p.get('lat')
            lng = p.get('lng')

            # 🔥 Skip if lat/lng missing or invalid
            if lat in [None, '', 0] or lng in [None, '', 0]:
                continue

            try:
                lat = float(lat)
                lng = float(lng)
            except:
                continue

            valid_phlebos.append(p)
            phlebo_state[p['id']] = {
                "id": p['id'],
                "name": p.get('name', ''),
                "lat": float(p['lat']),
                "lng": float(p['lng']),
                "total_orders": 0,
                "total_distance": 0,
                "visits": []
            }
        if not valid_phlebos:
            return jsonify({"error": "No valid phlebos with location"}), 400

        visit_list = []

        # 🔥 LOOP VISITS
        for v in visits:

            # visit_area = int(v.get('areacode_id', 0))
            # visit_city = int(v.get('city_id'))
            visit_area = int(v.get('areacode_id') or 0)
            visit_city = int(v.get('city_id') or 0)

            allowed_phlebos = []

            # ✅ STEP 1: FILTER PHLEBOS
            for p in valid_phlebos:

                # phlebo_areas = list(map(int, p.get('serving_areacode_id', [])))
                # phlebo_cities = list(map(int, p.get('serving_city_id', [])))
                phlebo_areas = parse_int_list(p.get('serving_areacode_id', []))
                phlebo_cities = parse_int_list(p.get('serving_city_id', []))

                # AREA MATCH
                if visit_area != 0 and visit_area in phlebo_areas:
                    allowed_phlebos.append(p)

                # CITY MATCH
                elif visit_city in phlebo_cities:
                    allowed_phlebos.append(p)

            # ❗ fallback → allow all
            if not allowed_phlebos:
                allowed_phlebos = valid_phlebos

            

            # 🔥 STEP 2: FIND BEST PHLEBO
            best_phlebo = None
            best_score = float('inf')

            for p in allowed_phlebos:

                state = phlebo_state[p['id']]

                # distance
                dist = distance(
                    (float(p['lat']), float(p['lng'])),
                    (float(v['lat']), float(v['lng']))
                )

                # score = distance + load factor
                score = dist + (state['total_orders'] * 5000)

                if score < best_score:
                    best_score = score
                    best_phlebo = p

            # ✅ ASSIGN
            phlebo_id = best_phlebo['id']
            phlebo_state[phlebo_id]['total_orders'] += 1
            phlebo_state[phlebo_id]['total_distance'] += best_score
            phlebo_state[phlebo_id]['visits'].append(v['id'])

            # 🔥 BUILD ALLOWED DROPDOWN
            allowed_dropdown = [
                {"id": p['id'], "name": p.get('name', '')}
                for p in allowed_phlebos
            ]

            visit_list.append({
                "visit_id": v['id'],
                "patient_name": v.get('patient_name', ''),
                "address": v.get('address', ''),
                "slot": v.get('slot', ''),
                "order_id": v.get('order_id', ''),

                "assigned_phlebo_id": phlebo_id,
                "allowed_phlebos": allowed_dropdown
            })

        # 🔥 BUILD SUMMARY
        phlebo_summary = []

        for p in phlebo_state.values():
            assigned_visits = [
                v for v in visits if v['id'] in p['visits']
            ]

            sorted_visits = get_nearest_route(p, assigned_visits)

            total_distance = 0
            current = (float(p['lat']), float(p['lng']))

            for v in sorted_visits:
                d = distance(
                    current,
                    (float(v['lat']), float(v['lng']))
                )
                total_distance += d
                current = (float(v['lat']), float(v['lng']))
            phlebo_summary.append({
                "id": p['id'],
                "name": p['name'],
                "total_orders": p['total_orders'],
                "total_distance": f"{round(total_distance/1000, 2)} KM"
            })

        return jsonify({
            "success": True,
            "phlebos": phlebo_summary,
            "visits": visit_list
        })

    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "trace": traceback.format_exc()
        }), 500


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

# Every AI service is a blueprint so they all serve from this one app on one
# port. Each module still runs on its own (`python delay_prediction.py`) for
# training or debugging - see its __main__ block.
from delay_prediction import bp as delay_bp
from dynamic_reallocation import bp as reallocation_bp
from no_show_prediction import bp as no_show_bp
from ondemand_assignment import bp as ondemand_bp
from route_schedule_optimization import bp as schedule_bp

for blueprint in (delay_bp, reallocation_bp, no_show_bp, ondemand_bp, schedule_bp):
    app.register_blueprint(blueprint)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "routes": sorted(
            rule.rule for rule in app.url_map.iter_rules()
            if rule.endpoint != 'static'
        ),
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
