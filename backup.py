from flask import Flask, request, jsonify
from ortools.constraint_solver import pywrapcp
import math

app = Flask(__name__)

# 🔹 Distance calculation (fast, no API cost)
def distance(a, b):
    return int(math.sqrt(
        (a[0] - b[0])**2 + (a[1] - b[1])**2
    ) * 100000)


@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.json

        locations = data['locations']
        start_index = data['start_index']
        end_index = data.get('end_index')  # optional

        coords = [(loc['lat'], loc['lng']) for loc in locations]
        size = len(coords)

        if size == 0:
            return jsonify({"error": "No locations provided"}), 400

        # 🔹 Distance matrix
        distance_matrix = [
            [distance(coords[i], coords[j]) for j in range(size)]
            for i in range(size)
        ]

        # 🔥 Routing Manager setup
        if end_index is not None:
            # Open route: start → end
            manager = pywrapcp.RoutingIndexManager(size, 1, [start_index], [end_index])
        else:
            # Default: round trip (start → ... → start)
            manager = pywrapcp.RoutingIndexManager(size, 1, start_index)

        routing = pywrapcp.RoutingModel(manager)

        # 🔹 Distance callback
        def distance_callback(from_index, to_index):
            return distance_matrix[
                manager.IndexToNode(from_index)
            ][
                manager.IndexToNode(to_index)
            ]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # 🔹 Solve
        solution = routing.Solve()

        if not solution:
            return jsonify({"error": "No solution found"}), 500

        # 🔹 Extract route
        route = []
        index = routing.Start(0)

        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))

        # ✅ Add last node (important for end_index case)
        route.append(manager.IndexToNode(index))

        # 🔥 If NO end_index → remove return to start
        if end_index is None and len(route) > 1 and route[0] == route[-1]:
            route.pop()

        return jsonify({
            "route": route,
            "start_index": start_index,
            "end_index": end_index
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000)