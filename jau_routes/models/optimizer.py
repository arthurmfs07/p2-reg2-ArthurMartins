# models/optimizer.py
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Tuple, List
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.optimize import curve_fit, minimize_scalar


# -------------------------
#  Fuel model calibration
# -------------------------
def fit_fuel_model(v_data: Sequence[float], f_data: Sequence[float], fuel_price: Optional[float] = None) -> Tuple[float, float]:
    """
    Fit A and B in f(v) = A*v^2 + B/v for fuel consumption per km (L/km).
    If fuel_price is provided, returned A,B are multiplied by fuel_price (R$/km units).
    Returns: (A, B)
    """
    v_arr = np.asarray(v_data, dtype=float)
    f_arr = np.asarray(f_data, dtype=float)

    def model(v, A, B):
        return A * v**2 + B / v

    popt, _ = curve_fit(model, v_arr, f_arr, p0=[1e-5, 1.0])
    A_est, B_est = float(popt[0]), float(popt[1])

    if fuel_price is not None:
        A_est *= float(fuel_price)
        B_est *= float(fuel_price)

    return A_est, B_est


# -------------------------
#  Helpers: classification
# -------------------------
def _is_highway_edge(i: int, j: int, depot_index: int = 0) -> bool:
    """Simple rule: edges from/to depot → highway."""
    return i == depot_index or j == depot_index


# -------------------------
#  Build cost / time matrices (updated to support A,B fuel model)
# -------------------------
def build_cost_time_matrices(dist_matrix: np.ndarray,
                             fuel_price: float,
                             driver_wage: float,
                             v_urban: float = 40.0,
                             v_highway: float = 80.0,
                             eff_urban: float = 3.0,
                             eff_highway: float = 4.0,
                             A_est: Optional[float] = None,
                             B_est: Optional[float] = None,
                             depot_index: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build and return:
      - cost_matrix (R$/arc)
      - time_matrix (h/arc)
      - speed_matrix (km/h)
      - fuel_used_matrix (L/arc)
    If A_est,B_est provided → use model f(v)=A*v²+B/v (in L/km); otherwise fall back to km/L efficiencies.
    """
    n = dist_matrix.shape[0]
    cost_matrix = np.zeros((n, n))
    time_matrix = np.zeros((n, n))
    speed_matrix = np.zeros((n, n))
    fuel_used_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            d = float(dist_matrix[i, j])
            if np.isnan(d) or d <= 0:
                cost_matrix[i, j] = 1e9
                continue

            is_highway = _is_highway_edge(i, j, depot_index)
            v = float(v_highway if is_highway else v_urban)
            time_h = d / v

            # ---- Fuel consumption ----
            if A_est is not None and B_est is not None:
                # Model-based (empirical)
                f_per_km = A_est * v**2 + B_est / v  # L/km
                fuel_L = f_per_km * d
            else:
                # Simple efficiency (km/L)
                eff = float(eff_highway if is_highway else eff_urban)
                fuel_L = d / eff

            fuel_cost = fuel_L * fuel_price
            driver_cost = time_h * driver_wage
            total_cost = fuel_cost + driver_cost

            cost_matrix[i, j] = total_cost
            time_matrix[i, j] = time_h
            speed_matrix[i, j] = v
            fuel_used_matrix[i, j] = fuel_L

    return cost_matrix, time_matrix, speed_matrix, fuel_used_matrix


# -------------------------
#  OR-Tools TSP optimizer (now supports empirical A,B)
# -------------------------
def optimize_costs(dist_matrix: np.ndarray,
                   names: Sequence[str],
                   fuel_price: float = 6.0,
                   driver_wage: float = 30.0,
                   v_urban: float = 30.0,
                   v_highway: float = 80.0,
                   eff_urban: float = 2.8,
                   eff_highway: float = 3.5,
                   A_est: Optional[float] = None,
                   B_est: Optional[float] = None,
                   depot_index: int = 0,
                   time_windows: Optional[Sequence[Tuple[float, float]]] = None,
                   time_limit_seconds: int = 10) -> Tuple[pd.DataFrame, dict, List[int]]:
    """
    Solve TSP minimizing cost (fuel+driver).
    If A_est,B_est provided → fuel uses empirical model.
    Returns (edge DataFrame, summary dict, route node indices).
    """
    cost_matrix, time_matrix, speed_matrix, fuel_matrix = build_cost_time_matrices(
        dist_matrix, fuel_price, driver_wage,
        v_urban=v_urban, v_highway=v_highway,
        eff_urban=eff_urban, eff_highway=eff_highway,
        A_est=A_est, B_est=B_est,
        depot_index=depot_index
    )

    n = cost_matrix.shape[0]
    cost_matrix_int = (cost_matrix * 100).astype(int)

    manager = pywrapcp.RoutingIndexManager(n, 1, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def cost_cb(from_idx, to_idx):
        i, j = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        return int(cost_matrix_int[i, j])

    cb_id = routing.RegisterTransitCallback(cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cb_id)

    # Time dimension for (optional) hard time windows
    time_matrix_s = (time_matrix * 3600).astype(int)
    time_cb_id = routing.RegisterTransitCallback(
        lambda fi, ti: int(time_matrix_s[manager.IndexToNode(fi), manager.IndexToNode(ti)])
    )
    routing.AddDimension(time_cb_id, 24*3600, 24*3600, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    if time_windows is not None:
        if len(time_windows) != n:
            raise ValueError("time_windows length must match number of nodes.")
        for node_idx, (open_h, close_h) in enumerate(time_windows):
            idx = manager.NodeToIndex(node_idx)
            time_dim.CumulVar(idx).SetRange(int(open_h*3600), int(close_h*3600))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_seconds

    sol = routing.SolveWithParameters(params)
    if not sol:
        raise RuntimeError("No feasible route found (tight windows?)")

    # Extract route
    index = routing.Start(0)
    route_nodes = []
    while not routing.IsEnd(index):
        route_nodes.append(manager.IndexToNode(index))
        index = sol.Value(routing.NextVar(index))
    route_nodes.append(manager.IndexToNode(index))

    # Build edge summary
    rows, total_fuel, total_driver, total_cost, total_time = [], 0, 0, 0, 0
    for k in range(len(route_nodes) - 1):
        i, j = route_nodes[k], route_nodes[k+1]
        d = float(dist_matrix[i, j])
        v = float(speed_matrix[i, j])
        fuel_L = float(fuel_matrix[i, j])
        time_h = float(time_matrix[i, j])
        fuel_cost = fuel_L * fuel_price
        driver_cost = time_h * driver_wage
        edge_cost = fuel_cost + driver_cost

        rows.append({
            "From": names[i], "To": names[j],
            "Distance (km)": d, "Speed (km/h)": v,
            "FuelUsed (L)": fuel_L, "FuelCost (R$)": fuel_cost,
            "Time (h)": time_h, "DriverCost (R$)": driver_cost,
            "TotalEdgeCost (R$)": edge_cost
        })

        total_fuel += fuel_cost
        total_driver += driver_cost
        total_cost += edge_cost
        total_time += time_h

    df = pd.DataFrame(rows)
    summary = {
        "Total fuel cost (R$)": total_fuel,
        "Total driver cost (R$)": total_driver,
        "Total cost (R$)": total_cost,
        "Total time (h)": total_time,
        "Route_nodes": route_nodes
    }
    return df, summary, route_nodes


# -------------------------
#  Soft time-window optimization (unchanged)
# -------------------------
def optimize_departure_time_for_route(segments: np.ndarray,
                                      v_profile: Sequence[float],
                                      fuel_price: float,
                                      driver_wage: float,
                                      time_windows: Optional[Sequence[Tuple[float, float]]],
                                      lambda_early: float = 0.0,
                                      lambda_late: float = 1.0,
                                      tau_bounds: Tuple[float, float] = (6.0, 18.0)):
    segments = np.asarray(segments, float)
    v_profile = np.asarray(v_profile, float)
    cumulative_offset = np.concatenate(([0.0], np.cumsum(segments / v_profile)))

    def J(tau):
        times = tau + cumulative_offset
        total_time_h = cumulative_offset[-1]
        driver_cost = total_time_h * driver_wage
        penalty = 0.0
        if time_windows is not None:
            for i, (open_h, close_h) in enumerate(time_windows):
                arr = float(times[i])
                if arr < open_h:
                    penalty += lambda_early * (open_h - arr)**2
                if arr > close_h:
                    penalty += lambda_late * (arr - close_h)**2
        return driver_cost + penalty

    res = minimize_scalar(J, bounds=tau_bounds, method='bounded')
    tau_opt = float(res.x)
    times_opt = tau_opt + cumulative_offset
    return tau_opt, float(res.fun), times_opt


# -------------------------
#  Compute metrics for arbitrary route (unchanged)
# -------------------------
def compute_route_metrics(route: Sequence[int],
                          dist_matrix: np.ndarray,
                          fuel_price: float,
                          driver_wage: float,
                          v_urban: float = 30.0,
                          v_highway: float = 80.0,
                          eff_urban: float = 2.8,
                          eff_highway: float = 3.5,
                          A_est: Optional[float] = None,
                          B_est: Optional[float] = None,
                          depot_index: int = 0,
                          time_windows: Optional[Sequence[Tuple[float, float]]] = None,
                          lambda_early: float = 0.0,
                          lambda_late: float = 1.0,
                          tau: Optional[float] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Recomputes per-edge metrics for a fixed route.
    If A_est and B_est are provided, uses the empirical f(v)=A v^2 + B/v (L/km).
    Otherwise falls back to km/L efficiencies.
    """
    rows = []
    total_fuel_cost = total_driver_cost = total_time_h = 0.0

    for k in range(len(route) - 1):
        i, j = route[k], route[k + 1]
        d = float(dist_matrix[i, j])
        is_highway = (i == depot_index or j == depot_index)
        v = float(v_highway if is_highway else v_urban)
        time_h = d / v

        # Fuel used: empirical (A,B) if provided, else simple eff
        if A_est is not None and B_est is not None:
            f_per_km = float(A_est) * v**2 + float(B_est) / v  # L/km
            fuel_L = f_per_km * d
        else:
            eff = float(eff_highway if is_highway else eff_urban)
            fuel_L = d / eff

        fuel_cost = fuel_L * fuel_price
        driver_cost = time_h * driver_wage
        total_cost_edge = fuel_cost + driver_cost

        rows.append({
            "From": i, "To": j,
            "Distance (km)": d,
            "Speed (km/h)": v,
            "FuelUsed (L)": fuel_L,
            "FuelCost (R$)": fuel_cost,
            "Time (h)": time_h,
            "DriverCost (R$)": driver_cost,
            "TotalEdgeCost (R$)": total_cost_edge
        })
        total_fuel_cost += fuel_cost
        total_driver_cost += driver_cost
        total_time_h += time_h

    df = pd.DataFrame(rows)
    summary = {
        "Total fuel cost (R$)": total_fuel_cost,
        "Total driver cost (R$)": total_driver_cost,
        "Total cost (R$)": total_fuel_cost + total_driver_cost,
        "Total time (h)": total_time_h
    }

    if tau is not None and time_windows is not None:
        arrival = [tau]
        for k in range(len(route) - 1):
            i, j = route[k], route[k + 1]
            d = float(dist_matrix[i, j])
            v = float(v_highway if (i == depot_index or j == depot_index) else v_urban)
            arrival.append(arrival[-1] + d / v)

        penalties = 0.0
        for idx_node, arr in enumerate(arrival[:-1]):
            open_h, close_h = time_windows[idx_node]
            if arr < open_h:
                penalties += lambda_early * (open_h - arr)**2
            if arr > close_h:
                penalties += lambda_late * (arr - close_h)**2

        summary["Penalties (R$)"] = penalties
        summary["Total cost with penalties (R$)"] = summary["Total cost (R$)"] + penalties
        summary["Arrival times (h)"] = arrival

    return df, summary