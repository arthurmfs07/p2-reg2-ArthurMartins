import numpy as np
import pandas as pd
import os

from .optimizer import (
    fit_fuel_model,
    optimize_costs,
    optimize_departure_time_for_route
)

from .visualizer import (
    generate_map,
    compute_cost_surface,
    plot_cost_surface_interactive,
    plot_cost_contour_interactive
)


def run_model(
    fuel_price=6.50,
    driver_wage=30.0,
    v_urban=40.0,
    eff_urban=3.0,
    eff_highway=4.0,
    tau_bounds=(6.0, 11.0),
    lambda_early=0.5,
    lambda_late=5.0,
):
    # 1. Load matrices
    dist_matrix = np.load("jau_routes/data/dist_matrix.npy")

    names = [
        "Depot",
        "Jaú Supermercados - Av. São Carlos1",
        "Jaú Supermercados - Av. São Carlos2",
        "Jaú Supermercados - Vila Prado",
        "Jaú Supermercados - Padre Teixeira",
        "Jaú Supermercados - Raimundo Correa",
        "Jaú Supermercados - Vila Nery",
        "Jaú Supermercados - Redencao",
        "Jaú Supermercados - Visconde",
    ]

    # 2. Fuel model fit f(v)=Av² - Bv + C
    v_meas = np.array([30, 40, 50, 60, 70, 80])
    f_meas = np.array([0.14, 0.13, 0.12, 0.115, 0.11, 0.12])
    A_est, B_est = fit_fuel_model(v_meas, f_meas)

    # 3. Optimal speed
    v_star = ((fuel_price * B_est + driver_wage) / (2 * fuel_price * A_est)) ** (1/3)

    # 4. Time windows
    time_windows = [(8.0, 22.0)] * len(names)
    depot_index = 0

    # 5. OR-Tools route optimization
    df, summary, route_nodes = optimize_costs(
        dist_matrix,
        names,
        fuel_price=fuel_price,
        driver_wage=driver_wage,
        v_urban=v_urban,
        v_highway=v_star,
        eff_urban=eff_urban,
        eff_highway=eff_highway,
        A_est=A_est, B_est=B_est,
        depot_index=depot_index,
        time_windows=time_windows,
        time_limit_seconds=5,
    )

    # Save route
    route_path = "jau_routes/results/route_baseline.npy"
    np.save(route_path, route_nodes)

    # 6. Continuous optimization of departure time
    segments = [
        dist_matrix[route_nodes[i], route_nodes[i+1]]
        for i in range(len(route_nodes)-1)
    ]

    v_profile = [
        v_star if (route_nodes[i] == depot_index or route_nodes[i+1] == depot_index)
        else v_urban
        for i in range(len(route_nodes)-1)
    ]

    tau_opt, J_val, arrival_times = optimize_departure_time_for_route(
        segments=np.array(segments),
        v_profile=v_profile,
        fuel_price=fuel_price,
        driver_wage=driver_wage,
        time_windows=time_windows,
        lambda_early=lambda_early,
        lambda_late=lambda_late,
        tau_bounds=tau_bounds,
    )

    # 7. Map
    map_path = generate_map(route_nodes)

    # 8. Cost surface + contour
    V, T0, Z = compute_cost_surface(
        dist_matrix=dist_matrix,
        route=route_nodes,
        A=A_est,
        B=B_est,
        fuel_price=fuel_price,
        driver_wage=driver_wage,
        time_windows=time_windows,
        lambda_early=lambda_early,
        lambda_late=lambda_late,
        v_urban=v_urban,
        v_highway=v_star,
        depot_index=depot_index,
    )

    fig_surface = plot_cost_surface_interactive(V, T0, Z)
    fig_contour = plot_cost_contour_interactive(V, T0, Z)

    print("DEBUG: fig_surface =", type(fig_surface))
    print("DEBUG: fig_contour =", type(fig_contour))


    return {
        "df": df,
        "summary": summary,
        "tau_opt": tau_opt,
        "arrival_times": arrival_times,
        "route_nodes": route_nodes,
        "optimal_speed": v_star,
        "objective_value": J_val,
        "map_path": map_path,
        "fig_surface": fig_surface,
        "fig_contour": fig_contour,
    }

