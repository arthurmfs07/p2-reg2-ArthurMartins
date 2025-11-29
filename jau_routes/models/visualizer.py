import folium
import pandas as pd
import numpy as np
import requests
import polyline
from folium.plugins import PolyLineTextPath
import plotly.graph_objects as go


def get_osrm_route(lat1, lon1, lat2, lon2):
    """Consulta OSRM e retorna lista de coordenadas seguindo as ruas."""
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    )

    r = requests.get(url)
    data = r.json()

    if "routes" not in data:
        return None

    coords = data["routes"][0]["geometry"]["coordinates"]
    return [(lat, lon) for lon, lat in coords]


def generate_map(route_nodes, save_path="jau_routes/results/jau_route_real_map.html"):
    stores = pd.read_csv("jau_routes/data/jau_sao_carlos_stores.csv")

    m = folium.Map(location=[-22.01, -47.89], zoom_start=13)

    for i, row in stores.iterrows():
        color = "red" if i == 0 else "blue"
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=row["name"],
            icon=folium.Icon(color=color),
        ).add_to(m)

    coords = stores[["latitude", "longitude"]].to_numpy()

    for idx in range(len(route_nodes) - 1):
        i = route_nodes[idx]
        j = route_nodes[idx + 1]

        lat1, lon1 = coords[i]
        lat2, lon2 = coords[j]

        street_path = get_osrm_route(lat1, lon1, lat2, lon2)
        if street_path is None:
            street_path = [(lat1, lon1), (lat2, lon2)]

        outline = folium.PolyLine(
            street_path, color="black", weight=9, opacity=0.6
        ).add_to(m)

        main = folium.PolyLine(
            street_path, color="red", weight=5, opacity=1.0
        ).add_to(m)

        # Route order label
        PolyLineTextPath(
            main,
            f" {idx+1} ",
            repeat=False,
            offset=25,
            center=True,
            attributes={
                "fill": "black",
                "font-weight": "bold",
                "font-size": "20",
            },
        ).add_to(m)

    m.save(save_path)
    return save_path


# ============================================================
#  COST SURFACE + PLOTLY
# ============================================================

def compute_cost_surface(
    dist_matrix,
    route,
    A, B,
    fuel_price,
    driver_wage,
    time_windows,
    lambda_early,
    lambda_late,
    v_urban=40.0,
    v_highway=80.0,
    depot_index=0,
    v_range=(40, 120),
    t0_range=(6, 12),
    n_v=150,
    n_t=150
):
    """
    Computes J(v, t0) over grid.
    Returns V, T0, Z matrices.
    """

    v_grid = np.linspace(*v_range, n_v)
    t0_grid = np.linspace(*t0_range, n_t)

    V, T0 = np.meshgrid(v_grid, t0_grid)
    Z = np.zeros_like(V)

    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = float(V[i, j])
            tau = float(T0[i, j])

            total_fuel_cost = 0.0
            total_driver_cost = 0.0
            cumulative_times = [tau]

            # Travel segments
            for k in range(len(route) - 1):
                a = route[k]
                b = route[k + 1]

                d = float(dist_matrix[a, b])
                t_seg = d / v
                cumulative_times.append(cumulative_times[-1] + t_seg)

                # Fuel model
                fpkm = A * v**2 + B / v
                total_fuel_cost += fpkm * d * fuel_price

                total_driver_cost += t_seg * driver_wage

            # Penalties
            penalty = 0.0
            arrival_times = cumulative_times[:-1]

            for idx, arr in enumerate(arrival_times):
                open_h, close_h = time_windows[idx]
                if arr < open_h:
                    penalty += lambda_early * (open_h - arr)**2
                if arr > close_h:
                    penalty += lambda_late * (arr - close_h)**2

            Z[i, j] = total_fuel_cost + total_driver_cost + penalty

    return V, T0, Z


def plot_cost_surface_interactive(V, T0, Z, title="Custo Total J(v, t₀) — Superfície"):

    # Main smooth surface
    surface = go.Surface(
        x=V,
        y=T0,
        z=Z,
        colorscale="Viridis",
        opacity=0.9
    )

    wireframe = go.Surface(
        x=V,
        y=T0,
        z=Z,
        showscale=False,           
        opacity=0.15,              
        colorscale=[[0, "black"], [1, "black"]],
        contours=dict(
            x=dict(show=True, color="black", width=2),
            y=dict(show=True, color="black", width=2)
        )
    )

    fig = go.Figure(data=[surface, wireframe])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Velocidade (km/h)",
            yaxis_title="Horário de Partida (h)",
            zaxis_title="Custo Total (R$)"
        ),
        height=650
    )

    return fig



def plot_cost_contour_interactive(V, T0, Z, title="Contorno de J(v, t₀)"):
    fig = go.Figure(
        data=go.Contour(
            x=V[0, :],
            y=T0[:, 0],
            z=Z,
            colorscale="Viridis",
            contours=dict(showlines=False)
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Velocidade (km/h)",
        yaxis_title="Horário de Partida (h)",
        height=650
    )
    return fig


