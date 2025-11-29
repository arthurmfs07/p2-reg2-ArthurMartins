import numpy as np
import pandas as pd
import requests
import time
import os
import math

# ============================================================
# 1. OSRM SETUP
# ============================================================

OSRM_URL = "http://router.project-osrm.org/route/v1/driving/"


def osrm_distance(lat1, lon1, lat2, lon2):
    """
    Consulta o OSRM e retorna a distância em km.
    """
    url = f"{OSRM_URL}{lon1},{lat1};{lon2},{lat2}?overview=false"

    try:
        r = requests.get(url, timeout=5)
        data = r.json()

        if "routes" not in data:
            return None

        meters = data["routes"][0]["distance"]
        return meters / 1000.0  # metros → km

    except Exception:
        return None


# ============================================================
# 2. Haversine (fallback)
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ============================================================
# 3. CACHE PARA ECONOMIZAR CONSULTAS
# ============================================================

cache_file = "jau_routes/data/osrm_cache.csv"

if os.path.exists(cache_file):
    cache = pd.read_csv(cache_file)
else:
    cache = pd.DataFrame(columns=["i", "j", "dist_km"])


def get_cached_distance(i, j):
    row = cache[(cache.i == i) & (cache.j == j)]
    if len(row) > 0:
        return float(row.dist_km.iloc[0])
    return None


def update_cache(i, j, dist):
    global cache
    cache = pd.concat(
        [
            cache,
            pd.DataFrame({"i": [i], "j": [j], "dist_km": [dist]}),
        ],
        ignore_index=True,
    )
    cache.to_csv(cache_file, index=False)


# ============================================================
# 4. Construção da matriz via OSRM
# ============================================================

def build_distance_matrix_osrm(stores_df):
    n = len(stores_df)
    dist_matrix = np.zeros((n, n))

    coords = list(zip(stores_df.latitude, stores_df.longitude))

    print(f"Building {n}×{n} distance matrix using OSRM…")

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            cached = get_cached_distance(i, j)
            if cached is not None:
                dist_matrix[i, j] = cached
                continue

            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]

            # consulta OSRM
            d = osrm_distance(lat1, lon1, lat2, lon2)

            # fallback
            if d is None:
                d = haversine(lat1, lon1, lat2, lon2)

            dist_matrix[i, j] = d
            update_cache(i, j, d)

            print(f"({i} → {j}) = {d:.3f} km")

            # dormir um pouco para evitar bloqueio
            time.sleep(0.06)

    return dist_matrix


# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    print("Loading store coordinates…")
    stores = pd.read_csv("jau_routes/data/jau_sao_carlos_stores.csv")

    dist_matrix = build_distance_matrix_osrm(stores)

    save_path = "jau_routes/data/dist_matrix.npy"
    np.save(save_path, dist_matrix)

    print("\n✓ Distance matrix saved to:", save_path)
    print("✓ Cache saved to:", cache_file)


'''import osmnx as ox
import numpy as np
import pandas as pd
import itertools
import networkx as nx
from shapely.geometry import Point
import geopandas as gpd
from math import radians, sin, cos, sqrt, atan2
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# -----------------------------
#  Helper: Haversine Distance (km)
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def build_distance_matrix_osm(stores_df, place_name="São Carlos, São Paulo, Brazil"):
    """
    Build a distance matrix using OSMnx shortest path lengths (weight='length').
    Uses the unprojected graph G and nearest_nodes(G, lon, lat) for robustness.
    Falls back to haversine if OSM path seems wrong.
    """
    n_total = len(stores_df)
    dist_matrix = np.zeros((n_total, n_total))

    # Load/create graph (unprojected)
    log.info(" Downloading São Carlos road network (unprojected)...")
    G = ox.graph_from_place(place_name, network_type="drive")
    # ensure graph has length attributes (meters)
    # nearest_nodes expects X=lon, Y=lat on an unprojected graph
    # Precompute node KDTree via ox.distance if needed (ox.distance is recommended)
    
    # Precompute list of (lon, lat) for nearest_nodes calls
    coords_lonlat = list(zip(stores_df.longitude, stores_df.latitude))

    # Step A: depot <-> others (we can use haversine for depot edges to be safe)
    depot = stores_df.iloc[0]
    for i in range(1, n_total):
        d = haversine(depot.latitude, depot.longitude, stores_df.loc[i, "latitude"], stores_df.loc[i, "longitude"])
        dist_matrix[0, i] = d
        dist_matrix[i, 0] = d

    # Step B: between non-depot nodes: use OSMnx shortest path on G (meters -> km)
    # For speed: find nearest node once per store
    log.info("🔎 Finding nearest graph nodes for each store (lon,lat)...")
    nearest_nodes = []
    for (lon, lat) in coords_lonlat[1:]:    # skip depot as it's outside São Carlos in your case
        try:
            node = ox.distance.nearest_nodes(G, lon, lat)
            nearest_nodes.append(node)
        except Exception as e:
            log.warning(f"Could not find nearest node for ({lat},{lon}): {e}")
            nearest_nodes.append(None)

    # For every pair i->j among São Carlos stores (indices 1..n_total-1 mapped to nearest_nodes index 0..n_local-1)
    n_local = len(nearest_nodes)
    for i_local, j_local in itertools.product(range(n_local), range(n_local)):
        if i_local == j_local:
            dist_matrix[i_local+1, j_local+1] = 0.0
            continue

        node_i = nearest_nodes[i_local]
        node_j = nearest_nodes[j_local]
        lat_i = stores_df.loc[i_local+1, "latitude"]
        lon_i = stores_df.loc[i_local+1, "longitude"]
        lat_j = stores_df.loc[j_local+1, "latitude"]
        lon_j = stores_df.loc[j_local+1, "longitude"]

        # If either nearest node not found, fallback to haversine
        if node_i is None or node_j is None:
            d_km = haversine(lat_i, lon_i, lat_j, lon_j)
            dist_matrix[i_local+1, j_local+1] = d_km
            continue

        try:
            # shortest path length in meters
            length_m = nx.shortest_path_length(G, node_i, node_j, weight="length")
            d_km = length_m / 1000.0
        except Exception as e:
            log.warning(f" OSMnx shortest_path failed for pair {i_local+1}-{j_local+1}: {e}")
            d_km = np.nan

        # sanity check: if OSM distance is ridiculously larger than haversine, fallback
        d_hav = haversine(lat_i, lon_i, lat_j, lon_j)
        if np.isnan(d_km) or d_km > 3.0 * d_hav + 1e-6:
            # fallback
            log.debug(f"Fallback to haversine for pair {i_local+1}-{j_local+1}: osm={d_km:.2f} km hav={d_hav:.2f} km")
            d_km = d_hav

        dist_matrix[i_local+1, j_local+1] = d_km

    return dist_matrix

if __name__ == "__main__":
    # load CSV (relative to project root)
    stores_path = os.path.join(os.path.dirname(__file__), "..", "data", "jau_sao_carlos_stores.csv")
    stores_path = os.path.abspath(stores_path)
    stores = pd.read_csv(stores_path)
    log.info(f" Loaded {len(stores)} locations (including depot).")

    # Build matrix (this will use OSMnx + fallbacks)
    D = build_distance_matrix_osm(stores, place_name="São Carlos, São Paulo, Brazil")

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "dist_matrix.npy")
    out_path = os.path.abspath(out_path)
    np.save(out_path, D)

    log.info("\n Distance matrix saved!")
    log.info("Shape: %s", D.shape)
    log.info("Has NaN values: %s", np.isnan(D).any())
    log.info("Number of NaNs: %s", np.isnan(D).sum())

    # report range but ignore zeros on diagonal
    offdiag = D[~np.eye(D.shape[0], dtype=bool)]
    log.info("Distance range: %.2f km – %.2f km", np.nanmin(offdiag), np.nanmax(offdiag))

    # Optional: show preview as table
    try:
        df_preview = pd.DataFrame(np.round(D, 2), columns=stores.name, index=stores.name)
        print("\nMatrix Preview (km):")
        print(df_preview)
    except Exception:
        pass
'''