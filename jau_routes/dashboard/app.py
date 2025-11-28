# (jauvenv) arthur@arthur:~/jau_optim$ streamlit run jau_routes/dashboard/app.py

import sys
import os
import importlib

# ------------------------------------------------------------
#  CONFIGURAÇÃO PARA GARANTIR QUE O ARQUIVO CORRETO É IMPORTADO
# ------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Forçar reload do módulo de interface para evitar cache antigo
import jau_routes.models.interface as interface
importlib.reload(interface)

from jau_routes.models.interface import run_model

# Debug: mostrar qual arquivo está sendo importado
import streamlit as st
#st.write("🔍 Interface importada de:", interface.__file__)

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Jaú Route Optimization",
    layout="wide"
)

st.title("🚚 Jaú Supermercados – Otimizador de Rotas")

# ------------------------------------------------------------
#  INPUT SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("Parâmetros do Modelo")

fuel_price = st.sidebar.number_input("Preço Combustível (R$/L)", 4.0, 10.0, 6.5)
driver_wage = st.sidebar.number_input("Preço do Motorista (R$/h)", 10.0, 100.0, 30.0)
v_urban = st.sidebar.number_input("Velocidade média urbana (km/h)", 10.0, 60.0, 40.0)

tau_min = st.sidebar.number_input("Horário mínimo de partida", 4.0, 18.0, 6.0)
tau_max = st.sidebar.number_input("Horário máximo de partida", 4.0, 18.0, 11.0)

lambda_early = st.sidebar.number_input("Penalidade por chegar cedo", 0.0, 30.0, 2.0)
lambda_late = st.sidebar.number_input("Penalidade por chegar tarde", 0.0, 30.0, 5.0)

run_btn = st.sidebar.button("🔍 Rodar Otimização")

# ------------------------------------------------------------
#  OUTPUT
# ------------------------------------------------------------
if run_btn:

    with st.spinner("Executando modelo..."):
        result = run_model(
            fuel_price=fuel_price,
            driver_wage=driver_wage,
            v_urban=v_urban,
            tau_bounds=(tau_min, tau_max),
            lambda_early=lambda_early,
            lambda_late=lambda_late
        )

    st.success("Otimização Concluída!")

    df = result["df"]
    summary = result["summary"]
    tau_opt = result["tau_opt"]
    arrival = np.round(result["arrival_times"], 2)
    v_star = result["optimal_speed"]
    J_val = result["objective_value"]
    route_nodes = result["route_nodes"]
    total_distance = np.round(np.sum(df['Distance (km)']), 2)
    total_cost = round(J_val, 2) + np.round(np.sum(df['FuelCost (R$)']), 2)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Resumo Geral",
        "📄 Tabela",
        "⏱️ Horários de Chegada",
        "🗺️ Mapa",
        "⚙️ Saída Bruta",
    ])

    # ------------------------------------------------------------
    # TAB 1 — Summary
    # ------------------------------------------------------------

    def format_hour_decimal(x):
        hours = int(x)
        minutes = int(round((x - hours) * 60))
        return f"{hours:02d}:{minutes:02d}"

    with tab1:
        st.subheader("Resumo da Rota")

        arrival_times = np.array(result["arrival_times"])
        total_trip_time = float(arrival_times[-1] - arrival_times[0])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Velocidade ótima v*", f"{v_star:.2f} Km/h")
        c2.metric("Melhor horário de partida t₀", format_hour_decimal(tau_opt))
        c3.metric("Custo total", f"R$ {total_cost}")
        c4.metric("Tempo total da rota", format_hour_decimal(total_trip_time))
        c5.metric("Distância Total", f"{total_distance} Km")

        #st.subheader("Superfície de Custo (Plotly Interativo)")
        st.plotly_chart(result["fig_surface"], use_container_width=True)

        #st.subheader("Contorno do Custo J(v, t₀)")
        st.plotly_chart(result["fig_contour"], use_container_width=True)

        st.write("### 📌 Resumo do OR-Tools")
        clean_summary = {
            k: (np.round(v, 2).tolist() if isinstance(v, np.ndarray)
                else round(v, 2) if isinstance(v, (int, float))
                else v)
            for k, v in summary.items()
        }
        st.json(clean_summary)

    # ------------------------------------------------------------
    # TAB 2 — DataFrame
    # ------------------------------------------------------------
    with tab2:
        st.subheader("Tabela detalhada da rota")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            label="⬇️ Baixar tabela CSV",
            data=df.to_csv(index=False),
            file_name="jau_route_table.csv",
            mime="text/csv"
        )

    # ------------------------------------------------------------
    # TAB 3 — Arrival Times
    # ------------------------------------------------------------
    with tab3:
        st.subheader("Horários de Chegada (por nó)")

        arrival_df = pd.DataFrame({
            "Ponto": route_nodes,
            "Chegada (h)": arrival,
            "Chegada (HH:MM)": [format_hour_decimal(x) for x in arrival]
        })

        st.dataframe(arrival_df[["Ponto", "Chegada (HH:MM)"]], use_container_width=True)
        st.write("### ⏱️ Gráfico de Chegada")
        st.line_chart(arrival_df["Chegada (h)"])

    # ------------------------------------------------------------
    # TAB 4 — Map
    # ------------------------------------------------------------
    with tab4:
        st.subheader("Mapa da Rota")
        st.components.v1.html(open(result["map_path"]).read(), height=650)

    # ------------------------------------------------------------
    # TAB 5 — RAW OUTPUT
    # ------------------------------------------------------------
    with tab5:
        st.subheader("Saída completa da função run_model()")
        st.json(result)

