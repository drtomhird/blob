import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Simulation function ---
def run_hawk_dove_simulation(
    n_doves, n_hawks, n_food, n_periods, payoff_matrix, seed=None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # time series storage
    ts_doves = np.zeros(n_periods + 1, dtype=int)
    ts_hawks = np.zeros(n_periods + 1, dtype=int)
    ts_doves[0], ts_hawks[0] = n_doves, n_hawks

    # initial population list
    population = ['D'] * n_doves + ['H'] * n_hawks

    for t in range(1, n_periods + 1):
        # each individual picks a random food item
        choices = [random.randrange(n_food) for _ in population]
        groups = {}
        for i, who in enumerate(population):
            f = choices[i]
            groups.setdefault(f, []).append(who)

        survivors = []
        offspring = []

        # resolve interactions at each food item
        for group in groups.values():
            if len(group) == 1:
                survivors.append(group[0])
                offspring.append(group[0])
            else:
                random.shuffle(group)
                while len(group) >= 2:
                    a, b = group.pop(), group.pop()
                    (s_a, r_a), (s_b, r_b) = payoff_matrix[(a, b)]
                    if random.random() < s_a:
                        survivors.append(a)
                    if random.random() < s_b:
                        survivors.append(b)
                    if random.random() < r_a:
                        offspring.append(a)
                    if random.random() < r_b:
                        offspring.append(b)
                if group:
                    lone = group.pop()
                    survivors.append(lone)
                    offspring.append(lone)

        population = survivors + offspring
        ts_doves[t] = population.count('D')
        ts_hawks[t] = population.count('H')

    return ts_doves, ts_hawks

# --- Streamlit App ---
st.set_page_config(page_title="Hawk–Dove Foraging Simulation", layout="wide")
st.title("🦅 Hawk–Dove Foraging Simulation")

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
n_doves = st.sidebar.number_input(
    "Initial # of Doves", min_value=1, max_value=1000, value=20, step=1
)
n_hawks = st.sidebar.number_input(
    "Initial # of Hawks", min_value=1, max_value=1000, value=20, step=1
)
n_food = st.sidebar.number_input(
    "Number of Food Items per Period", min_value=1, max_value=10000, value=200, step=1
)
n_periods = st.sidebar.number_input(
    "# of Periods", min_value=1, max_value=10000, value=400, step=1
)
seed = st.sidebar.number_input(
    "Random Seed (0 = None)", min_value=0, max_value=9999, value=42, step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Payoff Matrix (Survival %, Reproduction %)")

# Default payoff values for each ordered pair
default_payoffs = {
    ('D','D'): (100, 0, 100, 0),
    ('D','H'): (50, 0, 100, 50),
    ('H','D'): (100, 50, 50, 0),
    ('H','H'): (0, 0, 0, 0),
}

payoff_matrix = {}
# Use enumerate to ensure unique keys for each slider
for idx, (pair, vals) in enumerate(default_payoffs.items()):
    sd_def, rd_def, sh_def, rh_def = vals
    sd = st.sidebar.slider(
        f"Survive {pair[0]} vs {pair[1]}", 0, 100, sd_def,
        key=f"survive_{idx}_{pair[0]}_{pair[1]}"
    )
    rd = st.sidebar.slider(
        f"Reproduce {pair[0]} vs {pair[1]}", 0, 100, rd_def,
        key=f"reproduce_{idx}_{pair[0]}_{pair[1]}"
    )
    sh = st.sidebar.slider(
        f"Survive {pair[1]} vs {pair[0]}", 0, 100, sh_def,
        key=f"survive_swap_{idx}_{pair[1]}_{pair[0]}"
    )
    rh = st.sidebar.slider(
        f"Reproduce {pair[1]} vs {pair[0]}", 0, 100, rh_def,
        key=f"reproduce_swap_{idx}_{pair[1]}_{pair[0]}"
    )
    payoff_matrix[pair] = ((sd/100, rd/100), (sh/100, rh/100))

# Execute simulation when button is clicked
if st.sidebar.button("Run Simulation"):
    seed_val = seed if seed != 0 else None
    ts_doves, ts_hawks = run_hawk_dove_simulation(
        n_doves, n_hawks, n_food, n_periods, payoff_matrix, seed_val
    )

    df = pd.DataFrame({"Doves": ts_doves, "Hawks": ts_hawks})

    st.subheader("Population Over Time")
    st.line_chart(df)

    st.subheader("Final Counts")
    st.write(df.iloc[-1])
