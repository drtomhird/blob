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

    # initial list
    population = ['D'] * n_doves + ['H'] * n_hawks

    for t in range(1, n_periods + 1):
        # random assignment to food items
        choices = [random.randrange(n_food) for _ in population]
        groups = {}
        for i, who in enumerate(population):
            f = choices[i]
            groups.setdefault(f, []).append(who)

        survivors = []
        offspring = []

        # resolve each food patch
        for group in groups.values():
            if len(group) == 1:
                # solo: guaranteed survive & reproduce
                survivors.append(group[0])
                offspring.append(group[0])
            else:
                random.shuffle(group)
                # pair off
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
                # odd one left
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
n_doves = st.sidebar.number_input("Initial # of Doves", min_value=1, max_value=1000, value=20, step=1)
n_hawks = st.sidebar.number_input("Initial # of Hawks", min_value=1, max_value=1000, value=20, step=1)
n_food = st.sidebar.number_input("Number of Food Items per Period", min_value=1, max_value=1000, value=200, step=1)
n_periods = st.sidebar.number_input("# of Periods", min_value=1, max_value=1000, value=400, step=1)
seed = st.sidebar.number_input("Random Seed (optional, 0=none)", min_value=0, max_value=9999, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### Payoff Matrix (Survival %, Reproduction %)")
# Default payoff matrix values
default = {
    ('D','D'): (100, 0, 100, 0),
    ('D','H'): (50, 0, 100, 50),
    ('H','D'): (100, 50, 50, 0),
    ('H','H'): (0, 0, 0, 0),
}
payoff_matrix = {}
for pair, vals in default.items():
    sd, rd, sh, rh = vals
    sd = st.sidebar.slider(f"Survive {pair[0]} vs {pair[1]}", 0, 100, sd)
    rd = st.sidebar.slider(f"Reproduce {pair[0]} vs {pair[1]}", 0, 100, rd)
    sh = st.sidebar.slider(f"Survive {pair[1]} vs {pair[0]}", 0, 100, sh)
    rh = st.sidebar.slider(f"Reproduce {pair[1]} vs {pair[0]}", 0, 100, rh)
    payoff_matrix[pair] = ((sd/100, rd/100), (sh/100, rh/100))

if st.sidebar.button("Run Simulation"):
    # Run and display
    sd, sh = run_hawk_dove_simulation(
        n_doves, n_hawks, n_food, n_periods, payoff_matrix,
        seed if seed != 0 else None
    )
    df = pd.DataFrame({"Doves": sd, "Hawks": sh})

    st.subheader("Population Over Time")
    st.line_chart(df)

    st.subheader("Final Counts")
    st.write(df.iloc[-1])
