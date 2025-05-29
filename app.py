import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import time

# --- Simulation function ---
def run_hawk_dove_simulation(
    n_doves, n_hawks, n_food, n_periods, payoff_matrix, seed=None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    ts_doves = np.zeros(n_periods + 1, dtype=int)
    ts_hawks = np.zeros(n_periods + 1, dtype=int)
    ts_doves[0], ts_hawks[0] = n_doves, n_hawks
    population = ['D'] * n_doves + ['H'] * n_hawks

    for t in range(1, n_periods + 1):
        if len(population) > 2 * n_food:
            population = random.sample(population, 2 * n_food)
        choices = [random.randrange(n_food) for _ in population]
        groups = {}
        for who, f in zip(population, choices):
            groups.setdefault(f, []).append(who)

        survivors, offspring = [], []
        for group in groups.values():
            if len(group) == 1:
                survivors.append(group[0]); offspring.append(group[0])
            else:
                a, b = random.sample(group, 2)
                (s_a, r_a), (s_b, r_b) = payoff_matrix[(a, b)]
                if random.random() < s_a: survivors.append(a)
                if random.random() < s_b: survivors.append(b)
                if random.random() < r_a: offspring.append(a)
                if random.random() < r_b: offspring.append(b)
        population = survivors + offspring
        ts_doves[t], ts_hawks[t] = population.count('D'), population.count('H')
    return ts_doves, ts_hawks

# --- Streamlit App ---
st.set_page_config(page_title="Hawk–Dove Foraging Simulation", layout="wide")
st.title("🦅 Hawk–Dove Foraging Simulation")

with st.sidebar:
    st.header("Simulation Parameters")
    n_doves = st.number_input("Initial # of Doves", 1, 1000, 20)
    n_hawks = st.number_input("Initial # of Hawks", 1, 1000, 20)
    n_food = st.number_input("Food Items per Period", 1, 10000, 200)
    n_periods = st.number_input("# of Periods", 1, 10000, 400)
    seed = st.number_input("Random Seed (0=None)", 0, 9999, 42)

    st.markdown("---")
    st.subheader("Payoff Matrix (Survive %, Reproduce %)")
    default_payoffs = {('D','D'):(100,0,100,0), ('D','H'):(50,0,100,50), ('H','D'):(100,50,50,0), ('H','H'):(0,0,0,0)}
    payoff_matrix = {}
    for idx, (pair, vals) in enumerate(default_payoffs.items()):
        sd_def, rd_def, sh_def, rh_def = vals
        sd = st.slider(f"Survive {pair[0]} vs {pair[1]}",0,100,sd_def,key=f"sd_{idx}")
        rd = st.slider(f"Reproduce {pair[0]} vs {pair[1]}",0,100,rd_def,key=f"rd_{idx}")
        sh = st.slider(f"Survive {pair[1]} vs {pair[0]}",0,100,sh_def,key=f"sh_{idx}")
        rh = st.slider(f"Reproduce {pair[1]} vs {pair[0]}",0,100,rh_def,key=f"rh_{idx}")
        payoff_matrix[pair] = ((sd/100, rd/100),(sh/100, rh/100))

    st.markdown("---")
    st.subheader("Visualization Options")
    pop_anim = st.checkbox("Animate Population", value=False)
    pct_anim = st.checkbox("Animate Dove %", value=False)
    speed = st.radio("Animation Speed", ["Slow", "Normal", "Fast"], index=1)
    run = st.button("Run Simulation")

if run:
    # Run simulation
    seed_val = None if seed == 0 else seed
    ts_doves, ts_hawks = run_hawk_dove_simulation(n_doves, n_hawks, n_food, n_periods, payoff_matrix, seed_val)
    df = pd.DataFrame({"Doves": ts_doves, "Hawks": ts_hawks})
    df.index.name = 'Period'

    # Animation parameters: skip frames and delay per speed
    skip_map = {'Slow': 1, 'Normal': 2, 'Fast': 5}
    delay_map = {'Slow': 0.1, 'Normal': 0.05, 'Fast': 0.02}
    skip = skip_map[speed]
    delay = delay_map[speed]

    max_pop = max(df["Doves"].max(), df["Hawks"].max())

    # Population Over Time
    st.subheader("Population Over Time")
    placeholder_pop = st.empty()
    if pop_anim:
        for i in range(0, len(df), skip):
            fig, ax = plt.subplots()
            ax.plot(df.index[:i+1], df['Doves'][:i+1], label='Doves')
            ax.plot(df.index[:i+1], df['Hawks'][:i+1], label='Hawks')
            ax.set(xlabel='Period', ylabel='Population', ylim=(0, max_pop * 1.1))
            ax.legend()
            placeholder_pop.pyplot(fig)
            time.sleep(delay)
    else:
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Doves'], label='Doves')
        ax.plot(df.index, df['Hawks'], label='Hawks')
        ax.set(xlabel='Period', ylabel='Population', ylim=(0, max_pop * 1.1))
        ax.legend()
        placeholder_pop.pyplot(fig)

    # Final Counts
    st.subheader("Final Counts")
    st.write(df.iloc[-1])

    # Dove % Over Time
    percent = df['Doves'] / df.sum(axis=1) * 100
    st.subheader("Dove % Over Time")
    placeholder_pct = st.empty()
    if pct_anim:
        for i in range(0, len(df), skip):
            fig, ax = plt.subplots()
            ax.plot(df.index[:i+1], percent[:i+1], label='Dove %')
            ax.set(xlabel='Period', ylabel='Percentage', ylim=(0, 100))
            ax.legend()
            placeholder_pct.pyplot(fig)
            time.sleep(delay)
    else:
        fig, ax = plt.subplots()
        ax.plot(df.index, percent, label='Dove %')
        ax.set(xlabel='Period', ylabel='Percentage', ylim=(0, 100))
        ax.legend()
        placeholder_pct.pyplot(fig)
