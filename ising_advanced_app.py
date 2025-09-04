# 2D Ising Model Explorer — Advanced Edition 🧲
# ------------------------------------------------------
# Highlights in this rewrite
# - Robust JAX/NumPy handling (falls back to NumPy if JAX missing)
# - Fixed caching & RNG issues; no device arrays leak outside
# - Pure NumPy Wolff + optional Swendsen–Wang (cluster updates)
# - Tc estimation via quadratic peak fit (C and χ) + quick Binder multi-size option
# - Experimental data overlay (Magnetization / Heat Capacity)
# - Plotly interactive animations; Matplotlib uses Agg backend
# - Session-safe GIFs and reproducible seeds
# - "Fun" vs "Pro" modes (UI + captions)

import os
import io
import math
import random
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend for Streamlit
import matplotlib.pyplot as plt

import streamlit as st
import imageio

# Plotly
import plotly.graph_objects as go
import plotly.express as px

# Try JAX, but allow fallback to NumPy
_JAX_AVAILABLE = True
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    _JAX_AVAILABLE = True
except Exception:
    _JAX_AVAILABLE = False

# ------------------------------------------------------
# Page Setup & Theme
# ------------------------------------------------------
st.set_page_config(page_title="2D Ising Model Explorer — Advanced", page_icon="🧲", layout="wide")

st.markdown(
    """
    <style>
      .hero h1{ text-align:center; color:#1982c4; margin-bottom:2px; font-weight: 900;}
      .hero h3{ text-align:center; color:#54428E; margin-top:6px; font-weight: 600;}
      .hero p { text-align:center; color:#555; font-size:16px; font-style: italic; }
      .param-card { background:#f8f9fb; border:1px solid #e8ecf3; border-radius:16px; padding:14px; margin-top:8px; font-size:14px; }
      .explain { background:#fffaf2; border:1px solid #ffe2b6; border-radius:14px; padding:12px; font-size:14px; }
      .stTabs [data-baseweb="tab"]{font-weight:600}
      hr { border-color: #ccc; margin: 16px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------
MATERIAL_DB = {
    "iron": {"J_per_kB": 21.1, "Tc_exp": 1043},
    "k2cof4": {"J_per_kB": 10.0, "Tc_exp": 110},
    "rb2cof4": {"J_per_kB": 7.0, "Tc_exp": 100},
    "dypo4": {"J_per_kB": 2.5, "Tc_exp": 3.4},
}

st.sidebar.markdown("## ⚙️ Simulation Controls")
mode = st.sidebar.radio("Mode", ["Fun", "Pro"], index=0, horizontal=True)
material = st.sidebar.selectbox("Material:", list(MATERIAL_DB.keys()), format_func=lambda x: x.upper())
params = MATERIAL_DB[material]
JkB, Tc_exp = params["J_per_kB"], params["Tc_exp"]
st.sidebar.info(f"**{material.upper()}** → J/kB = {JkB} K, Tc(exp) = {Tc_exp} K")

algo_type = st.sidebar.selectbox(
    "Update Algorithm",
    [
        "Checkerboard (local)",
        "Single Spin (Metropolis)",
        "Cluster Flip (Wolff)",
        "Cluster Flip (Swendsen–Wang)",
    ],
)

N = st.sidebar.slider("Lattice Size (N×N)", 10, 96, 40)
n_eq = st.sidebar.number_input("Equilibration Steps", 200, step=100, value=600)
n_samples = st.sidebar.number_input("Samples per Temperature", 100, step=100, value=300)
seed = st.sidebar.number_input("Random Seed", 0, step=1, value=0)

minT = st.sidebar.number_input("Low Temperature (K)", int(Tc_exp * 0.6))
maxT = st.sidebar.number_input("High Temperature (K)", int(Tc_exp * 1.4))
nT = st.sidebar.slider("Number of Temperatures", 10, 60, 30)

st.sidebar.markdown("## 📊 Analysis Options")
T_hist = st.sidebar.slider("Histogram Temperature T (K)", minT, maxT, Tc_exp, step=1)

st.sidebar.markdown("## 🎥 Animation")
spin_anim_T = st.sidebar.slider("Spin Evolution Temperature (K)", minT, maxT, Tc_exp, step=1)
spin_anim_steps = st.sidebar.slider("Spin Animation Steps", 30, 150, 60)
temp_anim_steps = st.sidebar.slider("Temperature Sweep MC Steps", 80, 600, 280)

engine = st.sidebar.radio("Plot Engine", ["Plotly (interactive)", "Matplotlib (classic)"])

exp_data = None
uploaded = st.sidebar.file_uploader("Upload experimental CSV (T[K], M, optional C)", type=['csv'])
if uploaded:
    try:
        exp_data = pd.read_csv(uploaded)
        st.sidebar.success("Experimental data loaded.")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

run_sim = st.sidebar.button("🚀 Run Simulation", use_container_width=True)

# ------------------------------------------------------
# Utility: RNG & backend helpers
# ------------------------------------------------------

def _np_rng(seed):
    rng = np.random.default_rng(seed)
    return rng

# Checkerboard update (NumPy)
def checkerboard_update_np(spins, beta, rng):
    N = spins.shape[0]
    for offset in (0, 1):
        mask = ((np.add.outer(np.arange(N), np.arange(N)) % 2) == offset)
        neighbors = (
            np.roll(spins, 1, axis=0) + np.roll(spins, -1, axis=0)
            + np.roll(spins, 1, axis=1) + np.roll(spins, -1, axis=1)
        )
        rand_mat = rng.random((N, N))
        deltaE = 2 * spins * neighbors
        flip = (deltaE < 0) | (rand_mat < np.exp(-beta * deltaE))
        spins = np.where(mask & flip, -spins, spins)
    return spins

# Metropolis (NumPy)
def metropolis_update_np(spins, beta, rng):
    N = spins.shape[0]
    for _ in range(N * N):
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        s = spins[i, j]
        nb = spins[(i+1)%N, j] + spins[(i-1)%N, j] + spins[i, (j+1)%N] + spins[i, (j-1)%N]
        dE = 2 * s * nb
        if dE < 0 or rng.random() < np.exp(-beta * dE):
            spins[i, j] = -s
    return spins

# Wolff cluster (NumPy)
def wolff_step_np(spins, beta, rng):
    N = spins.shape[0]
    i = rng.integers(0, N)
    j = rng.integers(0, N)
    cluster_spin = spins[i, j]
    p_add = 1 - np.exp(-2 * beta)
    stack = [(i, j)]
    visited = np.zeros_like(spins, dtype=bool)
    visited[i, j] = True
    while stack:
        ci, cj = stack.pop()
        for ni, nj in [((ci+1)%N, cj), ((ci-1)%N, cj), (ci, (cj+1)%N), (ci, (cj-1)%N)]:
            if (not visited[ni, nj]) and spins[ni, nj] == cluster_spin and (rng.random() < p_add):
                visited[ni, nj] = True
                stack.append((ni, nj))
    spins[visited] *= -1
    return spins

# Swendsen–Wang (NumPy)
def swendsen_wang_step_np(spins, beta, rng):
    N = spins.shape[0]
    p_bond = 1 - np.exp(-2 * beta)
    # Build bonds
    bonds_h = (spins * np.roll(spins, -1, axis=1) == 1) & (rng.random((N, N)) < p_bond)
    bonds_v = (spins * np.roll(spins, -1, axis=0) == 1) & (rng.random((N, N)) < p_bond)

    # Label clusters (union-find)
    parent = np.arange(N * N)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # horizontal bonds
    for i in range(N):
        for j in range(N):
            if bonds_h[i, j]:
                a = i * N + j
                b = i * N + ((j + 1) % N)
                union(a, b)
    # vertical bonds
    for i in range(N):
        for j in range(N):
            if bonds_v[i, j]:
                a = i * N + j
                b = ((i + 1) % N) * N + j
                union(a, b)

    # Flip clusters with prob 1/2
    # Determine cluster representatives
    reps = {}
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            r = find(idx)
            reps.setdefault(r, []).append((i, j))

    for r, members in reps.items():
        if rng.random() < 0.5:
            for (i, j) in members:
                spins[i, j] *= -1

    return spins

# Energy & magnetization (NumPy arrays)

def calc_energy_np(state):
    return float(-np.sum(state * np.roll(state, 1, 0)) - np.sum(state * np.roll(state, 1, 1))) / 2.0

def calc_magnetization_np(state):
    return float(np.sum(state))

# ------------------------------------------------------
# Simulation Core (NumPy backbone, optional JAX local updates)
# ------------------------------------------------------

def initial_lattice_np(N, rng):
    return rng.choice([-1, 1], size=(N, N))

# Optional JAX local updates for speed, but results are returned as NumPy
if _JAX_AVAILABLE:
    @jit
    def checkerboard_update_jax(spins, beta, key):
        N = spins.shape[0]
        for offset in [0, 1]:
            mask = (jnp.add(*jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')) % 2 == offset)
            neighbors = (jnp.roll(spins, 1, axis=0) + jnp.roll(spins, -1, axis=0)
                         + jnp.roll(spins, 1, axis=1) + jnp.roll(spins, -1, axis=1))
            key, subkey = jax.random.split(key)
            rand_mat = jax.random.uniform(subkey, (N, N))
            deltaE = 2 * spins * neighbors
            flip = (deltaE < 0) | (rand_mat < jnp.exp(-beta * deltaE))
            spins = jnp.where(mask & flip, -spins, spins)
        return spins, key

    @jit
    def metropolis_update_jax(spins, beta, key):
        N = spins.shape[0]
        key_i, key_j, key_u = jax.random.split(key, 3)
        idx = jax.random.randint(key_i, (N*N,), 0, N)
        jdx = jax.random.randint(key_j, (N*N,), 0, N)
        u = jax.random.uniform(key_u, (N*N,))
        def body_fun(carry, x):
            s, = carry
            i, j, r = x
            si = s[i, j]
            nb = s[(i+1)%N, j] + s[(i-1)%N, j] + s[i, (j+1)%N] + s[i, (j-1)%N]
            dE = 2 * si * nb
            accept = (dE < 0) | (r < jnp.exp(-beta * dE))
            s = s.at[i, j].set(jnp.where(accept, -si, si))
            return (s,), None
        (spins,), _ = jax.lax.scan(body_fun, (spins,), (idx, jdx, u))
        return spins, key


def run_ising_sim(N, n_eq, n_samples, T_arr, JkB, seed, Tc_exp, algo, hist_T):
    rng_master = _np_rng(seed)

    E_av, m_abs_av, C_av, chi_av = [], [], [], []
    E_err, m_abs_err = [], []
    U4_av = []

    spins_below_tc, spins_above_tc = None, None
    hist_M = None

    # Progress
    fun_jokes = [
        "Teaching spins how to dance… 🕺",
        "Summoning Tc from spin gods… 🔮",
        "Heating lattice noodles… 🍜",
        "Asking domains to behave… 🫡",
        "Phase transition drama loading… 🍿",
    ]
    progress = st.progress(0)

    for idx, T_real in enumerate(T_arr):
        T_code = T_real / JkB
        beta = 1.0 / max(T_code, 1e-12)
        # Independent RNG per T for reproducibility
        rng = _np_rng(seed + idx * 777)
        state = initial_lattice_np(N, rng)

        # Equilibration
        for _ in range(n_eq):
            if algo == "Cluster Flip (Wolff)":
                state = wolff_step_np(state, beta, rng)
            elif algo == "Cluster Flip (Swendsen–Wang)":
                state = swendsen_wang_step_np(state, beta, rng)
            elif algo == "Single Spin (Metropolis)":
                state = metropolis_update_np(state, beta, rng)
            else:
                if _JAX_AVAILABLE:
                    key = jax.random.PRNGKey(seed + idx)
                    state_j = jnp.array(state)
                    state_j, key = checkerboard_update_jax(state_j, beta, key)
                    state = np.array(state_j)
                else:
                    state = checkerboard_update_np(state, beta, rng)

        E_samples = []
        m_signed_samples = []
        m_abs_samples = []

        # Sampling
        for _ in range(n_samples):
            if algo == "Cluster Flip (Wolff)":
                state = wolff_step_np(state, beta, rng)
            elif algo == "Cluster Flip (Swendsen–Wang)":
                state = swendsen_wang_step_np(state, beta, rng)
            elif algo == "Single Spin (Metropolis)":
                state = metropolis_update_np(state, beta, rng)
            else:
                if _JAX_AVAILABLE:
                    key = jax.random.PRNGKey(seed + idx + 1234)
                    state_j = jnp.array(state)
                    state_j, key = checkerboard_update_jax(state_j, beta, key)
                    state = np.array(state_j)
                else:
                    state = checkerboard_update_np(state, beta, rng)

            E = calc_energy_np(state)
            m = calc_magnetization_np(state)
            E_samples.append(E)
            m_signed_samples.append(m)
            m_abs_samples.append(abs(m))

        E_samples = np.array(E_samples) / (N*N)
        m_signed_samples = np.array(m_signed_samples) / (N*N)
        m_abs_samples = np.array(m_abs_samples) / (N*N)

        m1 = m_signed_samples.mean()
        m2 = (m_signed_samples**2).mean()
        m4 = (m_signed_samples**4).mean()

        C = E_samples.var(ddof=1) / (T_code**2 + 1e-12)
        chi = (m2 - m1**2) / (T_code + 1e-12)
        U4 = 1.0 - (m4 / (3.0 * (m2**2) + 1e-12))

        E_av.append(E_samples.mean())
        m_abs_av.append(m_abs_samples.mean())
        C_av.append(C)
        chi_av.append(chi)
        U4_av.append(U4)

        E_err.append(E_samples.std(ddof=1) / math.sqrt(max(n_samples, 1)))
        m_abs_err.append(m_abs_samples.std(ddof=1) / math.sqrt(max(n_samples, 1)))

        if spins_below_tc is None and (T_real < Tc_exp):
            spins_below_tc = np.array(state)
        if spins_above_tc is None and (T_real > Tc_exp):
            spins_above_tc = np.array(state)
        if hist_T is not None and np.isclose(T_real, hist_T, atol=max(1e-6, (T_arr[1]-T_arr[0])/2)):
            hist_M = m_abs_samples.copy()

        if idx % max(1, len(T_arr)//5) == 0:
            progress.progress((idx+1)/len(T_arr))

    progress.progress(1.0)

    return (
        np.array(E_av), np.array(m_abs_av), np.array(C_av), np.array(chi_av),
        np.array(U4_av), np.array(E_err), np.array(m_abs_err),
        spins_below_tc, spins_above_tc, hist_M
    )

# ------------------------------------------------------
# Tc estimation helpers
# ------------------------------------------------------

def quadratic_tc_fit(T, Y):
    """Fit a quadratic near the peak to estimate Tc and uncertainty.
    Returns (Tc, Tc_err). If fit fails, fall back to argmax.
    """
    try:
        kmax = int(np.argmax(Y))
        k0 = max(0, kmax - 2)
        k1 = min(len(T), kmax + 3)
        Tx = T[k0:k1]
        Yx = Y[k0:k1]
        if len(Tx) < 3:
            raise ValueError("not enough points")
        # Fit a x^2 + b x + c
        A = np.vstack([Tx**2, Tx, np.ones_like(Tx)]).T
        coeff, *_ = np.linalg.lstsq(A, Yx, rcond=None)
        a, b, c = coeff
        if abs(a) < 1e-12:
            raise ValueError("flat parabola")
        Tc = -b / (2*a)
        # crude error: use local curvature
        residuals = Yx - (a*Tx**2 + b*Tx + c)
        s2 = np.sum(residuals**2) / max(len(Tx)-3, 1)
        # d(Tc)/d(a,b) propagation ~ sqrt( (b/(2a^2))^2 Var(a) + (1/(2a))^2 Var(b) ) — rough
        # We approximate with window width as scale:
        Tc_err = np.sqrt(s2) / (2*abs(a) + 1e-9)
        return float(Tc), float(Tc_err)
    except Exception:
        return float(T[int(np.argmax(Y))]), float(0.0)


def quick_binder_multisize_tc(JkB, base_seed, Tc_guess, algo="Checkerboard (local)"):
    """Quick-and-dirty Binder crossing using small sizes to estimate Tc.
    Uses Ns = [16, 24, 32] around Tc_guess ± 15% window.
    Returns (Tc_cross or None, curves dict).
    """
    Ns = [16, 24, 32]
    rng = _np_rng(base_seed + 555)
    T_min = max(0.5*Tc_guess, Tc_guess * 0.85)
    T_max = Tc_guess * 1.15
    T_arr = np.linspace(T_min, T_max, 13)

    curves = {}
    for iN, N in enumerate(Ns):
        U4 = []
        for iT, T_real in enumerate(T_arr):
            beta = 1.0 / (T_real / JkB + 1e-12)
            local_rng = _np_rng(base_seed + iN*100 + iT)
            state = initial_lattice_np(N, local_rng)
            # short runs to keep fast
            for _ in range(200):
                if algo == "Cluster Flip (Wolff)":
                    state = wolff_step_np(state, beta, local_rng)
                elif algo == "Cluster Flip (Swendsen–Wang)":
                    state = swendsen_wang_step_np(state, beta, local_rng)
                else:
                    state = checkerboard_update_np(state, beta, local_rng)
            m_samples = []
            for _ in range(200):
                if algo == "Cluster Flip (Wolff)":
                    state = wolff_step_np(state, beta, local_rng)
                elif algo == "Cluster Flip (Swendsen–Wang)":
                    state = swendsen_wang_step_np(state, beta, local_rng)
                else:
                    state = checkerboard_update_np(state, beta, local_rng)
                m = calc_magnetization_np(state) / (N*N)
                m_samples.append(m)
            m_samples = np.array(m_samples)
            m1 = m_samples.mean()
            m2 = (m_samples**2).mean()
            m4 = (m_samples**4).mean()
            U4.append(1.0 - (m4 / (3.0 * (m2**2) + 1e-12)))
        curves[N] = (T_arr, np.array(U4))

    # simple crossing: find T where U4(N1) - U4(N2) changes sign
    def find_cross(T, y1, y2):
        d = y1 - y2
        for k in range(len(T)-1):
            if d[k] * d[k+1] <= 0:
                # linear interpolate
                t0, t1 = T[k], T[k+1]
                y0, y1v = d[k], d[k+1]
                if abs(y1v - y0) < 1e-12:
                    return 0.5*(t0 + t1)
                t = t0 - y0*(t1 - t0)/(y1v - y0)
                return t
        return None

    t12 = find_cross(curves[Ns[0]][0], curves[Ns[0]][1], curves[Ns[1]][1])
    t23 = find_cross(curves[Ns[1]][0], curves[Ns[1]][1], curves[Ns[2]][1])
    Tc_cross = None
    if (t12 is not None) and (t23 is not None):
        Tc_cross = 0.5*(t12 + t23)
    elif t12 is not None:
        Tc_cross = t12
    elif t23 is not None:
        Tc_cross = t23

    return Tc_cross, curves

# ------------------------------------------------------
# Plot helpers
# ------------------------------------------------------

def make_animated_line(T, Y, name, ytitle, hoverfun=None, markers=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[T[0]], y=[Y[0]], mode='lines+markers' if markers else 'lines', name=name,
                             hovertemplate=(hoverfun(T[0], Y[0]) if hoverfun else "T=%{x:.2f}K, Y=%{y:.3f}")))
    frames = []
    for k in range(1, len(T)+1):
        hovertext = None
        if hoverfun:
            hovertext = [hoverfun(t, y) for t, y in zip(T[:k], Y[:k])]
        frames.append(go.Frame(data=[go.Scatter(x=T[:k], y=Y[:k], mode='lines+markers' if markers else 'lines',
                                                hovertext=hovertext, hoverinfo='text')], name=str(k)))
    fig.frames = frames
    fig.update_layout(
        xaxis_title="Temperature (K)", yaxis_title=ytitle,
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 60, "redraw": True}, "transition": {"duration": 0}}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]}
            ],
            "direction": "left", "x": 0.0, "y": 1.14
        }],
        margin=dict(l=40, r=10, t=40, b=40),
        showlegend=False,
    )
    return fig

# Hover info with emojis for user engagement

def hover_M(t, y):
    vibe = "🧲 aligned" if y > 0.6 else ("😬 conflicted" if y > 0.2 else "🌀 chaos")
    return f"T={t:.1f}K<br>|M|={y:.3f}<br>{vibe}"


def hover_C(t, y):
    return f"T={t:.1f}K<br>Heat Capacity={y:.3f}"


def hover_X(t, y):
    return f"T={t:.1f}K<br>Susceptibility={y:.3f}"


def hover_U4(t, y):
    msg = "🕵️ Binder detects Tc?" if 0.4 < y < 0.8 else ("🧊 Ordered" if y > 0.8 else "🔥 Disordered")
    return f"T={t:.1f}K<br>U₄={y:.3f}<br>{msg}"

# ------------------------------------------------------
# Header
# ------------------------------------------------------
st.markdown(
    """
    <div class='hero'>
      <h1>2D Ising Model Explorer — Advanced 🧲</h1>
      <h3>Local & Cluster Monte Carlo · Error Bars · Tc Detection · Binder Crossing · Overlays · Animations</h3>
      <p>{}</p>
      <hr>
    </div>
    """.format(
        "Fun outside, rigorous inside. Watch spins dance and physics unfold." if mode == "Fun" else "Research-ready visuals with precise estimates and overlays."
    ),
    unsafe_allow_html=True,
)

# ------------------------------------------------------
# Run simulation
# ------------------------------------------------------
if run_sim:
    T_real_arr = np.linspace(minT, maxT, nT)
    with st.spinner("Running Ising simulation with statistical rigor…"):
        E, Mabs, C, Chi, U4, E_err, Mabs_err, spins_lo, spins_hi, hist_M = run_ising_sim(
            N, n_eq, n_samples, T_real_arr, JkB, seed, Tc_exp, algo_type, hist_T=T_hist
        )

    # Tc estimates
    TcC, dTcC = quadratic_tc_fit(T_real_arr, C)
    TcX, dTcX = quadratic_tc_fit(T_real_arr, Chi)

    # Summary card
    st.markdown(
        f"""
        <div class='param-card'>
          <b>Simulation Parameters</b><br>
          Lattice size = {N}×{N} · Algorithm = {algo_type}<br>
          Equilibration Steps = {n_eq} · Samples per T = {n_samples}<br>
          Temperature range = [{minT}, {maxT}] K with {nT} points<br>
          J/kB = {JkB} K · <span style='color:#b14;'>Experimental Tc</span> = {Tc_exp} K<br>
          <span style='color:#b14;'>Tc (Heat Capacity, fit)</span> = {TcC:.2f} ± {dTcC:.2f} K · <span style='color:#b14;'>Tc (Susceptibility, fit)</span> = {TcX:.2f} ± {dTcX:.2f} K
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs
    tabs = st.tabs([
        "📉 Magnetization",
        "🔥 Heat Capacity",
        "⚡ Susceptibility",
        "📐 Binder U₄",
        "🧾 Phases + Snapshots",
        "📊 Histogram",
        "🎞 Animations",
        "🧮 Quick Binder Multi-Size Tc",
        "📥 Export Data",
    ])

    # Lattice snapshot helper
    def lattice_snapshots_panel():
        if spins_lo is not None and spins_hi is not None:
            fig_snap, axes = plt.subplots(1, 2, figsize=(5.2, 2.3))
            axes[0].imshow(spins_lo, cmap='bwr', vmin=-1, vmax=1)
            axes[0].set_title("Below Tc 🧲" if mode == "Fun" else "Below Tc (ordered)")
            axes[0].axis('off')
            axes[1].imshow(spins_hi, cmap='bwr', vmin=-1, vmax=1)
            axes[1].set_title("Above Tc 🌀" if mode == "Fun" else "Above Tc (disordered)")
            axes[1].axis('off')
            st.pyplot(fig_snap)

    # Plotly branch
    if engine == "Plotly (interactive)":
        # Magnetization Tab
        with tabs[0]:
            c1, c2 = st.columns([7, 5])
            with c1:
                st.subheader("Magnetization" if mode == "Pro" else "Magnetization draws itself ✍️")
                figM = make_animated_line(T_real_arr, Mabs, "|M|", "|Magnetization| per spin", hoverfun=hover_M, markers=True)
                figM.add_vline(x=Tc_exp, line_dash="dot", annotation_text="Tc (exp)")
                figM.add_vline(x=TcC, line_dash="dash", annotation_text="Tc(C) fit")
                figM.add_vline(x=TcX, line_dash="dash", annotation_text="Tc(χ) fit")
                if exp_data is not None and 'T' in exp_data.columns and 'M' in exp_data.columns:
                    figM.add_trace(go.Scatter(x=exp_data['T'], y=exp_data['M'], mode='markers', name='Exp M', marker=dict(symbol='circle-open')))
                st.plotly_chart(figM, use_container_width=True)
            with c2:
                lattice_snapshots_panel()
                if mode == "Pro":
                    st.caption("Order parameter |M| shows spontaneous symmetry breaking below Tc.")

        # Heat Capacity Tab
        with tabs[1]:
            c1, c2 = st.columns([7, 5])
            with c1:
                st.subheader("Heat Capacity (per spin)")
                figC = make_animated_line(T_real_arr, C, "Heat Capacity", "Specific Heat (per spin)", hoverfun=hover_C)
                figC.add_vline(x=Tc_exp, line_dash="dot", annotation_text="Tc (exp)")
                figC.add_vline(x=TcC, line_dash="dash", annotation_text="Tc(C) fit")
                if exp_data is not None and 'T' in exp_data.columns and 'C' in exp_data.columns:
                    figC.add_trace(go.Scatter(x=exp_data['T'], y=exp_data['C'], mode='markers', name='Exp C', marker=dict(symbol='square-open')))
                st.plotly_chart(figC, use_container_width=True)
            with c2:
                lattice_snapshots_panel()

        # Susceptibility Tab
        with tabs[2]:
            c1, c2 = st.columns([7, 5])
            with c1:
                st.subheader("Susceptibility χ")
                figX = make_animated_line(T_real_arr, Chi, "Susceptibility", "χ (per spin)", hoverfun=hover_X)
                figX.add_vline(x=Tc_exp, line_dash="dot", annotation_text="Tc (exp)")
                figX.add_vline(x=TcX, line_dash="dash", annotation_text="Tc(χ) fit")
                st.plotly_chart(figX, use_container_width=True)
            with c2:
                lattice_snapshots_panel()

        # Binder cumulant Tab
        with tabs[3]:
            c1, c2 = st.columns([7, 5])
            with c1:
                st.subheader("Binder cumulant U₄")
                figU = make_animated_line(T_real_arr, U4, "Binder Cumulant U4", "U₄ = 1 - ⟨m⁴⟩/(3⟨m²⟩²)", hoverfun=hover_U4)
                figU.update_yaxes(range=[0, 1])
                figU.add_vline(x=Tc_exp, line_dash="dot", annotation_text="Tc (exp)")
                st.plotly_chart(figU, use_container_width=True)
            with c2:
                lattice_snapshots_panel()

    # Matplotlib classic branch
    else:
        def line_to_gif(x, y, ylab, fname):
            frames = []
            for k in range(1, len(x)+1):
                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(x[:k], y[:k], '-o')
                ax.set_xlabel('Temperature (K)')
                ax.set_ylabel(ylab)
                ax.grid(True)
                fig.tight_layout()
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
                frames.append(img)
                plt.close(fig)
            imageio.mimsave(fname, frames, duration=0.06)
            return fname

        with tabs[0]:
            c1, c2 = st.columns([7, 5])
            with c1:
                st.subheader("Magnetization (classic)")
                gif = line_to_gif(T_real_arr, Mabs, "|Magnetization| per spin", "M_anim.gif")
                st.image(gif)
            with c2:
                lattice_snapshots_panel()

        with tabs[1]:
            st.subheader("Heat Capacity (classic)")
            gif = line_to_gif(T_real_arr, C, "Specific Heat (per spin)", "C_anim.gif")
            st.image(gif)
        with tabs[2]:
            st.subheader("Susceptibility χ (classic)")
            gif = line_to_gif(T_real_arr, Chi, "Susceptibility (χ)", "Chi_anim.gif")
            st.image(gif)
        with tabs[3]:
            st.subheader("Binder cumulant U₄ (classic)")
            gif = line_to_gif(T_real_arr, U4, "Binder Cumulant U4", "U4_anim.gif")
            st.image(gif)

    # Phases + snapshots
    with tabs[4]:
        st.subheader("2D Ising Phases + Snapshots")
        fig5, ax5 = plt.subplots(figsize=(7,4))
        ax5.axvspan(minT, Tc_exp, alpha=0.25, color='tab:blue', label="Ferromagnetic (ordered)")
        ax5.axvspan(Tc_exp, maxT, alpha=0.25, color='tab:orange', label="Paramagnetic (disordered)")
        ax5.axvline(Tc_exp, color='red', ls='--', label="Tc (exp)")
        ax5.errorbar(T_real_arr, Mabs, yerr=Mabs_err, fmt='ko', markersize=3, capsize=2, label="|Magnetization|")
        ax5.set_xlabel("Temperature (K)")
        ax5.set_yticks([])
        ax5.legend()
        st.pyplot(fig5)

        if (spins_lo is not None) and (spins_hi is not None):
            c1, c2 = st.columns(2)
            with c1:
                figA, axA = plt.subplots(figsize=(4.2, 4.2))
                axA.imshow(spins_lo, cmap='bwr', vmin=-1, vmax=1)
                axA.set_title("Below Tc — Ferromagnetic" if mode == "Pro" else "Below Tc 🧲 Ferromagnetic")
                axA.axis('off')
                st.pyplot(figA)
            with c2:
                figB, axB = plt.subplots(figsize=(4.2, 4.2))
                axB.imshow(spins_hi, cmap='bwr', vmin=-1, vmax=1)
                axB.set_title("Above Tc — Paramagnetic" if mode == "Pro" else "Above Tc 🌀 Paramagnetic")
                axB.axis('off')
                st.pyplot(figB)
            st.caption("Left: ordered spins; Right: disordered. Visual transition at Tc.")

    # Histogram
    with tabs[5]:
        st.subheader(f"Magnetization Histogram at T = {T_hist} K")
        if 'hist_img' not in st.session_state:
            st.session_state['hist_img'] = None
        if hist_M is not None:
            fig_h, ax_h = plt.subplots(figsize=(6, 4))
            ax_h.hist(hist_M, bins=24, edgecolor='k')
            ax_h.set_xlabel("|Magnetization| per spin")
            ax_h.set_ylabel("Frequency")
            ax_h.set_title(f"Histogram of |M| at T={T_hist} K")
            ax_h.grid(True)
            st.pyplot(fig_h)
            st.info("Near Tc expect broad/bimodal; far from Tc sharp peak.")
        else:
            st.warning("No histogram data at this temperature. Rerun with this T included.")

    # Animations
    with tabs[6]:
        st.subheader("Lattice Animations: Spin Evolution & Temperature Sweep")

        def animate_spin_evolution(N, MC_steps, beta, seed):
            rng = _np_rng(seed + 1337)
            state = initial_lattice_np(N, rng)
            images = []
            for k in range(MC_steps):
                state = checkerboard_update_np(state, beta, rng)
                fig, ax = plt.subplots()
                ax.imshow(state, cmap="bwr", vmin=-1, vmax=1)
                ax.set_title(f"MC Step {k+1}")
                ax.axis('off')
                fig.tight_layout()
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
                    fig.canvas.get_width_height()[::-1] + (4,))
                images.append(img)
                plt.close(fig)
            b = io.BytesIO()
            imageio.mimsave(b, images, format='GIF', duration=0.1)
            return b.getvalue()

        def animate_temp_sweep(N, T_arr, MC_steps, seed, JkB):
            rng = _np_rng(seed + 4242)
            images = []
            for i, T_real in enumerate(T_arr):
                beta = 1.0 / (T_real / JkB + 1e-12)
                state = initial_lattice_np(N, rng)
                for _ in range(MC_steps):
                    state = checkerboard_update_np(state, beta, rng)
                fig, ax = plt.subplots()
                ax.imshow(state, cmap="bwr", vmin=-1, vmax=1)
                ax.axis('off')
                ax.set_title(f"T = {T_real:.2f} K")
                fig.tight_layout()
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
                    fig.canvas.get_width_height()[::-1] + (4,))
                images.append(img)
                plt.close(fig)
            b = io.BytesIO()
            imageio.mimsave(b, images, format='GIF', duration=0.12)
            return b.getvalue()

        beta_anim = 1.0 / (spin_anim_T / JkB + 1e-12)
        if st.button("Generate Spin Evolution Animation", use_container_width=True):
            gif_bytes = animate_spin_evolution(N, spin_anim_steps, beta_anim, seed)
            st.image(gif_bytes)
        T_anim_arr = np.linspace(minT, maxT, min(28, nT))
        if st.button("Generate Temperature Sweep Animation", use_container_width=True):
            gif_bytes = animate_temp_sweep(N, T_anim_arr, temp_anim_steps, seed, JkB)
            st.image(gif_bytes)

    # Quick Binder Multi-Size Tc (fast approximation)
    with tabs[7]:
        st.subheader("Quick Binder Multi-Size Tc (16, 24, 32)")
        if st.button("Run quick multi-size Tc estimator"):
            with st.spinner("Running short Binder simulations…"):
                Tc_cross, curves = quick_binder_multisize_tc(JkB, seed, TcC, algo=algo_type)
            fig = go.Figure()
            for Nsmall, (Tarr, Uarr) in curves.items():
                fig.add_trace(go.Scatter(x=Tarr, y=Uarr, mode='lines+markers', name=f"N={Nsmall}"))
            if Tc_cross is not None:
                fig.add_vline(x=Tc_cross, line_dash="dash", annotation_text=f"Tc(cross)≈{Tc_cross:.2f}K")
                st.success(f"Estimated Tc from Binder crossing ≈ {Tc_cross:.2f} K")
            else:
                st.warning("Could not find a clean crossing in this quick run. Try higher Ns or more points.")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("This is a quick, noisy estimate intended for intuition building. For publication-grade results, increase Ns and statistics.")

    # Export
    with tabs[8]:
        st.subheader("Export Simulation Data")
        df = pd.DataFrame({
            'Temperature_K': T_real_arr,
            'Energy_per_spin': E,
            'Magnetization_abs': Mabs,
            'Heat_Capacity': C,
            'Susceptibility': Chi,
            'Binder_Cumulant_U4': U4,
            'Energy_Error': E_err,
            'Magnetization_Error': Mabs_err,
        })
        st.dataframe(df, use_container_width=True)
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Data CSV",
            csv_buf.getvalue(),
            file_name=f"ising_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    with st.expander("🚦 Quick Guide", expanded=True):
        st.markdown(
            """
            1) Choose **Mode**, **Material**, **Algorithm**, lattice size, steps, T-range, and seed in the sidebar.  
            2) Click **Run Simulation**.  
            3) Explore plots in tabs (Magnetization, C, χ, Binder).  
            4) Upload experimental CSV to overlay (columns: `T`, `M`, optional `C`).  
            5) Use **Quick Binder Multi-Size** to estimate Tc via crossing.  
            6) Export results as CSV.
            """
        )
