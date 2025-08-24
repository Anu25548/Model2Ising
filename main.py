import streamlit as st
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import io
import os
from datetime import datetime
import random

# Interactive graphing
import plotly.graph_objects as go
M = np.array(magnetization_data)          # raw values
Mabs = np.abs(M)                          # absolute magnetization
m_abs_err = np.std(M) / np.sqrt(len(M))

# =============================
# --------- THEME / CSS -------
# =============================

st.set_page_config(page_title="2D Ising Model Explorer 🧲", page_icon="🧲", layout="wide")

st.markdown(
    """
    <style>
      .hero h1{ text-align:center; color:#1982c4; margin-bottom:2px; font-weight: 900;}
      .hero h3{ text-align:center; color:#54428E; margin-top:6px; font-weight: 600;}
      .hero p { text-align:center; color:#555; font-size:18px; font-style: italic; }
      .param-card { background:#f8f9fb; border:1px solid #e8ecf3; border-radius:16px; padding:16px; margin-top:8px; font-size:14px; }
      .explain { background:#fffaf2; border:1px solid #ffe2b6; border-radius:14px; padding:12px; font-size:14px; }
      .stTabs [data-baseweb="tab"]{font-weight:600}
      hr { border-color: #ccc; margin: 18px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# --------- HERO HEADER -------
# =============================

st.markdown(
    """
    <div class='hero'>
      <h1>2D Ising Model Explorer 🧲</h1>
      <h3>Monte Carlo · Cluster Updates · Error Bars · Histograms · Animations · Tc Detection · Export & Upload</h3>
      <p>Fun outside, rigorous inside. Watch the spins dance and the physics unfold.</p>
      <hr>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================
# --------- SIDEBAR UI --------
# =============================

MATERIAL_DB = {
    "iron": {"J_per_kB": 21.1, "Tc_exp": 1043},
    "k2cof4": {"J_per_kB": 10.0, "Tc_exp": 110},
    "rb2cof4": {"J_per_kB": 7.0, "Tc_exp": 100},
    "dypo4": {"J_per_kB": 2.5, "Tc_exp": 3.4},
}

st.sidebar.markdown("## ⚙️ Simulation Controls")

material = st.sidebar.selectbox("Material:", list(MATERIAL_DB.keys()), format_func=lambda x: x.upper())
params = MATERIAL_DB[material]
JkB, Tc_exp = params["J_per_kB"], params["Tc_exp"]
st.sidebar.info(f"**{material.upper()}** → J/kB = {JkB} K, Tc(exp) = {Tc_exp} K")

algo_type = st.sidebar.selectbox(
    "Update Algorithm",
    ["Checkerboard (JAX)", "Single Spin (Metropolis)", "Cluster Flip (Wolff, fast)"]
)

N = st.sidebar.slider("Lattice Size (N×N)", 10, 96, 40)
n_eq = st.sidebar.number_input("Equilibration Steps", 400, step=100, value=600)
n_samples = st.sidebar.number_input("Samples per Temperature", 200, step=100, value=300)
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
uploaded = st.sidebar.file_uploader("Upload experimental CSV (T[K], M)", type=['csv'])
if uploaded:
    try:
        exp_data = pd.read_csv(uploaded)
        st.sidebar.success("Experimental data loaded.")
    except Exception:
        st.sidebar.error("Error loading file. Please upload a valid CSV.")

run_sim = st.sidebar.button("🚀 Run Simulation", use_container_width=True)

# =============================
# ------ CORE SIMULATION FUNCTIONS -----
# =============================

def initial_lattice(N, key):
    return 2 * jax.random.randint(key, (N, N), 0, 2) - 1

@jax.jit
def checkerboard_update(spins, beta, key):
    N = spins.shape[0]
    for offset in [0, 1]:
        mask = jnp.fromfunction(lambda i, j: ((i + j) % 2 == offset), (N, N), dtype=jnp.int32).astype(bool)
        neighbors = (jnp.roll(spins, 1, axis=0) + jnp.roll(spins, -1, axis=0) +
                     jnp.roll(spins, 1, axis=1) + jnp.roll(spins, -1, axis=1))
        key, subkey = jax.random.split(key)
        rand_mat = jax.random.uniform(subkey, (N, N))
        deltaE = 2 * spins * neighbors
        flip = (deltaE < 0) | (rand_mat < jnp.exp(-beta * deltaE))
        spins = jnp.where(mask & flip, -spins, spins)
    return spins

@jax.jit
def metropolis_update(spins, beta, key):
    N = spins.shape[0]
    key_i, key_j, key_u = jax.random.split(key, 3)
    idx = jax.random.randint(key_i, (N*N,), 0, N)
    jdx = jax.random.randint(key_j, (N*N,), 0, N)
    u = jax.random.uniform(key_u, (N*N,))
    def body_fun(carry, x):
        spins, _ = carry
        i, j, r = x
        s = spins[i, j]
        nb = spins[(i+1)%N, j] + spins[(i-1)%N, j] + spins[i, (j+1)%N] + spins[i, (j-1)%N]
        dE = 2 * s * nb
        accept = (dE < 0) | (r < jnp.exp(-beta * dE))
        spins = spins.at[i, j].set(jnp.where(accept, -s, s))
        return spins, None
    spins, _ = jax.lax.scan(body_fun, spins, (idx, jdx, u))
    return spins

def wolff_step(spins, beta, seed):
    N = spins.shape[0]
    spins_np = np.array(spins)
    visited = np.zeros((N,N), dtype=bool)
    np.random.seed(seed)
    i, j = np.random.randint(0, N), np.random.randint(0, N)
    cluster_spin = spins_np[i, j]
    stack = [(i, j)]
    visited[i, j] = True
    p = 1 - np.exp(-2 * beta)
    while stack:
        ci, cj = stack.pop()
        for ni, nj in [((ci+1)%N, cj), ((ci-1)%N, cj), (ci, (cj+1)%N), (ci, (cj-1)%N)]:
            if not visited[ni, nj] and spins_np[ni, nj] == cluster_spin and np.random.rand() < p:
                visited[ni, nj] = True
                stack.append((ni, nj))
    spins_np[visited] *= -1
    return jnp.array(spins_np)

def calc_energy(state):
    return float(-jnp.sum(state * jnp.roll(state, 1, 0)) - jnp.sum(state * jnp.roll(state, 1, 1))) / 2.0

def calc_magnetization(state):
    return float(jnp.sum(state))

# =============================
# ------ SIMULATION CORE ------
# =============================

@st.cache_data(show_spinner=False)
def run_ising_sim(N, n_eq, n_samples, T_arr, JkB, seed, Tc_exp, maxT, algo, hist_T):
    E_av, m_abs_av, C_av, chi_av = [], [], [], []
    E_err, m_abs_err = [], []
    U4_av = []

    spins_below_tc, spins_above_tc = None, None
    hist_M = None

    key = jax.random.PRNGKey(seed)
    fun_jokes = [
        "Teaching spins how to dance… 🕺",
        "Summoning Tc from spin gods… 🔮",
        "Heating lattice noodles… 🍜",
        "Asking domains to behave… 🫡",
        "Phase transition drama loading… 🍿",
    ]
    progress = st.progress(0, text=random.choice(fun_jokes))

    for idx, T_real in enumerate(T_arr):
        T_code = T_real / JkB
        beta = 1.0 / T_code
        skey, key = jax.random.split(key)
        state = initial_lattice(N, skey)
        np.random.seed(seed)
        # Equilibration
        for _ in range(n_eq):
            skey, key = jax.random.split(key)
            if algo == "Cluster Flip (Wolff, fast)":
                state = wolff_step(state, beta, random.randint(0, 1_000_000))
            elif algo == "Single Spin (Metropolis)":
                state = metropolis_update(state, beta, skey)
            else:
                state = checkerboard_update(state, beta, skey)

        E_samples = []
        m_signed_samples = []
        m_abs_samples = []

        for _ in range(n_samples):
            skey, key = jax.random.split(key)
            if algo == "Cluster Flip (Wolff, fast)":
                state = wolff_step(state, beta, random.randint(0, 1_000_000))
            elif algo == "Single Spin (Metropolis)":
                state = metropolis_update(state, beta, skey)
            else:
                state = checkerboard_update(state, beta, skey)

            E = calc_energy(state)
            m = calc_magnetization(state)

            E_samples.append(E)
            m_signed_samples.append(m)
            m_abs_samples.append(abs(m))

        E_samples = np.array(E_samples) / (N*N)
        m_signed_samples = np.array(m_signed_samples) / (N*N)
        m_abs_samples = np.array(m_abs_samples) / (N*N)

        m1 = m_signed_samples.mean()
        m2 = (m_signed_samples**2).mean()
        m4 = (m_signed_samples**4).mean()

        C = E_samples.var(ddof=1) / (T_code**2)
        chi = (m2 - m1**2) / T_code
        U4 = 1.0 - (m4 / (3.0 * (m2**2) + 1e-12))

        E_av.append(E_samples.mean())
        m_abs_av.append(m_abs_samples.mean())
        C_av.append(C)
        chi_av.append(chi)
        U4_av.append(U4)

        E_err.append(E_samples.std(ddof=1) / np.sqrt(n_samples))
        m_abs_err.append(m_abs_samples.std(ddof=1) / np.sqrt(n_samples))

        if spins_below_tc is None and (T_real < Tc_exp):
            spins_below_tc = np.array(state)
        if spins_above_tc is None and (T_real > Tc_exp):
            spins_above_tc = np.array(state)
        if hist_T is not None and np.isclose(T_real, hist_T, atol=max(1e-6, (T_arr[1]-T_arr[0])/2)):
            hist_M = m_abs_samples.copy()

        if idx % max(1, len(T_arr)//5) == 0:
            progress.progress((idx+1)/len(T_arr), text=random.choice(fun_jokes))

    progress.progress(1.0, text="Phase transition drama at 100% 🍿")

    return (
        np.array(E_av), np.array(m_abs_av), np.array(C_av), np.array(chi_av),
        np.array(U4_av), np.array(E_err), np.array(m_abs_err),
        spins_below_tc, spins_above_tc, hist_M,
        np.array(m1), np.array(m2), np.array(m4)
    )

# =============================
# ---- ANIMATION HELPERS ----
# =============================

def animate_spin_evolution(N, MC_steps, beta, seed, filename="spin_evolution.gif"):
    key = jax.random.PRNGKey(seed+1337)
    state = initial_lattice(N, key)
    images = []
    for k in range(MC_steps):
        key, subkey = jax.random.split(key)
        state = checkerboard_update(state, beta, subkey)
        fig, ax = plt.subplots()
        ax.imshow(np.array(state), cmap="bwr", vmin=-1, vmax=1)
        ax.set_title(f"MC Step {k+1}")
        ax.axis('off')
        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))
        images.append(img)
        plt.close(fig)
    imageio.mimsave(filename, images, duration=0.1)
    return filename

def animate_temp_sweep(N, T_arr, MC_steps, seed, JkB, filename="ising_temp_sweep.gif"):
    key = jax.random.PRNGKey(seed+4242)
    images = []
    for i, T_real in enumerate(T_arr):
        T_code = T_real / JkB
        beta = 1.0 / T_code
        key, subkey = jax.random.split(key)
        state = initial_lattice(N, subkey)
        for _ in range(MC_steps):
            key, subkey = jax.random.split(key)
            state = checkerboard_update(state, beta, subkey)
        fig, ax = plt.subplots()
        ax.imshow(np.array(state), cmap="bwr", vmin=-1, vmax=1)
        ax.axis('off')
        ax.set_title(f"T = {T_real:.2f} K")
        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))
        images.append(img)
        plt.close(fig)
    imageio.mimsave(filename, images, duration=0.12)
    return filename

# =============================
# ----- PLOTLY ANIMATED LINES -----
# =============================

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
            "direction": "left", "x": 0.0, "y": 1.15
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
    msg = "🕵️ Binder cumulant detects Tc?" if 0.4 < y < 0.8 else ("🧊 Ordered" if y > 0.8 else "🔥 Disordered")
    return f"T={t:.1f}K<br>U₄={y:.3f}<br>{msg}"

# =============================
# -------- MAIN EXECUTION -------
# =============================

if run_sim:
    T_real_arr = np.linspace(minT, maxT, nT)

    with st.spinner("Running Ising simulation with statistical rigor and jokes..."):
        E, Mabs, C, Chi, U4, E_err, Mabs_err, spins_lo, spins_hi, hist_M, _, _, _ = run_ising_sim(
            N, n_eq, n_samples, T_real_arr, JkB, seed, Tc_exp, maxT, algo_type, hist_T=T_hist
        )

    Tc_from_C = T_real_arr[np.argmax(C)]
    Tc_from_Chi = T_real_arr[np.argmax(Chi)]

    st.markdown(
        f"""
        <div class='param-card'>
          <b>Simulation Parameters</b><br>
          Lattice size = {N}×{N} · Algorithm = {algo_type}<br>
          Equilibration Steps = {n_eq} · Samples per T = {n_samples}<br>
          Temperature range = [{minT}, {maxT}] K with {nT} points<br>
          J/kB = {JkB} K · <span style='color:#b14;'>Experimental Tc</span> = {Tc_exp} K · <span style='color:#b14;'>Tc (Heat Capacity peak)</span> = {Tc_from_C:.2f} K · <span style='color:#b14;'>Tc (Susceptibility peak)</span> = {Tc_from_Chi:.2f} K
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs organization
    tabs = st.tabs([
        "📉 Magnetization vs Temperature (Animated)",
        "🔥 Heat Capacity vs Temperature (Animated)",
        "⚡ Susceptibility vs Temperature (Animated)",
        "📐 Binder Cumulant U₄ (Animated)",
        "🧾 Phase + Snapshot Spins",
        "📊 Magnetization Histogram",
        "🎞 Lattice Animations",
        "📥 Export Data"
    ])

    # Interactive plots with Plotly or classic with Matplotlib
    def lattice_snapshots_panel():
        if spins_lo is not None and spins_hi is not None:
            fig_snap, axes = plt.subplots(1, 2, figsize=(5.2, 2.3))
            axes[0].imshow(spins_lo, cmap='bwr', vmin=-1, vmax=1)
            axes[0].set_title("Below Tc 🧲")
            axes[0].axis('off')
            axes[1].imshow(spins_hi, cmap='bwr', vmin=-1, vmax=1)
            axes[1].set_title("Above Tc 🌀")
            axes[1].axis('off')
            st.pyplot(fig_snap)

    # Plotly animated plots
    if engine == "Plotly (interactive)":
        # Magnetization Tab
        with tabs[0]:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Magnetization draws itself ✍️")
                figM = make_animated_line(T_real_arr, Mabs, "|M|", "|Magnetization| per spin", hoverfun=hover_M, markers=True)
                figM.add_vline(x=Tc_exp, line_dash="dot", annotation_text="Tc (exp)")
                st.plotly_chart(figM, use_container_width=True)
            with c2:
                lattice_snapshots_panel()
                st.caption("Spin-stagram: ordered vs disordered spin states ✨")

        # Heat capacity Tab
        with tabs[1]:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Heat capacity reveals the snack peak 🍿")
                figC = make_animated_line(T_real_arr, C, "Heat Capacity", "Specific Heat (per spin)", hoverfun=hover_C)
                figC.add_vline(x=Tc_exp, line_dash="dot", annotation_text="Tc (exp)")
                st.plotly_chart(figC, use_container_width=True)
            with c2:
                lattice_snapshots_panel()

        # Susceptibility Tab
        with tabs[2]:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Susceptibility being dramatic ⚡")
                figX = make_animated_line(T_real_arr, Chi, "Susceptibility", "χ (per spin)", hoverfun=hover_X)
                figX.add_vline(x=Tc_exp, line_dash="dot", annotation_text="Tc (exp)")
                st.plotly_chart(figX, use_container_width=True)
            with c2:
                lattice_snapshots_panel()

        # Binder cumulant Tab
        with tabs[3]:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Binder cumulant does the detective work 🕵️")
                figU = make_animated_line(T_real_arr, U4, "Binder Cumulant U4", "U₄ = 1 - ⟨m⁴⟩/(3⟨m²⟩²)", hoverfun=hover_U4)
                figU.update_yaxes(range=[0, 1])
                figU.add_vline(x=Tc_exp, line_dash="dot", annotation_text="Tc (exp)")
                st.plotly_chart(figU, use_container_width=True)
            with c2:
                lattice_snapshots_panel()

    # Matplotlib classic versions with GIFs fallback
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
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Magnetization draws itself ✍️ (classic)")
                gif = line_to_gif(T_real_arr, Mabs, "|Magnetization| per spin", "M_anim.gif")
                st.image(gif)
            with c2:
                lattice_snapshots_panel()

        with tabs[1]:
            gif = line_to_gif(T_real_arr, C, "Specific Heat (per spin)", "C_anim.gif")
            st.image(gif)
        with tabs[2]:
            gif = line_to_gif(T_real_arr, Chi, "Susceptibility (χ)", "Chi_anim.gif")
            st.image(gif)
        with tabs[3]:
            gif = line_to_gif(T_real_arr, U4, "Binder Cumulant U4", "U4_anim.gif")
            st.image(gif)

    # Phase diagram + snapshots tab
    with tabs[4]:
        st.subheader("2D Ising Phase + Spin Snapshots")
        fig5, ax5 = plt.subplots(figsize=(7,4))
        ax5.axvspan(minT, Tc_exp, alpha=0.25, color='tab:blue', label="Ferromagnetic (ordered)")
        ax5.axvspan(Tc_exp, maxT, alpha=0.25, color='tab:orange', label="Paramagnetic (disordered)")
        ax5.axvline(Tc_exp, color='red', ls='--', label="Tc (exp)")
        ax5.errorbar(T_real_arr, Mabs, yerr=m_abs_err, fmt='ko', markersize=3, capsize=2, label="|Magnetization|")
        ax5.set_xlabel("Temperature (K)")
        ax5.set_yticks([])
        ax5.legend()
        st.pyplot(fig5)

        if (spins_lo is not None) and (spins_hi is not None):
            c1, c2 = st.columns(2)
            with c1:
                figA, axA = plt.subplots(figsize=(4.2, 4.2))
                axA.imshow(spins_lo, cmap='bwr', vmin=-1, vmax=1)
                axA.set_title("Below Tc 🧲 Ferromagnetic")
                axA.axis('off')
                st.pyplot(figA)
            with c2:
                figB, axB = plt.subplots(figsize=(4.2, 4.2))
                axB.imshow(spins_hi, cmap='bwr', vmin=-1, vmax=1)
                axB.set_title("Above Tc 🌀 Paramagnetic")
                axB.axis('off')
                st.pyplot(figB)
            st.caption("Left: ordered spins; Right: disordered. Visual transition at Tc.")

        st.markdown(
            """
            ---
            #### 🌈 Explanation:
            - Blue region = ordered ferromagnetic phase (spins aligned).
            - Orange region = disordered paramagnetic phase (spins random).
            - Red dashed line = critical temperature Tc where phase transition occurs.
            - Spin snapshots visualize spin ordering across the transition.
            """
        )

    # Histogram tab
    with tabs[5]:
        st.subheader(f"Magnetization Histogram at T = {T_hist} K")
        if hist_M is not None:
            fig_h, ax_h = plt.subplots(figsize=(6, 4))
            ax_h.hist(hist_M, bins=24, edgecolor='k', color="#2a9d8f")
            ax_h.set_xlabel("|Magnetization| per spin")
            ax_h.set_ylabel("Frequency")
            ax_h.set_title(f"Histogram of |M| at T={T_hist} K")
            ax_h.grid(True)
            st.pyplot(fig_h)
            st.info(
                "Near Tc expect broad or bimodal histogram (fluctuating order). Far from Tc observe single sharp peak (stable order/disorder)."
            )
        else:
            st.warning("No histogram data available for this temperature. Rerun simulation with the selected T.")

        st.markdown(
            """
            ---
            #### 📊 Histogram Insights:
            - Shows distribution of magnetization values sampled during MC at one T.
            - Sharp peak near 1 when T << Tc (strong order).
            - Bimodal or broad around Tc (critical fluctuations).
            - Peak near 0 when T >> Tc (disorder).
            """
        )

    # Animation tab
    with tabs[6]:
        st.subheader("Lattice Animations: Spin Evolution & Temperature Sweep")

        st.markdown(f"**Spin Evolution at T={spin_anim_T} K for {spin_anim_steps} MC steps**")
        beta_anim = 1.0 / (spin_anim_T / JkB)
        gif1 = "spin_evolution.gif"
        if (not os.path.exists(gif1)) or st.button("Regenerate Spin Evolution Animation"):
            with st.spinner("Animating spins practicing their steps..."):
                animate_spin_evolution(N, spin_anim_steps, beta_anim, seed, gif1)
                st.image(gif1, caption="Spin lattice evolution over MC steps")

        st.markdown(f"**Temperature Sweep Animation over {temp_anim_steps} MC steps**")
        gif2 = "ising_temp_sweep.gif"
        T_anim_arr = np.linspace(minT, maxT, min(28, nT))
        if (not os.path.exists(gif2)) or st.button("Regenerate Temp Sweep Animation"):
            with st.spinner("Heating lattice, disorder melting in..."):
                animate_temp_sweep(N, T_anim_arr, temp_anim_steps, seed, JkB, gif2)
        st.image(gif2, caption="Spin order melting across temperature sweep")

    # Export tab
    with tabs[7]:
        st.subheader("Export Simulation Data")

        df = pd.DataFrame({
            'Temperature_K': T_real_arr,
            'Energy_per_spin': E,
            'Magnetization_abs': Mabs,
            'Heat_Capacity': C,
            'Susceptibility': Chi,
            'Binder_Cumulant_U4': U4,
            'Energy_Error': E_err,
            'Magnetization_Error': m_abs_err,
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
    with st.expander("🚦 Quick Guide to Use This Explorer", expanded=True):
        st.markdown(
            """
            1) Set **Material**, **Algorithm**, lattice size, equilibration, samples, temperature range, and seed from the sidebar.<br>
            2) Click **Run Simulation**.<br>
            3) Explore plots and animations in tabs.<br>
            4) Read explanations accompanying each visualization.<br>
            5) Upload your experimental data CSV to compare.<br>
            6) Export simulation data for further analysis.<br>
            <br>
            This interactive tool visualizes the famous 2D Ising model's magnetic phase transitions with Monte Carlo methods.
            """
        )
