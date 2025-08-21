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

# NEW: interactive graphs
import plotly.graph_objects as go
import plotly.express as px

# =============================
# --------- THEME / CSS -------
# =============================
st.set_page_config(page_title="2D Ising Model – Fun & Physics", page_icon="🧲", layout="wide")

st.markdown(
    """
    <style>
      .hero h1{ text-align:center; color:#1982c4; margin-bottom:2px }
      .hero h3{ text-align:center; color:#54428E; margin-top:6px }
      .hero p { text-align:center; color:#555; font-size:18px }
      .param-card{background:#f8f9fb;border:1px solid #e8ecf3;border-radius:16px;padding:16px;margin-top:8px}
      .explain{background:#fffaf2;border:1px solid #ffe2b6;border-radius:14px;padding:12px}
      .stTabs [data-baseweb="tab"]{font-weight:600}
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
      <h3>Monte Carlo · Cluster Updates · Error Bars · Histograms · Animation · χ(T) & U₄(T) — now with jokes 😄</h3>
      <p><i>Fun outside, rigorous inside. Let the graphs draw themselves while the spins dance.</i></p>
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
N        = st.sidebar.slider("Lattice Size (N×N)", 10, 96, 40)
n_eq     = st.sidebar.number_input("Equilibration Steps", 400, step=100, value=600)
n_samples= st.sidebar.number_input("Samples per T", 200, step=100, value=300)
seed     = st.sidebar.number_input("Random Seed", 0, step=1, value=0)
minT     = st.sidebar.number_input("Low Temp (K)", int(Tc_exp * 0.6))
maxT     = st.sidebar.number_input("High Temp (K)", int(Tc_exp * 1.4))
nT       = st.sidebar.slider("Number of Temperatures", 10, 60, 30)

st.sidebar.markdown("## 📊 Analysis Options")
T_hist   = st.sidebar.slider("Histogram: T (K)", minT, maxT, Tc_exp, step=1)

st.sidebar.markdown("## 🎥 Animation")
spin_anim_T = st.sidebar.slider("Spin Evolution: T (K)", minT, maxT, Tc_exp, step=1)
spin_anim_steps = st.sidebar.slider("Spin Animation Steps", 30, 150, 60)
temp_anim_steps = st.sidebar.slider("Temp Sweep MC Steps", 80, 600, 280)

engine = st.sidebar.radio("Plot Engine", ["Plotly (interactive)", "Matplotlib (classic)"])

exp_data = None
uploaded = st.sidebar.file_uploader("Upload experimental CSV (T[K],M)", type=['csv'])
if uploaded:
    exp_data = pd.read_csv(uploaded)
    st.sidebar.success("Experimental file loaded.")

run_sim  = st.sidebar.button("🚀 Run Simulation", use_container_width=True)

# =============================
# ------ CORE SIM HELPERS -----
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
    u   = jax.random.uniform(key_u, (N*N,))
    def body_fun(carry, x):
        spins, k = carry
        i, j, r = x
        s = spins[i, j]
        nb = spins[(i+1)%N, j] + spins[(i-1)%N, j] + spins[i, (j+1)%N] + spins[i, (j-1)%N]
        dE = 2 * s * nb
        accept = (dE < 0) | (r < jnp.exp(-beta * dE))
        spins = spins.at[i, j].set(jnp.where(accept, -s, s))
        return (spins, k), None
    (spins, _), _ = jax.lax.scan(body_fun, (spins, key), (idx, jdx, u))
    return spins

# Wolff (numpy-based)
def wolff_step(spins, beta, key):
    N = spins.shape[0]
    spins_np = np.array(spins)
    visited = np.zeros((N,N), dtype=bool)
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

# Observables

def calc_energy(state):
    return float(-jnp.sum(state * jnp.roll(state, 1, 0)) - jnp.sum(state * jnp.roll(state, 1, 1))) / 2.0

def calc_magnetization(state):
    return float(jnp.sum(state))

# =============================
# ------ MAIN SIMULATION ------
# =============================

@st.cache_data(show_spinner=False)
def run_ising_sim(N, n_eq, n_samples, T_arr, JkB, seed, Tc_exp, maxT, algo, hist_T):
    E_av, m_abs_av, C_av, chi_av = [], [], [], []
    E_err, m_abs_err = [], []
    U4_av = []
    m1_list, m2_list, m4_list = [], [], []

    spins_below_tc, spins_above_tc = None, None
    hist_M = None

    key = jax.random.PRNGKey(int(seed))
    funny = [
        "Teaching spins how to dance… 🕺",
        "Summoning Tc from spin gods… 🔮",
        "Heating lattice noodles… 🍜",
        "Asking domains to behave… 🫡",
        "Phase transition drama loading… 🍿",
    ]
    progress = st.progress(0, text=random.choice(funny))

    for idx, T_real in enumerate(T_arr):
        T_code = T_real / JkB
        beta = 1.0 / T_code
        skey, key = jax.random.split(key)
        state = initial_lattice(N, skey)
        np.random.seed(seed)
        for _ in range(n_eq):
            skey, key = jax.random.split(key)
            if algo == "Cluster Flip (Wolff, fast)":
                state = wolff_step(state, beta, skey)
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
                state = wolff_step(state, beta, skey)
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
        m1 = m_signed_samples.mean(); m2 = (m_signed_samples**2).mean(); m4 = (m_signed_samples**4).mean()
        C = E_samples.var(ddof=1)/(T_code**2)
        chi = (m2 - m1**2) / T_code
        U4  = 1.0 - (m4 / (3.0 * (m2**2) + 1e-12))
        E_av.append(E_samples.mean()); m_abs_av.append(m_abs_samples.mean()); C_av.append(C); chi_av.append(chi); U4_av.append(U4)
        E_err.append(E_samples.std(ddof=1)/np.sqrt(n_samples)); m_abs_err.append(m_abs_samples.std(ddof=1)/np.sqrt(n_samples))
        m1_list.append(m1); m2_list.append(m2); m4_list.append(m4)
        if spins_below_tc is None and (T_real < Tc_exp):
            spins_below_tc = np.array(state)
        if spins_above_tc is None and (T_real > Tc_exp):
            spins_above_tc = np.array(state)
        if hist_T is not None and np.isclose(T_real, hist_T, atol=max(1e-6, (T_arr[1]-T_arr[0])/2)):
            hist_M = m_abs_samples.copy()
        if idx % max(1, len(T_arr)//5) == 0:
            progress.progress((idx+1)/len(T_arr), text=random.choice(spin_msgs))
    progress.update(100, text="Phase transition drama at 100% 🔥")

    return (
        np.array(E_av), np.array(m_abs_av), np.array(C_av), np.array(chi_av),
        np.array(U4_av), np.array(E_err), np.array(m_abs_err),
        spins_below_tc, spins_above_tc, hist_M,
        np.array(m1_list), np.array(m2_list), np.array(m4_list)
    )

# =============================
# ---- ANIMS: LATTICE GIFS ----
# =============================

def animate_spin_evolution(N, MC_steps, beta, seed, filename="spin_evolution.gif"):
    key = jax.random.PRNGKey(int(seed)+1337)
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
    key = jax.random.PRNGKey(int(seed)+4242)
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
# ----- PLOTLY ANIM HELPERS ---
# =============================

def make_animated_line(T, Y, name, ytitle, hoverfun=None, markers=False):
    # Base figure
    fig = go.Figure()
    # initial empty trace
    fig.add_trace(go.Scatter(x=[T[0]], y=[Y[0]], mode='lines+markers' if markers else 'lines', name=name,
                             hovertemplate=(hoverfun(T[0], Y[0]) if hoverfun else "T=%{x:.2f}K, Y=%{y:.3f}")))
    # Frames that progressively reveal points
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
            "type":"buttons",
            "buttons":[
                {"label":"▶ Play","method":"animate","args":[None, {"fromcurrent":True, "frame":{"duration":60, "redraw":True}, "transition":{"duration":0}}]},
                {"label":"⏸ Pause","method":"animate","args":[[None], {"mode":"immediate", "frame":{"duration":0, "redraw":False}, "transition":{"duration":0}}]}
            ],
            "direction":"left","x":0.0,"y":1.15
        }],
        margin=dict(l=40,r=10,t=40,b=40),
        showlegend=False,
    )
    return fig

# Cute hover strings with emojis

def hover_M(t, y):
    vibe = "🧲 aligned" if y>0.6 else ("😬 conflicted" if y>0.2 else "🌀 chaos")
    return f"T={t:.1f}K<br>|M|={y:.3f}<br>{vibe}"

def hover_C(t, y):
    spice = "🔥 snack time" if y==y else ""
    return f"T={t:.1f}K<br>C={y:.3f}<br>{spice}"

def hover_X(t, y):
    drama = "⚡ drama queen" if y>np.nanmax([y*0+0.0]) else ""
    return f"T={t:.1f}K<br>χ={y:.3f}<br>{drama}"

def hover_U4(t, y):
    msg = "🕵️ Binder says: Tc?" if 0.4<y<0.8 else ("🧊 ordered vibes" if y>0.8 else "🔥 messy vibes")
    return f"T={t:.1f}K<br>U₄={y:.3f}<br>{msg}"

# =============================
# ---------- RUN --------------
# =============================

if run_sim:
    T_real_arr = np.linspace(minT, maxT, nT)

    with st.spinner("Running Ising simulation (with animated graphs & jokes)…"):
        E, Mabs, C, Chi, U4, E_err, Mabs_err, spins_lo, spins_hi, hist_M, m1, m2, m4 = run_ising_sim(
            N, n_eq, n_samples, T_real_arr, JkB, seed, Tc_exp, maxT, algo_type, hist_T=T_hist
        )

    Tc_from_C   = T_real_arr[np.argmax(C)]
    Tc_from_Chi = T_real_arr[np.argmax(Chi)]

    st.markdown(
        f"""
        <div class='param-card'>
          <b>Parameters</b><br>
          N = {N} · Algo = {algo_type} · Equil = {n_eq} · Samples = {n_samples}<br>
          T ∈ [{minT}, {maxT}] K with {nT} points · J/kB = {JkB} K<br>
          <span style='color:#b14;'>Tc(exp)</span> = {Tc_exp} K · <span style='color:#b14;'>Tc(C-peak)</span> = {Tc_from_C:.2f} K · <span style='color:#b14;'>Tc(χ-peak)</span> = {Tc_from_Chi:.2f} K
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "📉 M vs T (animated)",
        "🔥 C vs T (animated)",
        "⚡ χ vs T (animated)",
        "📐 U₄ vs T (animated)",
        "🧾 Phase + Snapshots",
        "📊 Histogram",
        "🎞 Lattice Animations",
        "📥 Export"
    ])

    # ---------- Animated PLOTLY section ----------
    if engine == "Plotly (interactive)":
        # Side-by-side layout for each plot with a small lattice snapshot
        def lattice_panel():
            if spins_lo is not None and spins_hi is not None:
                fig_snap, axes = plt.subplots(1,2, figsize=(5.2,2.3))
                axes[0].imshow(spins_lo, cmap='bwr', vmin=-1, vmax=1); axes[0].set_title("Below Tc 🧲"); axes[0].axis('off')
                axes[1].imshow(spins_hi, cmap='bwr', vmin=-1, vmax=1); axes[1].set_title("Above Tc 🌀"); axes[1].axis('off')
                st.pyplot(fig_snap)

        with tabs[0]:
            c1, c2 = st.columns([1,1])
            with c1:
                st.subheader("Magnetization draws itself ✍️")
                figM = make_animated_line(T_real_arr, Mabs, "|M|", "|Magnetization| per spin", hoverfun=hover_M, markers=True)
                figM.add_vline(x=Tc_exp, line_dash="dot", annotation_text="Tc (exp)")
                st.plotly_chart(figM, use_container_width=True)
            with c2:
                lattice_panel()
                st.caption("Spin-stagram: ordered vs disordered ✨")

        with tabs[1]:
            c1, c2 = st.columns([1,1])
            with c1:
                st.subheader("Heat capacity reveals the snack peak 🍿")
                figC = make_animated_line(T_real_arr, C, "C", "Specific Heat (per spin)", hoverfun=hover_C)
                figC.add_vline(x=Tc_exp, line_dash="dot", annotation_text="Tc (exp)")
                st.plotly_chart(figC, use_container_width=True)
            with c2:
                lattice_panel()

        with tabs[2]:
            c1, c2 = st.columns([1,1])
            with c1:
                st.subheader("Susceptibility being dramatic ⚡")
                figX = make_animated_line(T_real_arr, Chi, "chi", "χ (per spin)", hoverfun=hover_X)
                figX.add_vline(x=Tc_exp, line_dash="dot", annotation_text="Tc (exp)")
                st.plotly_chart(figX, use_container_width=True)
            with c2:
                lattice_panel()

        with tabs[3]:
            c1, c2 = st.columns([1,1])
            with c1:
                st.subheader("Binder cumulant does the detective work 🕵️")
                figU = make_animated_line(T_real_arr, U4, "U4", "U₄ = 1 - ⟨m⁴⟩/(3⟨m²⟩²)", hoverfun=hover_U4)
                figU.update_yaxes(range=[0,1])
                figU.add_vline(x=Tc_exp, line_dash="dot", annotation_text="Tc (exp)")
                st.plotly_chart(figU, use_container_width=True)
            with c2:
                lattice_panel()

    # ---------- Matplotlib (classic) animated GIFs ----------
    else:
        def line_to_gif(x, y, ylab, fname):
            frames = []
            for k in range(1, len(x)+1):
                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(x[:k], y[:k], '-o')
                ax.set_xlabel('Temperature (K)'); ax.set_ylabel(ylab); ax.grid(True)
                fig.tight_layout(); fig.canvas.draw()
                img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
                frames.append(img)
                plt.close(fig)
            imageio.mimsave(fname, frames, duration=0.06)
            return fname
        with tabs[0]:
            c1, c2 = st.columns([1,1])
            with c1:
                st.subheader("Magnetization draws itself ✍️ (classic)")
                gif = line_to_gif(T_real_arr, Mabs, "|Magnetization| per spin", "M_anim.gif")
                st.image(gif)
            with c2:
                if spins_lo is not None and spins_hi is not None:
                    fig_snap, axes = plt.subplots(1,2, figsize=(5.2,2.3))
                    axes[0].imshow(spins_lo, cmap='bwr', vmin=-1, vmax=1); axes[0].set_title("Below Tc 🧲"); axes[0].axis('off')
                    axes[1].imshow(spins_hi, cmap='bwr', vmin=-1, vmax=1); axes[1].set_title("Above Tc 🌀"); axes[1].axis('off')
                    st.pyplot(fig_snap)
        with tabs[1]:
            gif = line_to_gif(T_real_arr, C, "Specific Heat (per spin)", "C_anim.gif")
            st.image(gif)
        with tabs[2]:
            gif = line_to_gif(T_real_arr, Chi, "χ (per spin)", "Chi_anim.gif")
            st.image(gif)
        with tabs[3]:
            gif = line_to_gif(T_real_arr, U4, "U₄", "U4_anim.gif")
            st.image(gif)

    # -------- Phase + Histogram + Lattice --------
    with tabs[4]:
        st.subheader("2D Ising Phase + Snapshots")
        fig5, ax5 = plt.subplots(figsize=(7,4))
        ax5.axvspan(minT, Tc_exp, alpha=0.25, color='tab:blue', label="Ferromagnetic")
        ax5.axvspan(Tc_exp, maxT, alpha=0.25, color='tab:orange', label="Paramagnetic")
        ax5.axvline(Tc_exp, color='red', ls='--', label="Tc")
        ax5.errorbar(T_real_arr, Mabs, yerr=Mabs_err, fmt='ko', markersize=3, capsize=2, label="|M|")
        ax5.set_xlabel("Temperature (K)"); ax5.set_yticks([])
        ax5.legend(); st.pyplot(fig5)
        if (spins_lo is not None) and (spins_hi is not None):
            c1, c2 = st.columns(2)
            with c1:
                      figA, axA = plt.subplots(figsize=(4.2,4.2)); axA.imshow(spins_lo, cmap='bwr', vmin=-1, vmax=1); axA.axis('off'); axA.set_title("Below Tc 🧲"); st.pyplot(figA)
            with c2:
                figB, axB = plt.subplots(figsize=(4.2,4.2)); axB.imshow(spins_hi, cmap='bwr', vmin=-1, vmax=1); axB.axis('off'); axB.set_title("Above Tc 🌀"); st.pyplot(figB)

    with tabs[5]:
        st.subheader(f"Magnetization Histogram at T = {T_hist} K")
        if hist_M is not None:
            fig_h, ax_h = plt.subplots(figsize=(6,4))
            ax_h.hist(hist_M, bins=24, edgecolor='k')
            ax_h.set_xlabel("|M| per spin"); ax_h.set_ylabel("Frequency")
            ax_h.grid(True)
            st.pyplot(fig_h)
            st.info("Near Tc expect broad/bimodal behavior. Far from Tc → single sharp peak.")
        else:
            st.warning("No histogram captured — choose T and re-run.")

    with tabs[6]:
        st.subheader("Lattice Animations: double impact 🎞 + 📈")
        st.markdown(f"**Spin Evolution at T={spin_anim_T} K, Steps={spin_anim_steps}**")
        beta_anim = 1.0 / (spin_anim_T / JkB)
        gif1 = "spin_evolution.gif"
        if (not os.path.exists(gif1)) or st.button("Regenerate Spin Evolution Animation"):
            with st.spinner("Making spins practice their steps…"):
                animate_spin_evolution(N, spin_anim_steps, beta_anim, seed, gif1)
        st.image(gif1, caption="Spin lattice evolution at fixed T")

        st.markdown(f"**Temperature Sweep Animation (steps={temp_anim_steps})**")
        gif2 = "ising_temp_sweep.gif"
        T_anim_arr = np.linspace(minT, maxT, min(28, nT))
        if (not os.path.exists(gif2)) or st.button("Regenerate T Sweep Animation"):
            with st.spinner("Heating, heating… chaos arriving…"):
                animate_temp_sweep(N, T_anim_arr, temp_anim_steps, seed, JkB, gif2)
        st.image(gif2, caption="Order melts as T crosses Tc")

    with tabs[7]:
        st.subheader("Export")
        df = pd.DataFrame({
            'T_K': T_real_arr,
            'E_per_spin': E,
            'M_abs': Mabs,
            'C': C,
            'Chi': Chi,
            'Binder_U4': U4,
            'E_err': E_err,
            'M_abs_err': Mabs_err,
            'm1': m1,
            'm2': m2,
            'm4': m4,
        })
        st.dataframe(df, use_container_width=True)
        csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Steal the data (CSV)", csv_buf.getvalue(), file_name=f"ising_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

else:
    with st.expander("🚦 Quick Guide", expanded=True):
        st.markdown(
            """
            1) Choose **material & algorithm**. 2) Set lattice size, equilibration, samples, and T-range. 3) Click **Run Simulation**.
            All plots **animate while being drawn** (Plotly or classic). Left: spins; Right: graph — **double impact**!
            """
        )
        
