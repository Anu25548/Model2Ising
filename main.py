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

# =============================
# --------- THEME / CSS -------
# =============================
st.set_page_config(page_title="2D Ising Model Explorer", page_icon="🧲", layout="wide")

st.markdown(
    """
    <style>
      /* Center hero title */
      .hero h1{ text-align:center; color:#1982c4; margin-bottom:0 }
      .hero h3{ text-align:center; color:#54428E; margin-top:6px }
      .hero p { text-align:center; color:#555; font-size:18px }
      /* Card look for parameter summary */
      .param-card{background:#f8f9fb;border:1px solid #e8ecf3;border-radius:16px;padding:16px;margin-top:8px}
      .explain{background:#fffaf2;border:1px solid #ffe2b6;border-radius:14px;padding:12px}
      /* Tweak tabs to have emojis aligned */
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
      <h3>Monte Carlo · Cluster Updates · Error Bars · Histograms · Animation · <b>Susceptibility & Binder</b></h3>
      <p><i>Toggle algorithms, visualize transitions, analyze fluctuations, and animate order — research-ready!</i></p>
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
    # Single-spin random updates
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

# Wolff step uses numpy for cluster growth (fast enough for teaching; could be jitted with more work)
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
    # Per configuration energy (not per spin yet)
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
    m1_list, m2_list, m4_list = [], [], []  # for diagnostics

    spins_below_tc, spins_above_tc = None, None
    hist_M = None

    key = jax.random.PRNGKey(int(seed))

    # progress bar
    progress = st.progress(0)

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
                state = wolff_step(state, beta, skey)
            elif algo == "Single Spin (Metropolis)":
                state = metropolis_update(state, beta, skey)
            else:
                state = checkerboard_update(state, beta, skey)

        # Sampling
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

        E_samples = np.array(E_samples) / (N*N)  # per spin
        m_signed_samples = np.array(m_signed_samples) / (N*N)
        m_abs_samples = np.array(m_abs_samples) / (N*N)

        # Moments
        m1 = m_signed_samples.mean()
        m2 = (m_signed_samples**2).mean()
        m4 = (m_signed_samples**4).mean()

        # Thermodynamic quantities
        C = E_samples.var(ddof=1)/(T_code**2)
        chi = (m2 - m1**2) / T_code  # susceptibility per spin (code units)
        U4  = 1.0 - (m4 / (3.0 * (m2**2) + 1e-12))

        # Averages & errors
        E_av.append(E_samples.mean())
        m_abs_av.append(m_abs_samples.mean())
        C_av.append(C)
        chi_av.append(chi)
        U4_av.append(U4)

        E_err.append(E_samples.std(ddof=1)/np.sqrt(n_samples))
        m_abs_err.append(m_abs_samples.std(ddof=1)/np.sqrt(n_samples))

        m1_list.append(m1); m2_list.append(m2); m4_list.append(m4)

        # Snapshots for phase panel
        if spins_below_tc is None and (T_real < Tc_exp):
            spins_below_tc = np.array(state)
        if spins_above_tc is None and (T_real > Tc_exp):
            spins_above_tc = np.array(state)

        # Histogram capture
        if hist_T is not None and np.isclose(T_real, hist_T, atol=max(1e-6, (T_arr[1]-T_arr[0])/2)):
            hist_M = m_abs_samples.copy()

        progress.progress(int((idx+1)/len(T_arr)*100))

    return (
        np.array(E_av), np.array(m_abs_av), np.array(C_av), np.array(chi_av),
        np.array(U4_av), np.array(E_err), np.array(m_abs_err),
        spins_below_tc, spins_above_tc, hist_M,
        np.array(m1_list), np.array(m2_list), np.array(m4_list)
    )

# =============================
# ---------- ANIMS ------------
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
# ---------- RUN --------------
# =============================

if run_sim:
    T_real_arr = np.linspace(minT, maxT, nT)

    with st.spinner("Running Ising simulation (with susceptibility & Binder)..."):
        E, Mabs, C, Chi, U4, E_err, Mabs_err, spins_lo, spins_hi, hist_M, m1, m2, m4 = run_ising_sim(
            N, n_eq, n_samples, T_real_arr, JkB, seed, Tc_exp, maxT, algo_type, hist_T=T_hist
        )

    # Estimate Tc from peaks
    Tc_from_C   = T_real_arr[np.argmax(C)]
    Tc_from_Chi = T_real_arr[np.argmax(Chi)]

    # Parameter summary card
    st.markdown(
        f"""
        <div class='param-card'>
          <b>Parameters</b><br>
          N = {N} &nbsp;·&nbsp; Algo = {algo_type} &nbsp;·&nbsp; Equil = {n_eq} &nbsp;·&nbsp; Samples = {n_samples}<br>
          T ∈ [{minT}, {maxT}] K with {nT} points &nbsp;·&nbsp; J/kB = {JkB} K<br>
          <span style='color:#b14;'>Tc(exp)</span> = {Tc_exp} K &nbsp;·&nbsp; <span style='color:#b14;'>Tc(C-peak)</span> = {Tc_from_C:.2f} K &nbsp;·&nbsp; <span style='color:#b14;'>Tc(χ-peak)</span> = {Tc_from_Chi:.2f} K
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prepare a tidy dataframe for export
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

    tabs = st.tabs([
        "📉 Magnetization (M vs T)",
        "🔥 Heat Capacity (C vs T)",
        "🧲 Susceptibility χ(T) & Binder U4(T)",
        "🧾 Phase Diagram + Snapshots",
        "📊 Histogram",
        "🎞 Animations",
        "📥 Export & Comparison"
    ])

    # --------------- Magnetization ---------------
    with tabs[0]:
        st.subheader("Magnetization vs Temperature")
        fig, ax = plt.subplots(figsize=(7,4))
        ax.errorbar(T_real_arr, Mabs, yerr=Mabs_err, fmt='o-', capsize=4, label=f"Simulation ({algo_type})")
        if exp_data is not None:
            ax.plot(exp_data.iloc[:,0], exp_data.iloc[:,1], 's--', label="Experiment")
        ax.axvline(Tc_exp, color='red', ls=':', label=f"Tc (Exp)={Tc_exp}K", lw=2)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("|Magnetization| per spin")
        ax.legend(); ax.grid(True)
        st.pyplot(fig)

        st.markdown(
            """
            <div class='explain'>
            <b>Interpretation.</b> Magnetization collapses near <i>Tc</i>. Error bars grow close to Tc due to critical fluctuations.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------- Heat Capacity ---------------
    with tabs[1]:
        st.subheader("Heat Capacity vs Temperature")
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.plot(T_real_arr, C, 'd-', label="Heat Capacity (Sim)")
        ax2.axvline(Tc_exp, color='red', ls=':', label=f"Tc = {Tc_exp}K", lw=2)
        ax2.set_xlabel("Temperature (K)")
        ax2.set_ylabel("Specific Heat (per spin)")
        ax2.legend(); ax2.grid(True)
        st.pyplot(fig2)

    # --------------- Susceptibility & Binder ---------------
    with tabs[2]:
        left, right = st.columns(2)
        with left:
            st.subheader("Magnetic Susceptibility χ(T)")
            fig3, ax3 = plt.subplots(figsize=(7,4))
            ax3.plot(T_real_arr, Chi, 'o-', label="χ (per spin)")
            ax3.axvline(Tc_exp, color='red', ls=':', label="Tc (Exp)")
            ax3.set_xlabel("Temperature (K)"); ax3.set_ylabel("χ")
            ax3.legend(); ax3.grid(True)
            st.pyplot(fig3)
        with right:
            st.subheader("Binder Cumulant U4(T)")
            fig4, ax4 = plt.subplots(figsize=(7,4))
            ax4.plot(T_real_arr, U4, 's-', label="U4")
            ax4.axvline(Tc_exp, color='red', ls=':', label="Tc (Exp)")
            ax4.set_xlabel("Temperature (K)"); ax4.set_ylabel("U4 = 1 - ⟨m⁴⟩/(3⟨m²⟩²)")
            ax4.set_ylim(0,1)
            ax4.legend(); ax4.grid(True)
            st.pyplot(fig4)

        st.markdown(
            """
            <div class='explain'>
            <b>Why these?</b> χ peaks around Tc and U4 curves for different sizes cross at Tc. Use this tab for finite-size scaling.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------- Phase Diagram ---------------
    with tabs[3]:
        st.subheader("2D Ising Phase Diagram + Snapshots")
        fig5, ax5 = plt.subplots(figsize=(7,4))
        ax5.axvspan(minT, Tc_exp, alpha=0.25, color='tab:blue', label="Ferromagnetic")
        ax5.axvspan(Tc_exp, maxT, alpha=0.25, color='tab:orange', label="Paramagnetic")
        ax5.axvline(Tc_exp, color='red', ls='--', label="Transition Tc")
        ax5.errorbar(T_real_arr, Mabs, yerr=Mabs_err, fmt='ko', markersize=3, capsize=2, label="|M|")
        ax5.set_xlabel("Temperature (K)"); ax5.set_yticks([])
        ax5.legend()
        st.pyplot(fig5)

        if (spins_lo is not None) and (spins_hi is not None):
            fig_snap, axes = plt.subplots(1,2, figsize=(8,3))
            axes[0].imshow(spins_lo, cmap='bwr', vmin=-1, vmax=1); axes[0].set_title(r"Below $T_c$: Ordered"); axes[0].axis('off')
            axes[1].imshow(spins_hi, cmap='bwr', vmin=-1, vmax=1); axes[1].set_title(r"Above $T_c$: Disordered"); axes[1].axis('off')
            plt.tight_layout(); st.pyplot(fig_snap)
            st.caption("Left: spins aligned; Right: spins random. Phase transition = order/disorder!")

    # --------------- Histogram ---------------
    with tabs[4]:
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

    # --------------- Animations ---------------
    with tabs[5]:
        st.subheader("Animations: Ising in Motion")
        st.markdown(f"**Spin Evolution at T={spin_anim_T} K, Steps={spin_anim_steps}**")
        beta_anim = 1.0 / (spin_anim_T / JkB)
        gif1 = "spin_evolution.gif"
        if (not os.path.exists(gif1)) or st.button("Regenerate Spin Evolution Animation"):
            with st.spinner("Generating animation..."):
                animate_spin_evolution(N, spin_anim_steps, beta_anim, seed, gif1)
        st.image(gif1, caption="Spin lattice evolution at fixed T")

        st.markdown(f"**Temperature Sweep Animation (steps={temp_anim_steps})**")
        gif2 = "ising_temp_sweep.gif"
        T_anim_arr = np.linspace(minT, maxT, min(28, nT))
        if (not os.path.exists(gif2)) or st.button("Regenerate T Sweep Animation"):
            with st.spinner("Generating temperature sweep..."):
                animate_temp_sweep(N, T_anim_arr, temp_anim_steps, seed, JkB, gif2)
        st.image(gif2, caption="Order melts as T crosses Tc")

    # --------------- Export & Comparison ---------------
    with tabs[6]:
        st.subheader("Export & Comparison")
        st.dataframe(df, use_container_width=True)

        # Download CSV
        csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download CSV", csv_buf.getvalue(), file_name=f"ising_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

        # Plot downloads
        def fig_to_png_bytes(fig):
            b = io.BytesIO(); fig.savefig(b, format='png', dpi=200, bbox_inches='tight'); b.seek(0); return b
        # Quick re-render of key plots for download
        figs = []
        f1, a1 = plt.subplots(figsize=(7,4)); a1.errorbar(T_real_arr, Mabs, yerr=Mabs_err, fmt='o-'); a1.set_xlabel('T (K)'); a1.set_ylabel('|M|'); a1.grid(True); figs.append((f1,'magnetization.png'))
        f2, a2 = plt.subplots(figsize=(7,4)); a2.plot(T_real_arr, C, 'd-'); a2.set_xlabel('T (K)'); a2.set_ylabel('C'); a2.grid(True); figs.append((f2,'heat_capacity.png'))
        f3, a3 = plt.subplots(figsize=(7,4)); a3.plot(T_real_arr, Chi, 'o-'); a3.set_xlabel('T (K)'); a3.set_ylabel('χ'); a3.grid(True); figs.append((f3,'susceptibility.png'))
        f4, a4 = plt.subplots(figsize=(7,4)); a4.plot(T_real_arr, U4, 's-'); a4.set_xlabel('T (K)'); a4.set_ylabel('U4'); a4.grid(True); figs.append((f4,'binder.png'))

        cols = st.columns(4)
        for (figX, nameX), col in zip(figs, cols):
            with col:
                st.pyplot(figX)
                st.download_button("⬇️ PNG", data=fig_to_png_bytes(figX), file_name=nameX, mime="image/png")

        # Experimental RMSE if provided
        if exp_data is not None:
            try:
                interp_sim = np.interp(exp_data.iloc[:,0], T_real_arr, Mabs)
                rmse = np.sqrt(np.mean((exp_data.iloc[:,1] - interp_sim) ** 2))
                st.info(f"Simulation vs Experiment RMSE (|M|): {rmse:.4f}")
            except Exception:
                st.warning("Experimental comparison failed to compute.")
else:
    with st.expander("🚦 Quick Guide", expanded=True):
        st.markdown(
            """
            1) Choose **material & algorithm**. 2) Set lattice size, equilibration, samples, and T-range. 3) Click **Run Simulation**.
            Explore tabs for Magnetization, Heat Capacity, **Susceptibility**, **Binder cumulant**, Histograms, Snapshots, and Animations.
            Use **Export** to save CSV/plots for reports and research.
            """
      )
      
  
