import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import windows
import os

# --- Force Tech/Dark Theme for Matplotlib ---
plt.style.use('dark_background')
CYAN = '#00e5ff'
NEON_GREEN = '#39ff14'
MAGENTA = '#ff007f'
CRIMSON = '#dc143c'
DARK_BG = '#0e1117' 
plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor': DARK_BG,
    'axes.edgecolor': '#555555',
    'grid.color': '#333333',
    'text.color': '#ffffff',
    'xtick.color': '#aaaaaa',
    'ytick.color': '#aaaaaa',
})

st.set_page_config(page_title="mmWave Beam Calibration", layout="wide")
st.title("🌐 mmWave Beam Calibration & Optimization System")
st.write("Simulating **Distributed PA (Active Antenna System)** architecture. The PSO algorithm jointly optimizes **Phase Perturbation** and **PA Amplitude Tapering (Back-off)** to suppress sidelobes while strictly maintaining EIRP compliance. Now featuring **Hardware Impairments (Phase Error)**.")

# =========================================
# 1. Sidebar Configuration
# =========================================
st.sidebar.header("1. RF Physical Parameters")
freq_ghz = st.sidebar.slider("Operating Frequency (GHz)", 24.0, 43.5, 28.0, step=0.5)
element_type = st.sidebar.selectbox("Antenna Element Type", ["Patch Antenna (EF=cosθ)", "Isotropic (EF=1)"])
d_lambda = st.sidebar.slider("Element Spacing (d/λ)", 0.4, 0.7, 0.5, step=0.05)
elem_gain_dbi = st.sidebar.number_input("Single Element Peak Gain (dBi)", value=5.0, step=0.5)

st.sidebar.header("2. Distributed PA Architecture")
N = st.sidebar.slider("Number of Elements (N)", 8, 32, 16, step=2)
bits = st.sidebar.selectbox("Phase Shifter Resolution (Bits)", [3, 4, 5, "Ideal"], index=0)

pa_max_pout_dbm = st.sidebar.slider("Single PA Max Output Power (dBm)", 0.0, 20.0, 10.0, step=1.0)
pa_dynamic_range_db = st.sidebar.slider("PA Gain Dynamic Range (dB)", 0.0, 20.0, 10.0, step=1.0, help="Maximum allowed back-off for amplitude tapering.")
min_amp = 10 ** (-pa_dynamic_range_db / 20) 

st.sidebar.header("3. Hardware Impairments")
rms_phase_error_deg = st.sidebar.slider("RMS Phase Error (Degrees)", 0.0, 20.0, 5.0, step=1.0, help="Simulates manufacturing tolerances. Injects random Gaussian phase noise to each element.")

st.sidebar.header("4. Regulatory Mask & Scan Settings")
mask_type = st.sidebar.selectbox("Sidelobe Regulatory Mask", ["ETSI Class SS3 (Strict)", "ETSI Class SS2 (Standard)"])
sweep_step = st.sidebar.slider("Scan Resolution (Degrees)", 1, 10, 5)
num_particles = st.sidebar.slider("PSO Swarm Size", 50, 200, 100, step=50)
num_iterations = st.sidebar.slider("PSO Iterations", 50, 400, 150, step=50)

# =========================================
# 2. Physics & Core Functions
# =========================================
c = 3e8
lambda_m = c / (freq_ghz * 1e9)
d = d_lambda * lambda_m  
k = 2 * np.pi / lambda_m 

theta_scan_deg = np.linspace(-90, 90, 1000)
theta_scan = np.radians(theta_scan_deg)

if "Isotropic" in element_type:
    EF = np.ones_like(theta_scan)
else:
    EF = np.cos(theta_scan)
    EF[EF < 0] = 0 

def quantize_phase(phase_array):
    if bits == "Ideal": 
        return np.mod(phase_array, 2 * np.pi)
    step = 2 * np.pi / (2**bits)
    quantized_phase = np.round(phase_array / step) * step
    return np.mod(quantized_phase, 2 * np.pi) # Force 360 back to 0

def calculate_total_pattern_db(phases, amplitudes, hw_phase_error_rad):
    AF = np.zeros_like(theta_scan, dtype=complex)
    actual_phases = phases + hw_phase_error_rad
    for n in range(N):
        AF += amplitudes[n] * np.exp(1j * (n * k * d * np.sin(theta_scan) - actual_phases[n]))
    AF_norm = np.abs(AF) / (np.sum(amplitudes) + 1e-12)
    Total_Linear = AF_norm * EF
    return 20 * np.log10(Total_Linear + 1e-12)

def calculate_real_rf_metrics(phases, amps_linear, pa_max_dbm, elem_gain, hw_phase_error_rad):
    p_out_mw = (10 ** (pa_max_dbm / 10.0)) * (amps_linear ** 2)
    voltage_amps = np.sqrt(p_out_mw)
    actual_phases = phases + hw_phase_error_rad
    
    AF = np.zeros_like(theta_scan, dtype=complex)
    for n in range(N):
        AF += voltage_amps[n] * np.exp(1j * (n * k * d * np.sin(theta_scan) - actual_phases[n]))
    
    EIRP_dbm_pattern = 10 * np.log10(np.abs(AF * EF)**2 + 1e-12) + elem_gain
    Total_Conducted_dBm = 10 * np.log10(np.sum(p_out_mw) + 1e-12)
    return EIRP_dbm_pattern, Total_Conducted_dBm, p_out_mw

def get_mask(target_theta_deg):
    m = np.zeros_like(theta_scan_deg)
    for i, th in enumerate(theta_scan_deg):
        dist = np.abs(th - target_theta_deg)
        if "SS3" in mask_type:
            angles, limits = [0, 12, 20, 90, 105, 140, 180], [10, 10, -20, -20, -30, -35, -35]
        else:
            angles, limits = [0, 15, 25, 90, 100, 180], [10, 10, -15, -15, -25, -30]
        m[i] = np.interp(dist, angles, limits)
    return m

def evaluate_fitness(particle_2N, target_deg, hw_phase_error_rad):
    p_phases, p_amps = particle_2N[:N], particle_2N[N:]
    q_phases = quantize_phase(p_phases)
    total_db = calculate_total_pattern_db(q_phases, p_amps, hw_phase_error_rad)
    mask_current = get_mask(target_deg)
    
    target_idx = np.argmin(np.abs(theta_scan_deg - target_deg))
    gain_target = total_db[target_idx]
    
    peak_idx = np.argmax(total_db)
    pointing_error = np.abs(theta_scan_deg[peak_idx] - target_deg)
    alignment_penalty = 20000.0 if pointing_error > 3.0 else 0.0
    
    overshoot = total_db - mask_current
    violation = overshoot[overshoot > 0]
    sll_penalty = np.sum(violation ** 2) if len(violation) > 0 else 0
    
    cost = - (1.0 * gain_target) + (20.0 * sll_penalty) + alignment_penalty
    max_violation = np.max(violation) if len(violation) > 0 else 0.0
    
    return cost, gain_target, max_violation, total_db


# =========================================
# 3. Main UI Layout (Tabs)
# =========================================
tab_sim, tab_theory = st.tabs(["🎛️ Simulation Dashboard", "📐 Architecture & Theory"])

# -----------------------------------------
# Tab 1: Simulation Dashboard
# -----------------------------------------
with tab_sim:
    if st.button("🚀 Launch Global Scan & Auto-Rescue", type="primary"):
        np.random.seed() 
        hw_phase_error_rad = np.radians(np.random.normal(0, rms_phase_error_deg, N))
        
        scan_angles = np.arange(-60, 61, sweep_step)
        sweep_results = []
        
        st.write("### 🔍 Phase 1: Global Regulatory Scan (-60° to +60°)")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, angle in enumerate(scan_angles):
            status_text.text(f"Verifying scan angle: {angle}° ...")
            
            std_phases = np.arange(N) * k * d * np.sin(np.radians(angle))
            std_amps = windows.chebwin(N, at=30) 
            std_amps = np.clip(std_amps, min_amp, 1.0) 
            std_particle = np.concatenate([std_phases, std_amps])
            
            cost, gain, violation, pattern = evaluate_fitness(std_particle, angle, hw_phase_error_rad)
            
            sweep_results.append({
                "Angle": angle,
                "Peak_Gain_Norm": gain,
                "Max_Violation": violation,
                "Particle": std_particle,
                "Pattern": pattern
            })
            progress_bar.progress((i + 1) / len(scan_angles))
            
        status_text.success("✅ Global Scan Completed Successfully!")
        df_results = pd.DataFrame(sweep_results)
        
        st.write("#### ⚠️ Regulatory Violation State across Scan Angles")
        fig1, ax1 = plt.subplots(figsize=(12, 3))
        colors = [CRIMSON if v > 0 else NEON_GREEN for v in df_results['Max_Violation']]
        ax1.bar(df_results['Angle'], df_results['Max_Violation'], width=sweep_step*0.6, color=colors, edgecolor='black')
        ax1.axhline(0, color='white', linewidth=1)
        ax1.set_xlabel("Scan Angle (Degrees)", fontsize=10, fontweight='bold')
        ax1.set_ylabel("Max Violation (dB)", fontsize=10, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig1)
        
        worst_idx = df_results['Max_Violation'].idxmax()
        worst_case = df_results.iloc[worst_idx]
        worst_angle = worst_case['Angle']
        worst_violation = worst_case['Max_Violation']
        worst_std_pattern = worst_case['Pattern']
        worst_std_particle = worst_case['Particle']
        
        if worst_violation <= 0:
            st.balloons()
            st.success("🎉 Perfect Architecture! Current HW constraints passed all regulatory masks even with hardware impairments.")
        else:
            st.error(f"🚨 CRITICAL VIOLATION: At target angle **{worst_angle}°**, sidelobe exceeds mask by **{worst_violation:.2f} dB**.")
            st.write(f"### 🛠️ Phase 2: PSO Joint Optimization Rescue for Worst-Case ({worst_angle}°)")
            st.caption("PSO algorithm initiating search to mitigate hardware quantization limits and inherent phase errors...")
            
            particles = np.zeros((num_particles, 2*N))
            particles[0] = worst_std_particle 
            
            std_phases, std_amps = worst_std_particle[:N], worst_std_particle[N:]
            for p in range(1, int(num_particles * 0.7)):
                mutated_phases = std_phases + np.random.normal(0, 0.3, N)
                mutated_amps = std_amps + np.random.normal(0, 0.05, N)
                particles[p] = np.concatenate([mutated_phases, mutated_amps])
            for p in range(int(num_particles * 0.7), num_particles):
                particles[p] = np.concatenate([np.random.uniform(0, 2*np.pi, N), np.random.uniform(min_amp, 1.0, N)])
                
            velocities = np.random.uniform(-0.1, 0.1, (num_particles, 2*N))
            w, c1, c2 = 0.5, 1.5, 1.5
            
            pbest_positions = particles.copy()
            pbest_costs = np.array([evaluate_fitness(p, worst_angle, hw_phase_error_rad)[0] for p in particles])
            gbest_idx = np.argmin(pbest_costs)
            gbest_position = pbest_positions[gbest_idx].copy()
            gbest_cost = pbest_costs[gbest_idx]
            
            pso_prog = st.progress(0)
            pso_stat = st.empty()
            
            for i in range(num_iterations):
                r1, r2 = np.random.rand(num_particles, 2*N), np.random.rand(num_particles, 2*N)
                velocities = (w * velocities + c1 * r1 * (pbest_positions - particles) + c2 * r2 * (gbest_position - particles))
                particles = particles + velocities
                particles[:, :N] = np.mod(particles[:, :N], 2*np.pi)
                particles[:, N:] = np.clip(particles[:, N:], min_amp, 1.0)
                
                for p_idx in range(num_particles):
                    cost, _, _, _ = evaluate_fitness(particles[p_idx], worst_angle, hw_phase_error_rad)
                    if cost < pbest_costs[p_idx]:
                        pbest_costs[p_idx] = cost
                        pbest_positions[p_idx] = particles[p_idx].copy()
                        if cost < gbest_cost:
                            gbest_cost = cost
                            gbest_position = particles[p_idx].copy()
                if i % 10 == 0: pso_prog.progress((i + 1) / num_iterations)
                    
            pso_prog.progress(1.0)
            _, _, opt_violation, opt_pattern = evaluate_fitness(gbest_position, worst_angle, hw_phase_error_rad)
            
            if opt_violation <= 0:
                pso_stat.success(f"✅ RESCUE SUCCESSFUL: Sidelobes suppressed within ETSI mask by adjusting PA Back-off and Phase Perturbation.")
            else:
                pso_stat.warning(f"⚠️ PHYSICAL LIMIT REACHED: Hardware constraints/impairments are too severe. Residual violation of {opt_violation:.2f} dB remains.")

            # --- RF Metrics ---
            std_eirp_pattern, std_total_cond, _ = calculate_real_rf_metrics(worst_std_particle[:N], worst_std_particle[N:], pa_max_pout_dbm, elem_gain_dbi, hw_phase_error_rad)
            std_peak_eirp = np.max(std_eirp_pattern)
            
            opt_eirp_pattern, opt_total_cond, opt_pout_mw = calculate_real_rf_metrics(quantize_phase(gbest_position[:N]), gbest_position[N:], pa_max_pout_dbm, elem_gain_dbi, hw_phase_error_rad)
            opt_peak_eirp = np.max(opt_eirp_pattern)

            st.write("#### 📊 System Trade-off Dashboard (EIRP vs Power)")
            col1, col2 = st.columns(2)
            with col1:
                st.info("Standard Algorithm (Ideal Phase + Cheb.)")
                st.metric("Estimated Peak EIRP", f"{std_peak_eirp:.2f} dBm")
                st.metric("Total Conducted Power", f"{std_total_cond:.2f} dBm")
                st.metric("Max Mask Violation", f"{worst_violation:.2f} dB", delta="FAIL", delta_color="inverse")
            with col2:
                st.success("PSO Smart Rescued Algorithm (AAS Mode)")
                st.metric("Estimated Peak EIRP", f"{opt_peak_eirp:.2f} dBm", delta=f"{opt_peak_eirp - std_peak_eirp:.2f} dB (Tapering Loss)")
                st.metric("Total Conducted Power", f"{opt_total_cond:.2f} dBm", delta=f"{opt_total_cond - std_total_cond:.2f} dB (Back-off Saving)")
                if opt_violation <= 0:
                    st.metric("Max Mask Violation", "0.00 dB", delta="PASS", delta_color="normal")
                else:
                    st.metric("Max Mask Violation", f"{opt_violation:.2f} dB", delta=f"{opt_violation - worst_violation:.2f} dB (Mitigated)", delta_color="inverse")

            # --- Baseband & RFIC Status ---
            st.write("#### 🎛️ Under the Hood: Distributed PA Power & Independent Phase Control")
            st.caption("Top: Actual output power per PA element. Bottom: Quantized phase shift applied per element (inclusive of targeted perturbation).")
            
            ant_indices = np.arange(1, N+1)
            opt_pout_dbm = 10 * np.log10(opt_pout_mw + 1e-12)
            opt_phases_deg = np.degrees(quantize_phase(gbest_position[:N]))

            fig3, (ax_pa, ax_phase) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
            
            ax_pa.bar(ant_indices, opt_pout_dbm, color=CYAN, edgecolor='#ffffff', alpha=0.8)
            ax_pa.axhline(pa_max_pout_dbm, color='#888888', linestyle='--', linewidth=2, label='PA Saturation Limit (Pmax)')
            ax_pa.set_ylabel("PA Power (dBm)", color=CYAN, fontweight='bold')
            ax_pa.set_title("Independent PA Amplitude Tapering (Gain Control)", color='#ffffff', fontweight='bold')
            ax_pa.set_ylim(pa_max_pout_dbm - pa_dynamic_range_db - 2, pa_max_pout_dbm + 2)
            ax_pa.grid(True, linestyle='--', alpha=0.3, axis='y')
            ax_pa.legend(loc='lower right', facecolor=DARK_BG, edgecolor='#555555', labelcolor='#ffffff')
            
            ax_phase.bar(ant_indices, opt_phases_deg, color=MAGENTA, edgecolor='#ffffff', alpha=0.8)
            ax_phase.set_xlabel("Antenna Element Index", fontweight='bold')
            ax_phase.set_ylabel("Phase Shift (Degrees)", color=MAGENTA, fontweight='bold')
            ax_phase.set_title("Independent Phase Shifter State (Perturbation)", color='#ffffff', fontweight='bold')
            ax_phase.set_ylim(0, 360)
            ax_phase.set_yticks(np.arange(0, 361, 45)) 
            ax_phase.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig3)

            # --- Plot: Radiation Pattern Before vs After ---
            st.write(f"#### 📡 Radiation Pattern Rescue at Worst Case Angle ({worst_angle}°)")
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            
            mask_worst = get_mask(worst_angle)
            ax2.plot(theta_scan_deg, mask_worst, color=NEON_GREEN, linestyle='-', linewidth=2.5, label="ETSI Regulatory Mask")
            ax2.plot(theta_scan_deg, worst_std_pattern, color=CRIMSON, linestyle='--', linewidth=1.5, alpha=0.7, label="Standard Algorithm (Failed)")
            ax2.plot(theta_scan_deg, opt_pattern, color=CYAN, linewidth=2.5, label="PSO Rescued Pattern")
            
            ax2.fill_between(theta_scan_deg, worst_std_pattern, mask_worst, where=(worst_std_pattern > mask_worst), color=CRIMSON, alpha=0.4)
            
            ax2.set_xlabel("Angle (Degrees)", fontsize=11, fontweight='bold')
            ax2.set_ylabel("Normalized Gain (dBc)", fontsize=11, fontweight='bold')
            ax2.set_ylim(-45, 5)
            ax2.set_xlim(-90, 90)
            ax2.grid(True, linestyle='--', alpha=0.3)
            ax2.legend(loc='lower left', facecolor=DARK_BG, edgecolor='#555555', labelcolor='#ffffff')
            st.pyplot(fig2)

# -----------------------------------------
# Tab 2: Architecture & Theory
# -----------------------------------------
with tab_theory:
    st.header("1. Distributed AAS Architecture")
    st.markdown("In an Active Antenna System (AAS), the Digital Signal Processor (DSP) exerts independent control over both the **Phase Shifter (Beamforming Weights)** and the **Power Amplifier (Gain Control)** of each individual Tx branch. This dual-path control is crucial for advanced amplitude tapering and phase perturbation.")
    
    if os.path.exists("architecture.png"):
        st.image("architecture.png", caption="DSP-RF Co-design: Independent Gain & Phase Control Topology", use_column_width=True)
    else:
        st.info("💡 Tip: Upload 'architecture.png' to the project folder to view the system architecture diagram here.")

    st.markdown("---")
    
    st.header("2. Mathematical Foundation")
    st.markdown("### 📡 Array Factor with Hardware Impairments")
    st.markdown("The synthesized far-field radiation pattern is determined by the complex superposition of signals from all antenna elements, governed by the Array Factor (AF):")
    st.latex(r"AF(\theta) = \sum_{n=0}^{N-1} A_n \cdot e^{j(n k d \sin\theta - \phi_{actual, n})}")
    st.markdown(r"""
    * $A_n$: The PA output amplitude for the $n$-th element, controlled via dynamic Back-off limits.
    * $k$: Free-space wavenumber ($2\pi / \lambda$).
    * $d$: Antenna element spacing.
    * $\phi_{actual, n}$: The final phase applied to the element, accounting for both digital quantization and analog manufacturing defects.
    """)
    st.latex(r"\phi_{actual, n} = \phi_{quantized, n} + \phi_{error, n}")
    st.markdown(r"Where $\phi_{error, n} \sim \mathcal{N}(0, \sigma_{RMS}^2)$ injects real-world semiconductor process variation into the simulation.")

    st.markdown("### 🎯 Total Cost Function (Fitness Function)")
    st.markdown("To force the PSO algorithm to suppress sidelobes without completely destroying the main beam, the fitness function is formulated as a linear combination of three penalties:")
    st.latex(r"C(\mathbf{X}) = -w_{gain} \cdot G_{target}(\mathbf{X}) + w_{sll} \cdot \Phi_{sll}(\mathbf{X}) + \Phi_{align}(\mathbf{X})")
    st.markdown(r"""
    1. **Target Gain Objective ($-w_{gain} \cdot G_{target}$):** Seeks to maximize the Equivalent Isotropically Radiated Power (EIRP) towards the desired user equipment (UE).
    2. **Sidelobe Mask Penalty ($w_{sll} \cdot \Phi_{sll}$):** Heavily penalizes any radiated energy that breaches the strict ETSI regulatory mask using a squared-error approach.
    3. **Alignment Penalty ($\Phi_{align}$):** Acts as a hard constraint step-function. If the phase perturbation causes the main beam to squint (deviate) by more than a specified tolerance (e.g., $3^\circ$), the solution is rejected with an immense penalty cost.
    """)

    st.markdown("---")

    st.header("3. AI-Assisted Optimization Journey")
    st.markdown("Particle Swarm Optimization (PSO) navigates the complex $2N$-dimensional space (evaluating Phase and Gain simultaneously) to discover a Pareto optimal solution that satisfies all physical and regulatory constraints.")
    
    if os.path.exists("pso_flow.png"):
        st.image("pso_flow.png", caption="PSO Algorithm Flow for mmWave Sidelobe Suppression", use_column_width=True)
    else:
        st.info("💡 Tip: Upload 'pso_flow.png' to the project folder to view the algorithm workflow diagram here.")