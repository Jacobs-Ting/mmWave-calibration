import streamlit as st
import numpy as np
import skrf as rf
import matplotlib.pyplot as plt

# 網頁基本設定
st.set_page_config(page_title="RF Phase Shifter Simulator", layout="wide")
st.title("🎛️ 射頻移相器 (Phase Shifter) S-參數模擬器")
st.markdown("透過側邊欄調整參數，即時觀察移相器的頻率響應（S-參數）。")

# --- 側邊欄 (UI 介面) ---
st.sidebar.header("⚙️ 模擬參數設定")

# 頻率設定
st.sidebar.subheader("頻率範圍")
f_start = st.sidebar.number_input("起始頻率 (GHz)", value=1.0, step=0.1)
f_stop = st.sidebar.number_input("截止頻率 (GHz)", value=10.0, step=0.1)
f_pts = st.sidebar.slider("頻率點數", 11, 1001, 201)

# 移相器規格設定
st.sidebar.subheader("移相器規格")
phase_deg = st.sidebar.slider("目標相位偏移 (度)", -180, 180, 90)
insertion_loss = st.sidebar.slider("插入損耗 Insertion Loss (dB)", 0.0, 10.0, 1.5, step=0.1)
return_loss = st.sidebar.slider("反射損耗 Return Loss (dB)", 10.0, 40.0, 20.0, step=1.0) # 用來模擬非理想的 S11

# --- 後端運算 (scikit-rf) ---
# 1. 建立頻率軸
freq = rf.Frequency(f_start, f_stop, f_pts, 'ghz')

# 2. 準備 S-參數矩陣 (N x 2 x 2)
s_matrix = np.zeros((len(freq), 2, 2), dtype=complex)

# 數學轉換
phase_rad = np.deg2rad(phase_deg)
mag_s21 = 10**(-insertion_loss / 20)
mag_s11 = 10**(-return_loss / 20)

# 填入矩陣 (假設為互易網路 Reciprocal Network: S21 = S12)
s_matrix[:, 0, 0] = mag_s11  # S11
s_matrix[:, 1, 1] = mag_s11  # S22
s_matrix[:, 1, 0] = mag_s21 * np.exp(-1j * phase_rad)  # S21
s_matrix[:, 0, 1] = mag_s21 * np.exp(-1j * phase_rad)  # S12

# 3. 建立 scikit-rf Network 物件
ps_network = rf.Network(frequency=freq, s=s_matrix, name=f'{phase_deg}° Phase Shifter')

# --- 網頁圖表展示 (Matplotlib) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("幅度響應 (Magnitude)")
    fig_mag, ax_mag = plt.subplots(figsize=(6, 4))
    ps_network.plot_s_db(m=1, n=0, ax=ax_mag, label='$S_{21}$ (Insertion Loss)', color='blue')
    ps_network.plot_s_db(m=0, n=0, ax=ax_mag, label='$S_{11}$ (Return Loss)', color='red', linestyle='--')
    ax_mag.set_ylim(-40, 5)
    ax_mag.grid(True)
    st.pyplot(fig_mag)

with col2:
    st.subheader("相位響應 (Phase)")
    fig_phase, ax_phase = plt.subplots(figsize=(6, 4))
    ps_network.plot_s_deg(m=1, n=0, ax=ax_phase, label=f'$S_{21}$ Phase', color='green')
    ax_phase.set_ylim(-180, 180)
    ax_phase.set_yticks(range(-180, 181, 45))
    ax_phase.grid(True)
    st.pyplot(fig_phase)

# 史密斯圖展示
st.subheader("史密斯圖 (Smith Chart)")
fig_smith, ax_smith = plt.subplots(figsize=(4, 4))
ps_network.plot_s_smith(m=0, n=0, ax=ax_smith, label='$S_{11}$', color='red')
st.pyplot(fig_smith)