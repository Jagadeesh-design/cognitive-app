import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
from streamlit_geolocation import streamlit_geolocation

# --- App Configuration ---
st.set_page_config(page_title="Cognitive MIMO Beamforming for 5G", layout="wide")

# --- API Key Placeholder (if needed later) ---
API_KEY = "9a5a993fa23f03aa237335f2a4db293"

# --- App Title and Header ---
st.title("📡 Cognitive MIMO Beamforming for 5G")
st.header("Real-Time MIMO Beamforming Demo")
st.markdown("""
This app demonstrates **MIMO (Multiple Input Multiple Output)** beamforming for 5G communication.
It shows how a base station dynamically steers its beam toward the user's location using transmit (Tx) 
and receive (Rx) antennas, with adjustable array size and windowing for side-lobe reduction.
""")

# --- User Location Section ---
st.write("### 🧭 User Location")
location = streamlit_geolocation()
user_lat = None
user_lon = None

if location and location.get('latitude'):
    st.success("✅ Live Location retrieved successfully!")
    user_lat = location['latitude']
    user_lon = location['longitude']
    st.write(f"Your Location: **Latitude:** {user_lat}, **Longitude:** {user_lon}")
else:
    st.info("📍 Click the **Allow** button in the pop-up to get your live location.")

# --- Sidebar Controls ---
st.sidebar.markdown("## ⚙️ App Controls")
user_name = st.sidebar.text_input("Enter your name:", "Guest")

Nt = st.sidebar.slider("Number of Transmit Antennas (Nt)", 2, 32, 8)
Nr = st.sidebar.slider("Number of Receive Antennas (Nr)", 1, 16, 4)
snr_input = st.sidebar.slider("Input SNR (dB)", 0, 30, 10)

# Add window type selector
window_type = st.sidebar.selectbox(
    "Select Window Type for Beamforming",
    ["Uniform", "Hamming", "Hanning", "Blackman"],
    index=1
)

# --- MIMO Beamforming Function ---
def mimo_array_response(Nt, Nr, theta_user, theta, window_type="Hamming"):
    """
    Simulates MIMO beamforming pattern for Nt transmit and Nr receive antennas.
    """
    d = 0.5  # spacing = half wavelength
    k = 2 * np.pi  # wavenumber (λ = 1)

    # --- Transmit and Receive Windows ---
    if window_type == "Hamming":
        win_tx = np.hamming(Nt)
        win_rx = np.hamming(Nr)
    elif window_type == "Hanning":
        win_tx = np.hanning(Nt)
        win_rx = np.hanning(Nr)
    elif window_type == "Blackman":
        win_tx = np.blackman(Nt)
        win_rx = np.blackman(Nr)
    else:
        win_tx = np.ones(Nt)
        win_rx = np.ones(Nr)

    # Normalize
    win_tx = win_tx / np.linalg.norm(win_tx)
    win_rx = win_rx / np.linalg.norm(win_rx)

    # --- Steering Vectors ---
    tx_vec_user = win_tx * np.exp(1j * k * d * np.arange(Nt) * np.sin(np.radians(theta_user)))
    rx_vec_user = win_rx * np.exp(1j * k * d * np.arange(Nr) * np.sin(np.radians(theta_user)))

    # --- MIMO Channel Matrix ---
    H = np.outer(rx_vec_user, tx_vec_user.conj())

    response = []
    for ang in theta:
        tx_vec = win_tx * np.exp(1j * k * d * np.arange(Nt) * np.sin(np.radians(ang)))
        rx_vec = win_rx * np.exp(1j * k * d * np.arange(Nr) * np.sin(np.radians(ang)))
        H_steer = np.outer(rx_vec, tx_vec.conj())
        gain = np.abs(np.sum(H_steer * H.conj()))
        response.append(gain)

    response = np.array(response)
    return 20 * np.log10(response / np.max(response))

# --- Beam Steering Logic ---
final_theta = 0  # Default if no live location
bs_lat, bs_lon = 28.7041, 77.1025  # Example Base Station (Delhi)

if user_lat is not None and user_lon is not None:
    st.write("### Beam Steering Source: **Live Location**")

    delta_lon = user_lon - bs_lon
    delta_lat = user_lat - bs_lat
    angle_rad = np.arctan2(delta_lon, delta_lat)
    theta_from_location = np.degrees(angle_rad)

    # Normalize angle
    if theta_from_location > 90:
        theta_from_location = 180 - theta_from_location
    elif theta_from_location < -90:
        theta_from_location = -180 - theta_from_location

    final_theta = theta_from_location
    st.write(f"🧭 Steering angle auto-calculated: **{final_theta:.2f}°**")
else:
    st.warning("⚠️ No live location available. Defaulting steering angle to 0°.")

# --- Map Visualization ---
st.write("### 🌍 Map Visualization")
map_data = [{'lat': bs_lat, 'lon': bs_lon, 'type': 'Base Station'}]
if user_lat is not None and user_lon is not None:
    map_data.append({'lat': user_lat, 'lon': user_lon, 'type': 'Live User'})

df = pd.DataFrame(map_data)
st.map(df)

# --- Beam Pattern Calculation ---
theta = np.linspace(-90, 90, 500)
resp_mimo = mimo_array_response(Nt, Nr, final_theta, theta, window_type)
resp_ref = mimo_array_response(8, 4, final_theta, theta, window_type)

# --- Beam Pattern Plot (Polar) ---
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
ax.plot(np.radians(theta), resp_mimo, label=f"MIMO Nt={Nt}, Nr={Nr}")
ax.plot(np.radians(theta), resp_ref, '--', label="Reference Nt=8, Nr=4")

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_title(f"MIMO Beamforming Pattern (Steered to {final_theta:.2f}°)\nWindow: {window_type}", pad=20)
ax.set_ylim(-40, 0)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
st.pyplot(fig)

# --- Observations ---
st.write("### 📊 Key Observations")
st.write(f"- **Main lobe is steered at:** {final_theta:.2f}°")
st.write(f"- **Transmit antennas (Nt):** {Nt}")
st.write(f"- **Receive antennas (Nr):** {Nr}")
st.write(f"- **Window type applied:** {window_type}")
st.write("- Windowing reduces side lobes significantly at the cost of slightly wider main lobes.")
st.write("- Increasing Nt and Nr improves directivity, gain, and overall MIMO capacity.")

# --- Performance Metrics ---
snr_linear = 10**(snr_input/10)
snr_effective = snr_linear * Nt * Nr
capacity = np.log2(1 + snr_effective)

st.write("### ⚡ Performance Metrics")
st.write(f"- Input SNR = {snr_input} dB")
st.write(f"- Effective SNR with Nt={Nt}, Nr={Nr} ≈ {10 * np.log10(snr_effective):.2f} dB")
st.write(f"- Channel Capacity ≈ {capacity:.2f} bits/sec/Hz")

# --- Final Greeting ---
st.markdown(f"### 👋 Hello, **{user_name}!** — Enjoy exploring Cognitive MIMO Beamforming!")
