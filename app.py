import time
import io
import csv
import zipfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import find_peaks, fftconvolve


C = 299_792_458.0


# ---------------------------------------------------------
# 1. Generate the transmitted LFM chirp
# ---------------------------------------------------------
def generate_chirp(fs, pulse_width, bandwidth):
    n = max(2, int(round(fs * pulse_width)))
    t = np.arange(n) / fs
    centered_t = t - (n - 1) / (2 * fs)
    chirp_rate = bandwidth / pulse_width

    signal = np.exp(1j * np.pi * chirp_rate * centered_t**2)
    return t, signal


# ---------------------------------------------------------
# 2. Simulate multiple target echoes and noise
# ---------------------------------------------------------
def simulate_echoes(
    tx, fs, target_ranges, target_amplitudes, snr_db, seed
):
    delays = [
        int(round(2 * distance * fs / C))
        for distance in target_ranges
    ]

    rx_length = max(delays) + len(tx)
    rx_clean = np.zeros(rx_length, dtype=np.complex128)

    for delay, amplitude in zip(delays, target_amplitudes):
        rx_clean[delay:delay + len(tx)] += amplitude * tx

    rng = np.random.default_rng(seed)

    signal_power = np.mean(np.abs(rx_clean) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(rx_length)
        + 1j * rng.standard_normal(rx_length)
    )

    return rx_clean + noise, rx_clean, delays


# ---------------------------------------------------------
# 3. Time-domain and FFT-based matched filters
# ---------------------------------------------------------
def direct_matched_filter(rx, reference):
    """Direct time-domain linear convolution."""
    template = np.conj(reference[::-1])

    start = time.perf_counter()
    result = np.convolve(rx, template, mode="full")
    elapsed_ms = (time.perf_counter() - start) * 1000

    return result, elapsed_ms


def matched_filter(rx, reference):
    """FFT-based linear convolution."""
    template = np.conj(reference[::-1])

    start = time.perf_counter()
    result = fftconvolve(rx, template, mode="full")
    elapsed_ms = (time.perf_counter() - start) * 1000

    return result, elapsed_ms


# ---------------------------------------------------------
# 4. Convert matched-filter output to a range profile
# ---------------------------------------------------------
def range_profile(output, tx_length, fs):
    positive = output[tx_length - 1:]
    ranges = np.arange(len(positive)) * C / (2 * fs)

    magnitude = np.abs(positive)
    peak = max(float(np.max(magnitude)), 1e-15)

    db = 20 * np.log10(
        np.maximum(magnitude / peak, 1e-12)
    )

    return ranges, magnitude, db


# ---------------------------------------------------------
# 5. Estimate peak sidelobe level
# ---------------------------------------------------------
def peak_sidelobe_level(magnitude, fs, bandwidth):
    peak_index = int(np.argmax(magnitude))
    peak_value = max(float(magnitude[peak_index]), 1e-15)

    guard = max(2, int(np.ceil(fs / bandwidth)))

    left = max(0, peak_index - guard)
    right = min(len(magnitude), peak_index + guard + 1)

    sidelobes = np.concatenate((
        magnitude[:left],
        magnitude[right:]
    ))

    if len(sidelobes) == 0:
        return float("nan")

    sidelobe_peak = max(float(np.max(sidelobes)), 1e-15)

    return float(20 * np.log10(sidelobe_peak / peak_value))


# ---------------------------------------------------------
# 6. Detect prominent peaks
# ---------------------------------------------------------
def detect_targets(ranges, magnitude, fs, bandwidth):
    peak_height = max(float(np.max(magnitude)) * 0.12, 1e-12)
    min_distance = max(2, int(round(fs / bandwidth)))

    indices, _ = find_peaks(
        magnitude,
        height=peak_height,
        distance=min_distance,
        prominence=peak_height * 0.20
    )

    indices = sorted(
        indices,
        key=lambda i: magnitude[i],
        reverse=True
    )

    return [
        (float(ranges[i]), float(magnitude[i]))
        for i in indices[:10]
    ]


# ---------------------------------------------------------
# 7. Dashboard setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Radar Pulse Compression Lab",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Radar Pulse Compression via Matched Filtering")

st.caption(
    "Interactive DSP simulation | LFM chirps | Multiple targets | "
    "FFT filtering | Noise | Sidelobe analysis"
)

st.sidebar.header("Radar Parameters")

pulse_us = st.sidebar.slider(
    "Pulse width (µs)", 5.0, 40.0, 20.0, 1.0
)

fs_mhz = st.sidebar.slider(
    "Sampling frequency (MHz)", 10.0, 50.0, 20.0, 2.0
)

bandwidth_mhz = st.sidebar.slider(
    "Bandwidth (MHz)", 0.5, 15.0, 5.0, 0.5
)

snr_db = st.sidebar.slider(
    "Receive-record SNR (dB)", -10, 30, 10, 1
)

number_targets = st.sidebar.slider(
    "Number of targets", 1, 4, 2, 1
)

fs = fs_mhz * 1e6
bandwidth = bandwidth_mhz * 1e6
pulse_width = pulse_us * 1e-6

if bandwidth >= fs:
    st.error(
        "Bandwidth must be lower than the sampling frequency."
    )
    st.stop()

st.sidebar.subheader("Target Configuration")

target_defaults = [1500.0, 1650.0, 3500.0, 5000.0]
amplitude_defaults = [1.0, 0.65, 0.80, 0.50]

target_ranges = []
target_amplitudes = []

for i in range(number_targets):
    with st.sidebar.expander(
        f"Target {i + 1}", expanded=True
    ):
        distance = st.slider(
            f"Distance (m) — target {i + 1}",
            100, 10000,
            int(target_defaults[i]),
            50,
            key=f"distance_{i}"
        )

        amplitude = st.slider(
            f"Relative amplitude — target {i + 1}",
            0.1, 1.0,
            amplitude_defaults[i],
            0.05,
            key=f"amplitude_{i}"
        )

    target_ranges.append(float(distance))
    target_amplitudes.append(float(amplitude))

seed = st.sidebar.number_input(
    "Noise random seed",
    min_value=0,
    max_value=99999,
    value=42,
    step=1
)

st.sidebar.info(
    "Change a parameter to rerun the simulation. "
    "The same seed reproduces the same noise for unchanged settings."
)


# ---------------------------------------------------------
# 8. Run simulation and compare filtering methods
# ---------------------------------------------------------
time_axis, tx = generate_chirp(
    fs, pulse_width, bandwidth
)

rx, rx_clean, delays = simulate_echoes(
    tx,
    fs,
    target_ranges,
    target_amplitudes,
    snr_db,
    int(seed)
)

# Direct time-domain matched filtering
direct_output, direct_time_ms = direct_matched_filter(rx, tx)

# FFT-based matched filtering
rect_output, fft_time_ms = matched_filter(rx, tx)

# Numerical accuracy check
max_filter_difference = float(
    np.max(np.abs(direct_output - rect_output))
)

# Hamming-windowed reference comparison
window = np.hamming(len(tx))
windowed_reference = tx * window

hamming_output, hamming_time_ms = matched_filter(
    rx, windowed_reference
)

ranges, rect_magnitude, rect_db = range_profile(
    rect_output, len(tx), fs
)

_, hamming_magnitude, hamming_db = range_profile(
    hamming_output, len(tx), fs
)

resolution_m = C / (2 * bandwidth)
time_bandwidth_product = pulse_width * bandwidth

rect_pslr = peak_sidelobe_level(
    rect_magnitude, fs, bandwidth
)

hamming_pslr = peak_sidelobe_level(
    hamming_magnitude, fs, bandwidth
)

detections = detect_targets(
    ranges, rect_magnitude, fs, bandwidth
)


# ---------------------------------------------------------
# 9. Summary metrics
# ---------------------------------------------------------
st.header("Simulation Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Range resolution", f"{resolution_m:.2f} m")
m2.metric("Time-bandwidth product", f"{time_bandwidth_product:.1f}")
m3.metric("Rectangular PSLR", f"{rect_pslr:.2f} dB")
m4.metric("Detected peaks", str(len(detections)))

st.info(
    f"Pulse samples: {len(tx):,} | "
    f"Receive samples: {len(rx):,} | "
    f"FFT filtering time: {fft_time_ms:.3f} ms"
)


# ---------------------------------------------------------
# 10. Graph: transmitted radar pulse
# ---------------------------------------------------------
st.header("1. Transmitted Radar Pulse")

tx_fig = go.Figure()

tx_fig.add_trace(go.Scatter(
    x=time_axis * 1e6,
    y=np.real(tx),
    mode="lines",
    name="In-phase (real)"
))

tx_fig.add_trace(go.Scatter(
    x=time_axis * 1e6,
    y=np.imag(tx),
    mode="lines",
    name="Quadrature (imaginary)"
))

tx_fig.update_layout(
    xaxis_title="Time (µs)",
    yaxis_title="Amplitude",
    height=370,
    hovermode="x unified"
)

st.plotly_chart(tx_fig, use_container_width=True)


# ---------------------------------------------------------
# 11. Graph: received signal
# ---------------------------------------------------------
st.header("2. Received Radar Echoes")

rx_time_us = np.arange(len(rx)) / fs * 1e6
stride = max(1, len(rx) // 10000)

rx_fig = go.Figure()

rx_fig.add_trace(go.Scatter(
    x=rx_time_us[::stride],
    y=np.real(rx[::stride]),
    mode="lines",
    name="Received noisy signal"
))

rx_fig.add_trace(go.Scatter(
    x=rx_time_us[::stride],
    y=np.real(rx_clean[::stride]),
    mode="lines",
    name="Noise-free reference",
    opacity=0.7
))

rx_fig.update_layout(
    xaxis_title="Time (µs)",
    yaxis_title="Amplitude",
    height=370,
    hovermode="x unified"
)

st.plotly_chart(rx_fig, use_container_width=True)


# ---------------------------------------------------------
# 12. Graph: matched-filter range profile
# ---------------------------------------------------------
st.header("3. FFT-Based Matched Filter Output")

range_fig = go.Figure()

range_fig.add_trace(go.Scatter(
    x=ranges,
    y=rect_db,
    mode="lines",
    name="Rectangular matched filter"
))

range_fig.add_trace(go.Scatter(
    x=ranges,
    y=hamming_db,
    mode="lines",
    name="Hamming-windowed reference"
))

for i, distance in enumerate(target_ranges):
    range_fig.add_vline(
        x=distance,
        line_dash="dot",
        opacity=0.45,
        annotation_text=f"T{i + 1}",
        annotation_position="top"
    )

range_fig.update_layout(
    xaxis_title="Range (m)",
    yaxis_title="Normalized magnitude (dB)",
    yaxis=dict(range=[-65, 3]),
    height=470,
    hovermode="x unified"
)

st.plotly_chart(range_fig, use_container_width=True)

st.caption(
    "Dotted lines mark configured target distances. Detected peaks "
    "may differ because of noise, sidelobes and target interference."
)


# ---------------------------------------------------------
# 13. Graph: sidelobe and execution-time comparisons
# ---------------------------------------------------------
st.header("4. Sidelobe and Filtering Performance")

left, right = st.columns(2)

with left:
    st.subheader("Peak sidelobe level")

    fig_pslr = go.Figure()

    fig_pslr.add_trace(go.Bar(
        x=["Rectangular", "Hamming-windowed"],
        y=[rect_pslr, hamming_pslr],
        text=[
            f"{rect_pslr:.2f} dB",
            f"{hamming_pslr:.2f} dB"
        ],
        textposition="auto"
    ))

    fig_pslr.update_layout(
        yaxis_title="PSLR (dB; more negative is better)",
        height=350
    )

    st.plotly_chart(fig_pslr, use_container_width=True)

with right:
    st.subheader("Filter execution time")

    fig_time = go.Figure()

    fig_time.add_trace(go.Bar(
        x=[
            "Direct time-domain",
            "FFT rectangular",
            "FFT Hamming"
        ],
        y=[
            direct_time_ms,
            fft_time_ms,
            hamming_time_ms
        ],
        text=[
            f"{direct_time_ms:.3f} ms",
            f"{fft_time_ms:.3f} ms",
            f"{hamming_time_ms:.3f} ms"
        ],
        textposition="auto"
    ))

    fig_time.update_layout(
        yaxis_title="Execution time (ms)",
        xaxis_title="Filtering method",
        height=350
    )

    st.plotly_chart(fig_time, use_container_width=True)

st.subheader("Numerical comparison: direct vs FFT")

st.metric(
    "Maximum absolute output difference",
    f"{max_filter_difference:.3e}"
)

st.caption(
    "The direct and FFT filters use the same reference template. "
    "Small differences can arise from floating-point rounding. "
    "Execution times depend on your computer and system workload."
)


# ---------------------------------------------------------
# 14. Detected target table
# ---------------------------------------------------------
st.header("5. Target Detection Results")

if detections:
    st.dataframe(
        [
            {
                "Rank": i + 1,
                "Estimated range (m)": round(distance, 2),
                "Relative peak magnitude": round(float(amplitude), 4)
            }
            for i, (distance, amplitude) in enumerate(detections)
        ],
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning(
        "No prominent peaks detected. Try increasing SNR "
        "or adjusting the target parameters."
    )

st.subheader("Configured target ranges")

for i, distance in enumerate(target_ranges):
    delay_seconds = 2 * distance / C

    st.write(
        f"Target {i + 1}: {distance:.0f} m | "
        f"Round-trip delay: {delay_seconds * 1e6:.3f} µs"
    )


# ---------------------------------------------------------
# 15. Export CSV and interactive graphs
# ---------------------------------------------------------
st.header("6. Export Results")

st.write(
    "Download the simulation settings, performance metrics, "
    "detected peaks and interactive graphs in a ZIP file."
)

csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer)

writer.writerow(["Category", "Parameter", "Value"])

settings = [
    ("Pulse width (us)", pulse_us),
    ("Sampling frequency (MHz)", fs_mhz),
    ("Bandwidth (MHz)", bandwidth_mhz),
    ("Receive-record SNR (dB)", snr_db),
    ("Number of targets", number_targets),
    ("Noise random seed", seed),
]

for name, value in settings:
    writer.writerow(["Radar", name, value])

metrics = [
    ("Range resolution (m)", resolution_m),
    ("Time-bandwidth product", time_bandwidth_product),
    ("Rectangular PSLR (dB)", rect_pslr),
    ("Hamming PSLR (dB)", hamming_pslr),
    ("Direct filtering time (ms)", direct_time_ms),
    ("FFT filtering time (ms)", fft_time_ms),
    ("Hamming FFT time (ms)", hamming_time_ms),
    ("Max direct-vs-FFT difference", max_filter_difference),
]

for name, value in metrics:
    writer.writerow(["Performance", name, value])

for i, distance in enumerate(target_ranges):
    writer.writerow([
        "Configured target",
        f"Target {i + 1} range (m)",
        distance
    ])

    writer.writerow([
        "Configured target",
        f"Target {i + 1} relative amplitude",
        target_amplitudes[i]
    ])

for i, (distance, amplitude) in enumerate(detections):
    writer.writerow([
        "Detected peak",
        f"Detection {i + 1} range (m)",
        distance
    ])

    writer.writerow([
        "Detected peak",
        f"Detection {i + 1} magnitude",
        amplitude
    ])

zip_buffer = io.BytesIO()

graphs = {
    "transmitted_signal.html": tx_fig,
    "received_signal.html": rx_fig,
    "matched_filter_output.html": range_fig,
    "sidelobe_comparison.html": fig_pslr,
    "filter_execution_time.html": fig_time,
}

with zipfile.ZipFile(
    zip_buffer,
    mode="w",
    compression=zipfile.ZIP_DEFLATED
) as archive:

    archive.writestr(
        "results.csv",
        csv_buffer.getvalue()
    )

    for filename, figure in graphs.items():
        html = figure.to_html(
            full_html=True,
            include_plotlyjs="cdn"
        )

        archive.writestr(
            f"graphs/{filename}",
            html
        )

st.download_button(
    label="Download Radar DSP Results (ZIP)",
    data=zip_buffer.getvalue(),
    file_name="radar_dsp_results.zip",
    mime="application/zip"
)


# ---------------------------------------------------------
# 16. DSP theory and limitations
# ---------------------------------------------------------
with st.expander("DSP Theory and Interpretation"):
    st.markdown(
        r"""
        **Range resolution**

        \[
        \Delta R = \frac{c}{2B}
        \]

        **Round-trip delay**

        \[
        \tau = \frac{2R}{c}
        \]

        **Time-bandwidth product**

        \[
        TB = T B
        \]

        FFT convolution calculates linear convolution efficiently
        for sufficiently long signals. Direct convolution is a useful
        reference for comparing numerical accuracy and execution time.

        A Hamming-windowed reference can reduce sidelobes while
        broadening the mainlobe and changing peak response.

        The PSLR shown here is an approximate metric around the
        strongest peak. Other targets may influence the result.

        This educational simulation uses ideal integer-sample delays.
        Fractional delays, Doppler shifts, propagation losses, antenna
        patterns and clutter are not yet modelled.
        """
    )

st.caption(
    "Radar Pulse Compression Lab | Educational DSP Project"
)