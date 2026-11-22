
import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
sampling_rate = 1000
duration = 1

t = np.arange(0, duration, 1 / sampling_rate)

# Generate a signal with 50 Hz and 120 Hz components
signal = (
    np.sin(2 * np.pi * 50 * t)
    + 0.5 * np.sin(2 * np.pi * 120 * t)
)

# Calculate FFT
fft_values = np.fft.rfft(signal)
frequencies = np.fft.rfftfreq(
    len(signal), d=1 / sampling_rate
)
magnitude = np.abs(fft_values) / len(signal)

# Time-domain plot
plt.figure(figsize=(10, 4))
plt.plot(t[:100], signal[:100])
plt.title("Time-Domain Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Frequency-domain plot
plt.figure(figsize=(10, 4))
plt.plot(frequencies, magnitude)
plt.title("Frequency Spectrum using FFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 200)
plt.grid(True)
plt.tight_layout()
plt.show()
