
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# ==========================================
# 1. Generate a test audio signal
# ==========================================

sampling_rate = 22050  # Samples per second
duration = 3            # Duration in seconds

t = np.arange(0, duration, 1 / sampling_rate)

# Combine two sine waves: 440 Hz and 880 Hz
audio = (
    0.6 * np.sin(2 * np.pi * 440 * t)
    + 0.3 * np.sin(2 * np.pi * 880 * t)
)

print("Audio signal generated successfully!")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Duration: {duration} seconds")
print(f"Number of samples: {len(audio)}")


# ==========================================
# 2. Plot the time-domain waveform
# ==========================================

# Display only the first 10 milliseconds
samples_to_plot = int(0.01 * sampling_rate)

plt.figure(figsize=(12, 4))

plt.plot(
    t[:samples_to_plot],
    audio[:samples_to_plot]
)

plt.title("Audio Waveform (First 10 ms)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================
# 3. Calculate the Short-Time Fourier Transform
# ==========================================

D = librosa.stft(audio)

# Convert magnitude to decibels
S_db = librosa.amplitude_to_db(
    np.abs(D),
    ref=np.max
)


# ==========================================
# 4. Plot the audio spectrogram
# ==========================================

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    S_db,
    sr=sampling_rate,
    x_axis="time",
    y_axis="hz"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Audio Spectrogram")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.ylim(0, 2000)
plt.tight_layout()
plt.show()


# ==========================================
# 5. Completion message
# ==========================================

print("\nAudio signal analysis completed successfully!")
