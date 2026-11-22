
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

# ==========================================
# 1. Locate the GTZAN dataset
# ==========================================

project_path = Path(__file__).resolve().parent
dataset_path = project_path.parent / "GTZAN" / "genres_original"

if not dataset_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {dataset_path}"
    )

print("Dataset found:", dataset_path)

# ==========================================
# 2. Explore the dataset
# ==========================================

genres = sorted(
    folder.name
    for folder in dataset_path.iterdir()
    if folder.is_dir()
)

records = []

for genre in genres:
    files = sorted(
        (dataset_path / genre).glob("*.wav")
    )

    records.append({
        "Genre": genre,
        "Number of Tracks": len(files)
    })

df = pd.DataFrame(records)

print("\nGenre Distribution:")
print(df.to_string(index=False))
print("\nTotal tracks:", df["Number of Tracks"].sum())

# ==========================================
# 3. Plot genre distribution
# ==========================================

plt.figure(figsize=(10, 5))

plt.bar(df["Genre"], df["Number of Tracks"])
plt.title("GTZAN Dataset: Tracks per Genre")
plt.xlabel("Music Genre")
plt.ylabel("Number of Tracks")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 4. Load one real audio track
# ==========================================

genre = "disco"

audio_files = sorted(
    (dataset_path / genre).glob("*.wav")
)

if not audio_files:
    raise FileNotFoundError(
        f"No WAV files found for genre: {genre}"
    )

audio_path = audio_files[0]

audio, sr = librosa.load(
    str(audio_path),
    sr=None,
    mono=True
)

duration = len(audio) / sr

print("\nSample audio:", audio_path.name)
print("Genre:", genre)
print("Sampling rate:", sr, "Hz")
print("Number of samples:", len(audio))
print("Duration:", round(duration, 2), "seconds")

# ==========================================
# 5. Plot the real audio waveform
# ==========================================

plt.figure(figsize=(12, 4))

librosa.display.waveshow(audio, sr=sr)

plt.title(f"Waveform of a {genre.title()} Track")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# ==========================================
# 6. Plot the real audio spectrogram
# ==========================================

D = librosa.stft(audio)
S_db = librosa.amplitude_to_db(
    np.abs(D),
    ref=np.max
)

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    S_db,
    sr=sr,
    x_axis="time",
    y_axis="hz"
)

plt.colorbar(format="%+2.0f dB")
plt.title(f"Spectrogram of a {genre.title()} Track")
plt.ylim(0, 5000)
plt.tight_layout()
plt.show()

print("\nEDA completed successfully!")
