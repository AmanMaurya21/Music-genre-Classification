
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

# ==========================================
# 1. Project and dataset paths
# ==========================================

PROJECT_PATH = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_PATH.parent
    / "GTZAN"
    / "genres_original"
)

OUTPUT_PATH = PROJECT_PATH / "data" / "audio_features.csv"

SAMPLE_RATE = 22050
N_MFCC = 13

# ==========================================
# 2. Extract features from one audio file
# ==========================================

def extract_features(file_path):
    """Extract statistical audio features from a WAV file."""

    audio, sr = librosa.load(
        str(file_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    features = {}

    # MFCC: 13 coefficients
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512
    )

    for i in range(N_MFCC):
        features[f"mfcc_{i+1}_mean"] = float(
            np.mean(mfcc[i])
        )
        features[f"mfcc_{i+1}_std"] = float(
            np.std(mfcc[i])
        )

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=2048, hop_length=512
    )[0]

    features["centroid_mean"] = float(np.mean(centroid))
    features["centroid_std"] = float(np.std(centroid))

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, n_fft=2048, hop_length=512
    )[0]

    features["bandwidth_mean"] = float(np.mean(bandwidth))
    features["bandwidth_std"] = float(np.std(bandwidth))

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=2048, hop_length=512
    )[0]

    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # Root mean square energy
    rms = librosa.feature.rms(
        y=audio, frame_length=2048, hop_length=512
    )[0]

    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        roll_percent=0.85
    )[0]

    features["rolloff_mean"] = float(np.mean(rolloff))
    features["rolloff_std"] = float(np.std(rolloff))

    return features


# ==========================================
# 3. Process the complete dataset
# ==========================================

def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    genres = sorted(
        folder
        for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
    )

    audio_files = []

    for genre_folder in genres:
        files = sorted(genre_folder.glob("*.wav"))

        for file_path in files:
            audio_files.append(
                (file_path, genre_folder.name)
            )

    if not audio_files:
        raise FileNotFoundError(
            f"No WAV files found in {DATASET_PATH}"
        )

    print("Genres found:", len(genres))
    print("Audio files found:", len(audio_files))
    print("\nStarting feature extraction...\n")

    records = []
    failed_files = []

    for index, (file_path, genre) in enumerate(
        audio_files, start=1
    ):
        try:
            features = extract_features(file_path)

            features["filename"] = file_path.name
            features["genre"] = genre

            records.append(features)

        except Exception as error:
            failed_files.append(
                (file_path.name, str(error))
            )

        if index % 25 == 0 or index == len(audio_files):
            print(
                f"Processed {index}/{len(audio_files)} "
                f"| Successful: {len(records)} "
                f"| Failed: {len(failed_files)}"
            )

    # Save results
    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    if records:
        df = pd.DataFrame(records)

        # Put identification columns first
        feature_columns = [
            col for col in df.columns
            if col not in ("filename", "genre")
        ]

        df = df[
            ["filename", "genre"] + feature_columns
        ]

        df.to_csv(OUTPUT_PATH, index=False)

        print("\nFeature extraction completed!")
        print("Saved to:", OUTPUT_PATH)
        print("Successful files:", len(df))
        print("Feature columns:", len(feature_columns))
        print("CSV shape:", df.shape)

        print("\nTracks per genre:")
        print(df["genre"].value_counts().sort_index())

    else:
        print("No features were extracted.")

    if failed_files:
        print("\nFailed files:", len(failed_files))

        for filename, error in failed_files[:10]:
            print(f"- {filename}: {error}")


if __name__ == "__main__":
    main()
