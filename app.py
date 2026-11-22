from pathlib import Path
import tempfile

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import butter, lfilter


# ==========================================
# 1. Configuration & Setup
# ==========================================

PROJECT_PATH = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_PATH / "models" / "music_genre_model.joblib"

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

# --- Digital Filter Helpers (Signals & Systems) ---
def apply_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# ==========================================
# 2. Feature extraction
# ==========================================

def extract_features(file_path):
    audio, sr = librosa.load(str(file_path), sr=22050, mono=True)
    if len(audio) == 0:
        raise ValueError("The audio file is empty.")
    return extract_features_from_audio(audio, sr), audio, sr

def extract_features_from_audio(audio, sr):
    """Extract the same 36 features used during training."""
    features = {}

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=2048, hop_length=512)[0]
    features["centroid_mean"] = float(np.mean(centroid))
    features["centroid_std"] = float(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=2048, hop_length=512)[0]
    features["bandwidth_mean"] = float(np.mean(bandwidth))
    features["bandwidth_std"] = float(np.std(bandwidth))

    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=2048, hop_length=512)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=2048, hop_length=512, roll_percent=0.85)[0]
    features["rolloff_mean"] = float(np.mean(rolloff))
    features["rolloff_std"] = float(np.std(rolloff))

    return features


# ==========================================
# 3. Application interface
# ==========================================

st.title("🎵 Music Genre Classification")
st.write("Upload a music track to analyze its audio signal and predict its genre using Machine Learning.")
st.caption("Data Science + Signals and Systems | GTZAN Dataset | Random Forest")

# --- Interactive Sidebar for Signals & Systems ---
st.sidebar.header("🎛️ Signal Processing")
st.sidebar.write("Apply a digital filter before feature extraction to observe how frequency alteration impacts ML predictions.")
apply_filter = st.sidebar.checkbox("Apply Low-Pass Filter")
cutoff_freq = st.sidebar.slider(
    "Cutoff Frequency (Hz)", 
    min_value=500, max_value=10000, value=3000, step=500, 
    disabled=not apply_filter,
    help="Frequencies above this threshold will be attenuated."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Please run train_model.py first.")
    st.stop()

uploaded_file = st.file_uploader("Upload a music file", type=["wav", "mp3", "flac", "ogg", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("Analyze Music", type="primary"):
        try:
            suffix = Path(uploaded_file.name).suffix.lower() or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = Path(temp_file.name)

            try:
                with st.spinner("Loading and processing audio signal..."):
                    audio, sr = librosa.load(str(temp_path), sr=22050, mono=True)
                    
                    # --- Apply Selected Digital Filter ---
                    if apply_filter:
                        audio = apply_lowpass_filter(audio, cutoff_freq, sr)
            finally:
                temp_path.unlink(missing_ok=True)

            if len(audio) == 0:
                raise ValueError("The audio file is empty.")

            bundle = joblib.load(MODEL_PATH)
            model = bundle["model"]
            feature_columns = bundle["feature_columns"]

            segment_seconds = 30
            segment_samples = segment_seconds * sr
            segment_results = []

            with st.spinner("Predicting segments..."):
                for start in range(0, len(audio), segment_samples):
                    end = min(start + segment_samples, len(audio))
                    segment = audio[start:end]

                    if len(segment) < sr * 5:
                        continue

                    features = extract_features_from_audio(segment, sr)
                    input_df = pd.DataFrame([features]).reindex(columns=feature_columns)

                    if input_df.isnull().any().any():
                        raise ValueError("Extracted features do not match the trained model.")

                    genre = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0]

                    segment_results.append({
                        "Segment": f"{start / sr:.0f}–{end / sr:.0f} sec",
                        "Genre": str(genre).title(),
                        "Score": float(np.max(probabilities)),
                        "Probabilities": probabilities
                    })

            if not segment_results:
                raise ValueError("The audio is too short to analyze.")

            # ======================================
            # Dashboard Display
            # ======================================
            
            st.divider()
            
            # --- Combined Prediction & Calibration ---
            all_probabilities = np.array([item["Probabilities"] for item in segment_results])
            mean_probabilities = np.mean(all_probabilities, axis=0)
            
            final_index = int(np.argmax(mean_probabilities))
            final_genre = str(model.classes_[final_index]).title()

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Genre", final_genre)
            col2.metric("Mean Prediction Confidence", f"{mean_probabilities[final_index]:.1%}")
            col3.metric("Segments Analyzed", len(segment_results))

            st.info(
                "**Data Science Note on Confidence:** A low confidence score (e.g., < 30%) indicates model uncertainty. "
                "This frequently occurs when a song contains mixed musical textures, changing tempos, or characteristics "
                "that overlap multiple genres across its segments."
            )

            # --- Dynamic Genre Probabilities Over Time (Data Science) ---
            st.subheader("📊 Dynamic Genre Probabilities Over Time")
            prob_df = pd.DataFrame(all_probabilities, columns=[str(c).title() for c in model.classes_])
            prob_df.index = [item["Segment"] for item in segment_results]
            st.area_chart(prob_df, use_container_width=True)

            # --- Segment Results Table ---
            with st.expander("View Raw Segment Predictions"):
                results_df = pd.DataFrame([
                    {"Segment": item["Segment"], "Predicted Genre": item["Genre"], "Confidence": f'{item["Score"]:.1%}'}
                    for item in segment_results
                ])
                st.dataframe(results_df, use_container_width=True, hide_index=True)

            # --- Feature Importances (Data Science) ---
            if hasattr(model, "feature_importances_"):
                st.subheader("🔍 Top ML Feature Importances")
                feat_df = pd.DataFrame({
                    "Feature": feature_columns, 
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=False).head(10)
                st.bar_chart(feat_df.set_index("Feature"), use_container_width=True)

            # --- Waveform ---
            st.divider()
            st.subheader("📈 Time-Domain Signal (Waveform)")
            fig, ax = plt.subplots(figsize=(12, 3))
            librosa.display.waveshow(audio, sr=sr, ax=ax)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # --- Spectrogram with Centroid Overlay (Signals & Systems) ---
            st.subheader("🎼 Frequency Content (Spectrogram) & Spectral Centroid")
            
            D = librosa.stft(audio, n_fft=2048, hop_length=512)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            # Calculate spectral centroid for the whole track to overlay
            cent = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=2048, hop_length=512)[0]
            times = librosa.times_like(cent, sr=sr, hop_length=512)

            fig, ax = plt.subplots(figsize=(12, 4))
            image = librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="hz", ax=ax)
            
            # Plot the overlay
            ax.plot(times, cent, label='Spectral Centroid (Brightness)', color='white', linewidth=1.5, alpha=0.8)
            ax.legend(loc='upper right')
            
            ax.set_ylim(0, min(5000, sr / 2))
            fig.colorbar(image, ax=ax, format="%+2.0f dB")
            fig.tight_layout()
            
            st.pyplot(fig)
            plt.close(fig)
            # ======================================
            # Advanced Signal Processing (New Block)
            # ======================================
            
            st.divider()
            st.subheader("🎛️ Advanced Signal Deconstruction")
            
            # 1. Harmonic-Percussive Source Separation (HPSS)
            st.write("**1. Harmonic vs. Percussive Components:** Splitting the raw waveform into tonal sounds (melody/chords) and transient sounds (drums/beats).")
            
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            
            fig_hpss, ax_hpss = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(12, 5))
            
            librosa.display.waveshow(y_harmonic, sr=sr, ax=ax_hpss[0], color='dodgerblue')
            ax_hpss[0].set_title('Harmonic Signal (Melody & Chords)')
            ax_hpss[0].set_ylabel("Amplitude")
            
            librosa.display.waveshow(y_percussive, sr=sr, ax=ax_hpss[1], color='tomato')
            ax_hpss[1].set_title('Percussive Signal (Drums & Beats)')
            ax_hpss[1].set_ylabel("Amplitude")
            ax_hpss[1].set_xlabel("Time (seconds)")
            
            fig_hpss.tight_layout()
            st.pyplot(fig_hpss)
            plt.close(fig_hpss)
            
            # 2. Chromagram
            st.write("**2. Chromagram (Pitch Class Profile):** Folding the frequency spectrum into the 12 standard musical pitch classes to show the harmonic content of the song over time.")
            
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=2048, hop_length=512)
            
            fig_chroma, ax_chroma = plt.subplots(figsize=(12, 4))
            img = librosa.display.specshow(
                chroma, y_axis='chroma', x_axis='time', ax=ax_chroma, cmap='coolwarm', hop_length=512
            )
            ax_chroma.set_title('Chromagram')
            fig_chroma.colorbar(img, ax=ax_chroma, label="Intensity")
            
            fig_chroma.tight_layout()
            st.pyplot(fig_chroma)
            plt.close(fig_chroma)

        except Exception as error:
            st.error(f"Could not analyze audio: {error}")