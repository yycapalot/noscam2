# test_model_v2.py

import numpy as np
import librosa
import pickle
import os
import argparse
from tensorflow.keras.models import load_model
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AudioDeepfakeDetector:
    def __init__(self, model_path, scaler_path, encoder_path):
        self.model = load_model(model_path)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

    def load_model_and_scaler(self):
        try:
            self.model = load_model(self.model_path)
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            with open(self.encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading model or scaler: {e}")
            return False

    def extract_advanced_features(self, file_path, max_len=4, sr=22050):
        """Extract a comprehensive set of advanced audio features."""
        try:
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return None

            y, sr = librosa.load(file_path, sr=sr, duration=max_len)
            
            target_length = sr * max_len
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), 'constant')
            else:
                y = y[:target_length]

            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)

            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)

            # Concatenate all features
            features = np.concatenate([
                mfccs,
                mfccs_delta,
                mfccs_delta2,
                chroma,
                contrast,
                tonnetz,
                zcr
            ], axis=0)
            
            return features.T

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def predict(self, file_path):
        """Predict if a single audio file is fake or real."""
        features = self.extract_advanced_features(file_path)
        if features is None:
            return "Error processing file", 0.0

        features_reshaped = features.reshape(-1, features.shape[1])
        features_scaled = self.scaler.transform(features_reshaped).reshape(1, features.shape[0], features.shape[1])

        prediction = self.model.predict(features_scaled)
        predicted_class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_class_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
        
        return str(predicted_class_label), float(confidence)

