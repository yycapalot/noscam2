from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import librosa
import io
import soundfile as sf
import os
import uuid
from datetime import datetime
import json
from werkzeug.utils import secure_filename
from google import genai
from google.genai import types
import numpy as np
from dotenv import load_dotenv 
import tensorflow as tf

# Load environment variables
load_dotenv()
my_secret = os.getenv('GEMINI_API_KEY')

# Load Keras model
try:
    model_path = os.path.join("model", "mel_cnn_model.h5")
    model = tf.keras.models.load_model(model_path)
    print(f"[✓] Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"[✗] Error loading model: {e}")
    model = None

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

user_data = {}
analytics_data = {
    'total_uploads': 0,
    'files_processed': [],
    'subscription_conversions': 0,
    'daily_usage': {}
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

def increment_usage(user_id):
    if user_id not in user_data:
        user_data[user_id] = {'usage_count': 0, 'is_subscribed': False}
    if not user_data[user_id]['is_subscribed']:
        user_data[user_id]['usage_count'] += 1
    return user_data[user_id]['usage_count']

def audio_standardization(uploaded_file_path, target_sr=16000):
    try:
        audio_data, original_sr = librosa.load(uploaded_file_path, sr=target_sr)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, target_sr, format='WAV')
        wav_buffer.seek(0)

        original_name = os.path.splitext(os.path.basename(uploaded_file_path))[0]
        new_filename = f"{original_name}_standardized.wav"
        standardized_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

        with open(standardized_path, "wb") as f:
            f.write(wav_buffer.read())

        return audio_data, target_sr, standardized_path
    except Exception as e:
        print(f"[✗] Error processing audio: {e}")
        return None, None, None

def extract_mel_spectrogram(file_path, sr=16000, n_mels=128, duration=5):
    try:
        print(f"[✓] Extracting mel spectrogram from: {file_path}")
        
        audio, _ = librosa.load(file_path, sr=sr)
        audio = librosa.util.fix_length(audio, size=sr * duration)

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_db = (mel_db - np.min(mel_db)) / (np.max(mel_db) - np.min(mel_db))
        mel_db = np.expand_dims(mel_db, axis=-1)
        mel_db = np.expand_dims(mel_db, axis=0)

        return mel_db
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

@app.route('/')
def home():
    user_id = get_user_id()
    user_info = user_data.get(user_id, {'usage_count': 0, 'is_subscribed': False})
    return render_template('index.html', user_info=user_info)


@app.route('/upload', methods=['POST'])
def upload_file():
    user_id = get_user_id()
    if user_id not in user_data:
        user_data[user_id] = {'usage_count': 0, 'is_subscribed': False}

    if not user_data[user_id]['is_subscribed'] and user_data[user_id]['usage_count'] >= 3:
        return jsonify({'error': 'subscription_required', 'usage_count': user_data[user_id]['usage_count']})

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file and allowed_file(file.filename):
        if not user_data[user_id]['is_subscribed']:
            user_data[user_id]['usage_count'] += 1

        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        audio_data, sample_rate, standardized_path = audio_standardization(file_path)

    if standardized_path:
        features = extract_mel_spectrogram(standardized_path)

        if features is not None and model:
            try:
                if features.shape[2] != 128:
                    features = tf.image.resize(features[0], (128, 128))
                    features = tf.expand_dims(features, axis=0)

                prediction = model.predict(features)
                label = 'Fake' if prediction[0][0] > 0.5 else 'Real'

                response_data = {
                    'success': True,
                    'filename': filename,
                    'standardized_file': os.path.basename(standardized_path),
                    'sample_rate': sample_rate,
                    'usage_count': user_data[user_id]['usage_count'],
                    'prediction': label,
                    'confidence': float(prediction[0][0])
                }

                print(f"[✓] Prediction: {label} with confidence {prediction[0][0]}")
                print(f"[✓] File processed: {response_data}")
                return jsonify(response_data)

            except Exception as e:
                return jsonify({'error': f'Prediction error: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Failed to extract features or model not loaded'}), 400
    else:
        return jsonify({'error': 'Failed to standardize audio file'}), 400


@app.route('/subscribe')
def subscribe():
    return render_template('subscription.html')

@app.route('/process_subscription', methods=['POST'])
def process_subscription():
    user_id = get_user_id()
    if user_id not in user_data:
        user_data[user_id] = {'usage_count': 0, 'is_subscribed': False}

    user_data[user_id]['is_subscribed'] = True
    analytics_data['subscription_conversions'] += 1
    flash('Subscription successful! You now have unlimited access.', 'success')
    return redirect(url_for('home'))

@app.route('/api/analytics')
def api_analytics():
    return jsonify(analytics_data)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        client = genai.Client(api_key=my_secret)

        system_prompt = """You are a helpful AI assistant for a Deepfake Audio Detector application.
        You help users with:
        - How to upload and analyze audio files
        - Supported file formats (WAV, MP3, OGG, FLAC)
        - Explaining the audio standardization process (converting to 16kHz WAV)
        - Subscription plans and pricing ($9.99/month for unlimited usage)
        - Technical questions about deepfake detection
        - General usage questions about the app

        Be concise, friendly, and helpful. Focus on the audio detection features of the app."""

        full_prompt = system_prompt + "\n\nUser question: " + user_message

        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=[full_prompt]
        )

        bot_response = response.text
        return jsonify({'response': bot_response})

    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'response': 'Sorry, I encountered an error. Please try again later.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
