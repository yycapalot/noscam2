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
from detector import AudioDeepfakeDetector
from pydub import AudioSegment
import traceback

# Load environment variables
load_dotenv()
my_secret = os.getenv('GEMINI_API_KEY')

deepfake_detector = AudioDeepfakeDetector(
    model_path="model/deepfake_model_v2.keras",
    scaler_path="model/scaler_v2.pkl",
    encoder_path="model/label_encoder_v2.pkl"
)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'webm'}
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

def audio_standardization_uploaded(uploaded_file_path, target_sr=16000):
    try:
        audio_data, original_sr = librosa.load(uploaded_file_path, sr=target_sr)
        print(f"[✓] Loaded audio file: {audio_data}")
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
    
    
def audio_standardization_recorded(recorded_file_path, target_sr=16000):
    try:
        print(f"[MIC] Loading recorded audio: {recorded_file_path}")
        fixed_path = fix_wav_format(recorded_file_path)
        print(f"[MIC] Fixed audio format: {fixed_path}")
        audio_data, sr = librosa.load(fixed_path, sr=target_sr)

        standardized_path = recorded_file_path.replace(".wav", "_mic_standardized.wav")
        sf.write(standardized_path, audio_data, sr, format='WAV')
        return audio_data, sr, standardized_path

    except Exception as e:
        print(f"[✗] Mic standardization failed: {e}")
        return None, None, None
    
def fix_wav_format(input_path):
    audio = AudioSegment.from_file(input_path)
    print(f"[✓] Original audio format: {audio.format}, Sample rate: {audio.frame_rate}")
    fixed_path = input_path.replace(".wav", "_fixed.wav")
    audio.export(fixed_path, format="wav")
    print(f"[✓] Fixed audio format saved to: {fixed_path}")
    return fixed_path



@app.route('/')
def home():
    user_id = get_user_id()
    user_info = user_data.get(user_id, {'usage_count': 0, 'is_subscribed': False})
    return render_template('index.html', user_info=user_info)


@app.route('/upload', methods=['POST'])
def upload_file():
    print("RUnnign upload_file")
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
        print(f"filename: {file.filename}")


        if "recorded_audio" in file.filename.lower():
            audio_data, sample_rate, standardized_path = audio_standardization_recorded(file_path)
        else:
            audio_data, sample_rate, standardized_path = audio_standardization_uploaded(file_path)



        #Call prediction model
        return prediction_model(user_id, filename, sample_rate, standardized_path)
    
    return jsonify({'error': 'Invalid file format. Supported formats: WAV, MP3, OGG, FLAC'}), 400

def prediction_model(user_id, filename, sample_rate, standardized_path):
    if standardized_path:
        print(f"[✓] File uploaded and standardized: {standardized_path}")
        
        # Receive tuple: (label, confidence)
        label, confidence = deepfake_detector.predict(standardized_path)

        if label:
            print(f"[✓] Prediction: {label} with confidence {confidence:.4f}")

            return jsonify({
                'success': True,
                'filename': filename,
                'standardized_file': os.path.basename(standardized_path),
                'sample_rate': sample_rate,
                'usage_count': user_data[user_id]['usage_count'],
                'prediction': label.capitalize(),  # 'Real' or 'Fake'
                'confidence': round(confidence, 4),  # Optional rounding
                'analyzed_on': datetime.now().strftime('%d %b %Y'),
                'user_info': user_data[user_id]
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
    else:
        return jsonify({'error': 'Prediction failed: Prediction model'}), 500



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