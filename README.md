# noscam2
Deepfake Audio Detector - README
📌 Overview
The Deepfake Audio Detector is an AI-powered web application that analyzes audio files to detect AI-generated (deepfake) voices. It uses a TensorFlow/Keras deep learning model trained on mel-spectrogram features to classify audio clips as either Real (authentic) or Fake (AI-generated).
🔍 Key Features:
✅ File Upload & Analysis – Supports WAV, MP3, OGG, and FLAC formats
✅ Real-Time Recording – Record audio directly from your microphone for instant analysis
✅ Deep Learning Model – Uses a CNN-based classifier for high-accuracy detection
✅ Confidence Scoring – Provides a percentage-based confidence level for predictions
✅ Usage Tracking – Free tier allows 3 detections per session (subscription for unlimited)
✅ AI Chatbot Assistant – Get help with usage, formats, and technical questions
✅ Responsive UI – Works on desktop and mobile devices
________________________________________
🚀 Quick Start
Prerequisites
•	Python 3.8+
•	Flask
•	TensorFlow/Keras
•	Librosa (for audio processing)
•	Google Gemini API (for chatbot)
Installation
1.	Clone the repository
bash
git clone https://github.com/your-repo/deepfake-audio-detector.git
cd deepfake-audio-detector
2.	Set up a virtual environment (recommended)
bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows
3.	Install dependencies
bash
pip install -r requirements.txt
4.	Set up environment variables
Create a .env file and add:
env
FLASK_SECRET_KEY=your-secret-key
GEMINI_API_KEY=your-gemini-api-key
5.	Run the Flask app
bash
python main.py
The app will be available at localhost.
________________________________________
🛠 How It Works
1. Audio Processing Pipeline
•	Standardization: Converts all audio files to 16kHz WAV format.
•	Feature Extraction: Extracts mel-spectrograms for deep learning analysis.
•	Model Prediction: Uses a pre-trained CNN model to classify audio as real or fake.
2. Usage Flow
•	Free Users: Limited to 3 detections per session.
•	Subscribed Users: Unlimited detections (RM 9.99/month).
3. AI Chatbot
Powered by Google Gemini, the chatbot helps users with:
•	Supported file formats
•	How to analyze audio
•	Subscription details
•	Technical FAQs
________________________________________
📂 File Structure
text
deepfake-audio-detector/  
├── main.py            # Flask backend (routes, model prediction, chatbot)  
├── templates          # Frontend UI (upload, recording, results)  
  └── subscription.html  
  └── index.html
├── uploads/           # Stores processed audio files  
├── model/             # Contains the trained Keras model  
├── static/            # (Optional) CSS/JS assets  
└── .env               # Environment variables  
________________________________________
🔮 Future Improvements
•	Multi-model Ensemble (improve accuracy)
•	Batch Processing (analyze multiple files at once)
•	API Endpoint (for integration with other apps)
________________________________________
 Ready to Detect Deepfake Audio?
 Run the app now and test your audio files!
