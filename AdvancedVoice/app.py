from flask import Flask, request, jsonify, render_template
import os
import io
import sys
from faster_whisper import WhisperModel
import edge_tts
import ollama
import time

UPLOAD_FOLDER = 'static'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
    
TEMP_AUDIO = os.path.join(UPLOAD_FOLDER, "temp_audio.mp3")
VOICE = "en-US-SteffanNeural"

model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2")

def transcribe_audio(audio_path):
    segments, _ = model.transcribe(audio_path, beam_size=1)
    transcription = " ".join([segment.text for segment in segments])
    return transcription.strip()

def generate_llm_content(prompt):
    try:
        response = ollama.chat(model='llama3.2:1b', messages=[
        {'role': 'system', 'content': "You are a quick, helpful Assistant powered by Voice. Provide concise answers to ensure fast TTS responses."},
        {'role': 'user', 'content': prompt}
        ])
        
        if 'message' in response and 'content' in response['message']:
            return response['message']['content'].strip()
        else:
            print("Unexpected response format:", response)
            return "Error generating response."
    
    except Exception as e:
        print(f"LLM Generation error: {e}")
        return "Error generating response."

def generate_tts(text):
    try:
        stdout_backup = sys.stdout
        stderr_backup = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        if os.path.exists(TEMP_AUDIO):
            os.remove(TEMP_AUDIO)
            
        communicate = edge_tts.Communicate(text, VOICE, rate="+16%", pitch="-25Hz")
        communicate.save_sync(TEMP_AUDIO)
        sys.stdout = stdout_backup
        sys.stderr = stderr_backup
        return TEMP_AUDIO
    except Exception as e:
        print(f"TTS Generation error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    audio_path = os.path.join(UPLOAD_FOLDER, "input.wav")
    audio_file.save(audio_path)

    user_text = transcribe_audio(audio_path)

    ai_response = generate_llm_content(user_text)

    audio_path = generate_tts(ai_response)
    if not audio_path:
        return jsonify({"error": "Failed to generate TTS"}), 500

    timestamp = int(time.time())
    return jsonify({"response": ai_response, "audio_url": f"/{audio_path}?t={timestamp}"})

if __name__ == '__main__':
    app.run(debug=True)
