import streamlit as st
import sounddevice as sd
import numpy as np
import wavio
import os
import tempfile
import httpx
import io

# Initialize Groq API key
GROQ_API_KEY = st.secrets["groq_api_key"]

# Set up Streamlit page configuration
st.set_page_config(page_title="Audio Recorder and Transcriber", page_icon="🎤")
st.title("Audio Recorder and Transcriber")

# Audio recording parameters
SAMPLE_RATE = 44100
CHANNELS = 1

def record_audio(duration, samplerate=SAMPLE_RATE, channels=CHANNELS):
    """Record audio for a specified duration."""
    recording = sd.rec(int(duration * samplerate),
                      samplerate=samplerate,
                      channels=channels,
                      dtype='float32')
    st.info("Recording...")
    sd.wait()
    return recording

def process_audio(audio_content: bytes, model: str = "whisper-large-v3"):
    """Process audio content using Groq's Whisper API."""
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    # Prepare audio file
    audio_file = io.BytesIO(audio_content)
    audio_file.name = "audio.wav"

    files = {"file": audio_file}
    data = {
        "model": model,
        "response_format": "verbose_json"
    }

    # Create two transcription requests, one for English and one for Hindi
    languages = ["en", "hi"]
    transcriptions = {}

    for lang in languages:
        data["language"] = lang
        with httpx.Client() as client:
            response = client.post(
                url, headers=headers, files=files, data=data, timeout=None
            )
            response.raise_for_status()
            transcriptions[lang] = response.json()["text"]

    return transcriptions

# Session state for recording status
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

# Create the main interface
col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Recording duration (seconds)", min_value=1, max_value=30, value=5)

with col2:
    if st.button("Start Recording"):
        st.session_state.audio_data = record_audio(duration)
        st.session_state.recording = True

# Process and display the recording
if st.session_state.recording and st.session_state.audio_data is not None:
    # Create a temporary directory to store the audio file
    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, "recorded_audio.wav")
    
    # Save as WAV file
    wavio.write(temp_audio_path, st.session_state.audio_data, SAMPLE_RATE, sampwidth=3)
    
    # Add a playback option
    st.audio(temp_audio_path)
    
    # Transcribe with Groq
    with st.spinner("Transcribing audio in English and Hindi..."):
        try:
            with open(temp_audio_path, "rb") as file:
                audio_content = file.read()
                transcriptions = process_audio(audio_content)
                
            # Display transcriptions
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### English Transcription:")
                st.write(transcriptions["en"])
                
            with col2:
                st.write("### Hindi Transcription:")
                st.write(transcriptions["hi"])
                
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
    
    # Cleanup temporary files
    try:
        os.remove(temp_audio_path)
        os.rmdir(temp_dir)
    except:
        pass
    
    # Reset recording state
    st.session_state.recording = False
    st.session_state.audio_data = None
