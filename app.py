import streamlit as st
import os
import time
from pydub import AudioSegment
import tempfile
import pyperclip
import requests
import json

# Add API key to environment if provided
if 'GROQ_API_KEY' not in os.environ:
    os.environ['GROQ_API_KEY'] = st.secrets["groq_api_key"]

def get_audio_info(input_file):
    audio = AudioSegment.from_file(input_file)
    duration = len(audio) / 1000  # Duration in seconds
    return duration

def calculate_bitrate(duration, target_size):
    bitrate = (target_size * 8) / (1.048576 * duration)
    return int(bitrate)

def compress_audio(input_file, bitrate):
    audio = AudioSegment.from_file(input_file)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    audio.export(temp_file.name, format='mp3', bitrate=f'{bitrate}k')
    return temp_file.name

def is_valid_audio_format(filename):
    valid_formats = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']
    _, extension = os.path.splitext(filename)
    return extension.lower() in valid_formats

def transcribe_audio_groq(input_file):
    try:
        headers = {
            "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}"
        }
        
        start_time = time.time()
        
        with open(input_file, 'rb') as f:
            files = {
                'file': (os.path.basename(input_file), f, 'audio/mpeg'),
                'model': (None, 'whisper-large-v3-turbo'),
                'response_format': (None, 'verbose_json')
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                files=files
            )
            
        end_time = time.time()
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
            
        result = response.json()
        return result.get('text', ''), end_time - start_time
    except Exception as e:
        if 'api_key' in str(e).lower():
            raise Exception("Groq API key not set. Please set the GROQ_API_KEY environment variable.")
        raise e

def save_transcript_to_file(transcript, filename):
    try:
        with open(filename, "w") as f:
            f.write(transcript)
        return True
    except Exception as e:
        st.error(f"Failed to save transcript: {str(e)}")
        return False

def main():
    st.title("Whisper Web UI")

    # Use session state to store the transcript and user confirmation
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    if 'transcription_time' not in st.session_state:
        st.session_state.transcription_time = 0

    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/mp3")

        if st.button("Process Audio"):
            with st.spinner("Processing audio..."):
                # Save uploaded file temporarily
                temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
                temp_input.write(uploaded_file.getvalue())
                temp_input.close()

                file_size = os.path.getsize(temp_input.name) / (1024 * 1024)  # File size in MB
                st.write(f"Input file size: {file_size:.2f} MB")

                if file_size > 25:
                    st.write("File size exceeds 25MB. Compressing...")
                    target_size = 24.9 * 1024  # Target size in kilobytes (just under 25MB)
                    duration = get_audio_info(temp_input.name)
                    bitrate = calculate_bitrate(duration, target_size)
                    compressed_file = compress_audio(temp_input.name, bitrate)
                    input_file = compressed_file
                    st.write("Compression complete.")
                else:
                    st.write("File size is within the allowed limit. No compression needed.")
                    input_file = temp_input.name

                st.write("Transcribing audio using Groq API...")
                try:
                    st.session_state.transcript, st.session_state.transcription_time = transcribe_audio_groq(input_file)
                    st.write(f"Transcription complete! Time taken: {st.session_state.transcription_time:.2f} seconds")
                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")

                # Cleanup temporary files
                os.unlink(temp_input.name)
                if file_size > 25:
                    os.unlink(compressed_file)

        # Display transcript and controls
        if st.session_state.transcript:
            st.subheader("Transcript:")
            st.text_area("", value=st.session_state.transcript, height=300)

            if st.button("Copy to Clipboard"):
                try:
                    pyperclip.copy(st.session_state.transcript)
                    st.success("Transcript copied to clipboard!")
                except Exception as e:
                    st.error(f"Failed to copy to clipboard: {str(e)}")

            st.subheader("Save Transcript to File")
            output_filename = st.text_input("Enter output filename for transcript:")
            if st.button("Save Transcript"):
                if output_filename:
                    if save_transcript_to_file(st.session_state.transcript, output_filename):
                        st.success(f"Transcript saved to {output_filename}")
                else:
                    st.warning("Please enter a filename to save the transcript.")

if __name__ == '__main__':
    main()

