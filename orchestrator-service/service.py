# orchestrator-service/service.py
import os
import io
import sys
import time
import tempfile
import subprocess
from typing import Optional
import requests
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read
from scipy.signal import resample as resample_audio

# ---- Service URLs and Configuration ----
try:
    # These now correctly point to the internal container names and ports
    RESOURCE_MANAGER_URL = os.environ["RESOURCE_MANAGER_URL"]
    ASR_SERVICE_URL = os.environ["ASR_SERVICE_URL"]
    TTS_SERVICE_URL = os.environ["TTS_SERVICE_URL"]
    RAG_SERVICE_URL = os.environ["RAG_SERVICE_URL"]
    LLM_SERVICE_URL = os.environ["LLM_SERVICE_URL"] 
    
except KeyError as e:
    print(f"🔥 Critical environment variable missing: {e}", file=sys.stderr)
    sys.exit(1)
RECORD_SECONDS = 5
RECORD_SAMPLE_RATE = 48000
WHISPER_SAMPLE_RATE = 16000
PLAYBACK_SAMPLE_RATE = 48000
SLEEP_SEC = 2.0

# ---- Core Functions ----

    
def record_audio() -> Optional[io.BytesIO]:
    """Records at a high sample rate and resamples down for Whisper."""
    print(f"⏺ Recording for {RECORD_SECONDS} seconds at {RECORD_SAMPLE_RATE}Hz...")
    try:
        # Record at the hardware-friendly sample rate
        recording = sd.rec(
            int(RECORD_SECONDS * RECORD_SAMPLE_RATE),
            samplerate=RECORD_SAMPLE_RATE,
            channels=1,
            dtype='int16'
        )
        sd.wait()

        print("🎤 Resampling audio to 16000Hz for ASR...")
        # Calculate the number of samples in the new audio
        num_samples = int(len(recording) * WHISPER_SAMPLE_RATE / RECORD_SAMPLE_RATE)
        # Resample the audio
        resampled_recording = resample_audio(recording, num_samples).astype(np.int16)

        # Convert the RESAMPLED NumPy array to an in-memory WAV file
        buf = io.BytesIO()
        write(buf, WHISPER_SAMPLE_RATE, resampled_recording) # Save with the correct Whisper sample rate
        buf.seek(0)
        print("✅ Recording complete and resampled.")
        return buf
    except Exception as e:
        print(f"⚠ Audio recording failed: {e}", file=sys.stderr)
        return None

  
def tts_play_wav_bytes(wav_bytes: bytes):
    """Resamples and plays WAV audio bytes using sounddevice."""
    try:
        buf = io.BytesIO(wav_bytes)
        original_sr, audio = read(buf)
        
        print(f"🔊 Resampling TTS audio from {original_sr}Hz to {PLAYBACK_SAMPLE_RATE}Hz...")
        audio_float = audio.astype(np.float32) / 32768.0
        num_samples = int(len(audio_float) * PLAYBACK_SAMPLE_RATE / original_sr)
        audio_resampled = resample(audio_float, num_samples)
        
        sd.play(audio_resampled, PLAYBACK_SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"⚠ TTS playback failed: {e}", file=sys.stderr)

def wait_for_health(name: str, url: str, path="/health", retries=10):
    """Checks the health endpoint of a service, retrying on failure."""
    full_url = f"{url}{path}"
    print(f"🩺 Checking health of {name} at {full_url}...")
    for i in range(1, retries + 1):
        try:
            r = requests.get(full_url, timeout=2)
            r.raise_for_status()
            print(f"✅ {name} healthy.")
            return True
        except Exception:
            print(f"⏳ waiting for {name} ({i}/{retries})...")
            time.sleep(SLEEP_SEC)
    print(f"❌ {name} not healthy after {retries} retries.", file=sys.stderr)
    return False

def transcribe_audio(wav_obj: Optional[io.BytesIO]):
    """Sends audio data to the ASR service and returns the transcribed text."""
    if not wav_obj:
        return None
        
    # --- START DEBUGGING CHANGE ---
    # Save the recorded audio to a file we can inspect.
    # We'll save it in the tts voices directory as it's an easy volume to access.
    try:
        with open("/voices/last_recording.wav", "wb") as f:
            wav_obj.seek(0)
            f.write(wav_obj.read())
        print("🎤 Saved recording to /voices/last_recording.wav for inspection.")
    except Exception as e:
        print(f"🔥 Failed to save debug audio file: {e}", file=sys.stderr)
    # --- END DEBUGGING CHANGE ---
    
    print("📝 Transcribing audio via ASR...")
    try:
        wav_obj.seek(0)
        files = {'audio_file': ('rec.wav', wav_obj.read(), 'audio/wav')}
        r = requests.post(f"{ASR_SERVICE_URL}/asr", files=files, timeout=30)
        r.raise_for_status()
        text = r.json().get("text", "").strip()
        print(f"🗣 ASR → '{text}'")
        return text
    except Exception as e:
        print(f"❌ ASR request failed: {e}", file=sys.stderr)
        return None

def query_rag_system(prompt_text: str):
    """
    Sends the user's input to the RAG Service and gets a synthesized answer.
    """
    if not prompt_text:
        return None
    
    print("ORCHESTRATOR: Sending question to RAG Service...")
    try:
        response = requests.post(f"{RAG_SERVICE_URL}/query", json={"input_text": prompt_text}, timeout=300) # Long timeout
        response.raise_for_status()
        return response.json().get("response")
    except Exception as e:
        print(f"❌ RAG system request failed: {e}", file=sys.stderr)
        return "Sorry, I had a problem processing that request."

def speak_text(text: str, voice_path="/voices/my_voice.wav"):
    """
    Ensures the TTS model is loaded via the Resource Manager, then synthesizes speech.
    """
    if not text:
        return
    try:
        print("🗣️  Requesting TTS model from Resource Manager...")
        requests.post(f"{RESOURCE_MANAGER_URL}/request_model", json={"model_name": "tts"}, timeout=180).raise_for_status()
        print("✅ TTS model is ready in RAM.")
        
        print(f"🔊 Synthesizing: '{text}'")
        r = requests.post(f"{TTS_SERVICE_URL}/api/tts", json={"text": text, "speaker_wav": voice_path}, timeout=120)
        r.raise_for_status()
        tts_play_wav_bytes(r.content)
    except requests.RequestException as e:
        print(f"❌ TTS workflow failed: {e}", file=sys.stderr)
    except Exception as e:
        print(f"❌ An unexpected error occurred during TTS: {e}", file=sys.stderr)

def main():
    """Main application loop."""
    print("🚀 Orchestrator starting.")
    # Add the resource manager to the health checks
    if not all([
        wait_for_health("Resource Manager", RESOURCE_MANAGER_URL),
        wait_for_health("ASR Service", ASR_SERVICE_URL),
        wait_for_health("TTS Service", TTS_SERVICE_URL),
        wait_for_health("RAG Service", RAG_SERVICE_URL),
        wait_for_health("LLM Service", LLM_SERVICE_URL)
    ]):
        print("🔥 One or more critical services are not healthy. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    print("✅ All services are responsive. Ready.")
    
    while True:
        try:
            input("\nPress Enter to record (or Ctrl+C to quit)...")
            wav_data = record_audio()
            user_text = transcribe_audio(wav_data)
            
            if not user_text:
                print("…no input, try again.")
                continue
                
            reply = query_rag_system(user_text)
            print(f"🤖 LLM → '{reply}'")
            speak_text(reply)
            
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
        except Exception as e:
            print(f"⚠ An unexpected error occurred in the main loop: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
