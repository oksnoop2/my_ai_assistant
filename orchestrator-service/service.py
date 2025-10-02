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
    RESOURCE_MANAGER_URL = os.environ["RESOURCE_MANAGER_URL"]
    ASR_SERVICE_URL = os.environ["ASR_SERVICE_URL"]
    TTS_SERVICE_URL = os.environ["TTS_SERVICE_URL"]
    RAG_SERVICE_URL = os.environ["RAG_SERVICE_URL"]
    LLM_SERVICE_URL = os.environ["LLM_SERVICE_URL"] 
except KeyError as e:
    print(f"🔥 Critical environment variable missing: {e}", file=sys.stderr)
    sys.exit(1)
RECORD_SECONDS = 5
RECORD_SAMPLE_RATE = 48000 # This will be ignored by arecord but kept for consistency
WHISPER_SAMPLE_RATE = 16000
PLAYBACK_SAMPLE_RATE = 48000
SLEEP_SEC = 2.0

def wait_for_health(service_name: str, service_url: str, timeout_sec: int = 180) -> bool:
    """
    Waits for a service to become healthy by polling its /health endpoint.

    Args:
        service_name: The human-readable name of the service.
        service_url: The base URL of the service.
        timeout_sec: The maximum time to wait in seconds.

    Returns:
        True if the service becomes healthy, False otherwise.
    """
    start_time = time.time()
    health_url = f"{service_url}/health"
    while time.time() - start_time < timeout_sec:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                # Some services have a secondary check
                if "index_ready" in response.json() and not response.json()["index_ready"]:
                     print(f"⏳ Waiting for {service_name} index to be ready...")
                else:
                    print(f"✅ {service_name} is healthy at {health_url}")
                    return True
        except requests.RequestException:
            # This is expected if the service isn't up yet
            pass
        except Exception as e:
            print(f"An unexpected error occurred while checking {service_name}: {e}")

        time.sleep(3) # Wait before retrying

    print(f"🔥 Timed out waiting for {service_name} at {health_url}", file=sys.stderr)
    return False

# ---- Core Functions ----

def record_audio() -> Optional[io.BytesIO]:
    """Records audio using sounddevice, resampling for Whisper."""
    print(f"⏺ Recording for {RECORD_SECONDS} seconds at {RECORD_SAMPLE_RATE}Hz...")
    try:
        recording = sd.rec(
            int(RECORD_SECONDS * RECORD_SAMPLE_RATE),
            samplerate=RECORD_SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()

        print("🎤 Resampling audio to 16000Hz for ASR...")
        num_samples = int(len(recording) * WHISPER_SAMPLE_RATE / RECORD_SAMPLE_RATE)
        resampled_recording = resample_audio(recording, num_samples)

        buf = io.BytesIO()
        write(buf, WHISPER_SAMPLE_RATE, (resampled_recording * 32767).astype(np.int16))
        buf.seek(0)
        
        print("✅ Recording complete.")
        return buf
        
    except Exception as e:
        print(f"⚠ Audio recording failed: {e}", file=sys.stderr)
        # We can add the traceback back in for good measure if this fails.
        import traceback
        traceback.print_exc()
        return None

def tts_play_wav_bytes(wav_bytes: bytes):
    """Resamples and plays WAV audio bytes using sounddevice."""
    try:
        buf = io.BytesIO(wav_bytes)
        original_sr, audio = read(buf)
        
        print(f"🔊 Resampling TTS audio from {original_sr}Hz to {PLAYBACK_SAMPLE_RATE}Hz...")
        audio_float = audio.astype(np.float32) / 32768.0
        num_samples = int(len(audio_float) * PLAYBACK_SAMPLE_RATE / original_sr)
        audio_resampled = resample_audio(audio_float, num_samples)
        
        sd.play(audio_resampled, PLAYBACK_SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"⚠ TTS playback failed: {e}", file=sys.stderr)

def transcribe_audio(wav_obj: Optional[io.BytesIO]):
    if not wav_obj:
        return None
        
    try:
        with open("/voices/last_recording.wav", "wb") as f:
            wav_obj.seek(0)
            f.write(wav_obj.read())
        print("🎤 Saved recording to /voices/last_recording.wav for inspection.")
    except Exception as e:
        print(f"🔥 Failed to save debug audio file: {e}", file=sys.stderr)
    
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
    if not prompt_text:
        return None
    
    print("ORCHESTRATOR: Sending question to RAG Service...")
    try:
        response = requests.post(f"{RAG_SERVICE_URL}/query", json={"input_text": prompt_text}, timeout=300)
        response.raise_for_status()
        return response.json().get("response")
    except Exception as e:
        print(f"❌ RAG system request failed: {e}", file=sys.stderr)
        return "Sorry, I had a problem processing that request."

def speak_text(text: str, voice_path="/voices/my_voice.wav"):
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
