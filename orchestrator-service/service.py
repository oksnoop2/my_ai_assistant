# orchestrator-service/service.py (Final Version with Memory Loop)
import os
import io
import sys
import time
import requests
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read
from scipy.signal import resample as resample_audio
from typing import Optional
import threading # <-- NEW: Import threading
import re
import logging

# ---- Service URLs and Configuration ----
try:
    # (The environment variables section remains the same)
    RESOURCE_MANAGER_URL = os.environ["RESOURCE_MANAGER_URL"]
    ASR_SERVICE_URL = os.environ["ASR_SERVICE_URL"]
    TTS_SERVICE_URL = os.environ["TTS_SERVICE_URL"]
    RAG_SERVICE_URL = os.environ["RAG_SERVICE_URL"]
    LLM_SERVICE_URL = os.environ["LLM_SERVICE_URL"]
    EMOTION_CLASSIFIER_URL = os.environ["EMOTION_CLASSIFIER_URL"]
    PERSONA_SERVICE_URL = os.environ["PERSONA_SERVICE_URL"]
except KeyError as e:
    print(f"🔥 Critical environment variable missing: {e}", file=sys.stderr)
    sys.exit(1)
RECORD_SECONDS = 5
RECORD_SAMPLE_RATE = 48000
WHISPER_SAMPLE_RATE = 16000
PLAYBACK_SAMPLE_RATE = 48000

# (The wait_for_health, record_audio, tts_play_wav_bytes, transcribe_audio, query_rag_system, and speak_text functions remain unchanged)
def wait_for_health(service_name: str, service_url: str, timeout_sec: int = 180) -> bool:
    start_time = time.time()
    health_url = f"{service_url}/health"
    while time.time() - start_time < timeout_sec:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                if "index_ready" in response.json() and not response.json()["index_ready"]:
                     print(f"⏳ Waiting for {service_name} index to be ready...")
                else:
                    print(f"✅ {service_name} is healthy at {health_url}")
                    return True
        except requests.RequestException:
            pass
        except Exception as e:
            print(f"An unexpected error occurred while checking {service_name}: {e}")
        time.sleep(3)
    print(f"🔥 Timed out waiting for {service_name} at {health_url}", file=sys.stderr)
    return False

def record_audio() -> Optional[io.BytesIO]:
    print(f"⏺ Recording for {RECORD_SECONDS} seconds at {RECORD_SAMPLE_RATE}Hz...")
    try:
        recording = sd.rec(int(RECORD_SECONDS * RECORD_SAMPLE_RATE), samplerate=RECORD_SAMPLE_RATE, channels=1, dtype='float32')
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
        import traceback
        traceback.print_exc()
        return None

def tts_play_wav_bytes(wav_bytes: bytes):
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
    if not wav_obj: return None
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

def get_ai_response(prompt_text: str):
    if not prompt_text:
        return None
    
    # The orchestrator's job is simpler now: just talk to the "brain"
    print("ORCHESTRATOR: Sending question to Persona Service...")
    try:
        response = requests.post(f"{PERSONA_SERVICE_URL}/generate_response", json={"text": prompt_text}, timeout=300)
        response.raise_for_status()
        return response.json().get("response")
    except Exception as e:
        print(f"❌ Persona system request failed: {e}", file=sys.stderr)
        return "Sorry, I had a problem thinking about that."


def speak_text(text: str, voice_path="/voices/my_voice.wav"):
    if not text:
        return
    try:
        print("🗣️  Requesting TTS model from Resource Manager...")
        requests.post(f"{RESOURCE_MANAGER_URL}/request_model", json={"model_name": "tts"}, timeout=180).raise_for_status()
        print("✅ TTS model is ready in RAM.")

        # Split the response into sentences for more fluid playback
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        for sentence in sentences:
            print(f"🔊 Synthesizing: '{sentence}'")
            r = requests.post(f"{TTS_SERVICE_URL}/api/tts", json={"text": sentence, "speaker_wav": voice_path}, timeout=120)
            r.raise_for_status()
            tts_play_wav_bytes(r.content)

    except requests.RequestException as e:
        print(f"❌ TTS workflow failed: {e}", file=sys.stderr)
    except Exception as e:
        print(f"❌ An unexpected error occurred during TTS: {e}", file=sys.stderr)


# --- NEW: Function to save memories in the background ---
def save_memory_async(memory_text: str):
    """
    Analyzes the emotion of the full conversational turn and sends the memory
    with its emotional vector to the RAG service.
    """
    if not memory_text:
        return

    print("🧠 Analyzing emotional fingerprint of the exchange before saving memory...")
    try:
        # 1. Call the emotion classifier on the COMPLETE exchange.
        emotion_response = requests.post(f"{EMOTION_CLASSIFIER_URL}/classify", json={"text": memory_text}, timeout=60)
        emotion_response.raise_for_status()
        emotion_data = emotion_response.json()
        emotion_vector = emotion_data.get("emotion_vector")

        if not emotion_vector:
            print("⚠️ Could not get emotion vector for memory. Skipping emotional context.")
            return

        # 2. Send the memory text AND its new emotional vector to the RAG service.
        print(f"✅ Emotional fingerprint analyzed. Sending to RAG service for storage...")
        requests.post(
            f"{RAG_SERVICE_URL}/add_memory", 
            json={"text": memory_text, "emotion_vector": emotion_vector}, 
            timeout=60
        )
        print("✅ Memory with its emotional vector has been sent for storage.")
    
    except requests.RequestException as e:
        print(f"⚠️  Could not save emotionally-contextualized memory: {e}", file=sys.stderr)
def main():
    """Main application loop."""
    print("🚀 Orchestrator starting.")

    if not all([
        wait_for_health("Resource Manager", RESOURCE_MANAGER_URL),
        wait_for_health("ASR Service", ASR_SERVICE_URL),
        wait_for_health("TTS Service", TTS_SERVICE_URL),
        wait_for_health("RAG Service", RAG_SERVICE_URL),
        wait_for_health("LLM Service", LLM_SERVICE_URL),
        wait_for_health("Persona Service", PERSONA_SERVICE_URL)
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
            
            # --- MODIFIED: Call the new response generation function ---
            reply = get_ai_response(user_text)

            print(f"🤖 LLM → '{reply}'")
            speak_text(reply)

            if user_text and reply:
                memory_to_save = f"A user said: '{user_text}'. I replied: '{reply}'."
                memory_thread = threading.Thread(target=save_memory_async, args=(memory_to_save,))
                memory_thread.start()
            
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
        except Exception as e:
            print(f"⚠ An unexpected error occurred in the main loop: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
