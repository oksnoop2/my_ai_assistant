# my_ai_assistant/orchestrator/orchestrator.py (CORRECT AND FINAL VERSION)

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
from scipy.signal import resample

# ---- Service base URLs and Configuration ----
ASR_URL = "http://asr:9000"
TTS_URL = "http://tts:8000"
LLM_URL = "http://llm:8080"

RECORD_SECONDS = 5
WHISPER_SAMPLE_RATE = 16000
PLAYBACK_SAMPLE_RATE = 48000
SLEEP_SEC = 2.0

# ---- Core Functions ----

def record_audio() -> Optional[io.BytesIO]:
    """Records audio from the specified ALSA device and returns it as an in-memory WAV object."""
    card, dev = 1, 7
    tmp_name = tempfile.mktemp(suffix=".wav")
    hw_str = f"hw:{card},{dev}"
    cmd = [
        "arecord", "-D", hw_str, "-f", "S16_LE", "-c", "2",
        "-r", str(WHISPER_SAMPLE_RATE), "-d", str(RECORD_SECONDS), tmp_name
    ]
    print(f"⏺ Recording via arecord (device {hw_str}) -> {tmp_name}")
    try:
        subprocess.run(cmd, check=True, timeout=RECORD_SECONDS + 10, capture_output=True)
        sr, data = read(tmp_name)
        data_mono = data.mean(axis=1).astype(np.int16)
        buf = io.BytesIO()
        write(buf, sr, data_mono)
        buf.seek(0)
        return buf
    except subprocess.CalledProcessError as e:
        print(f"⚠ arecord failed:\n{e.stderr.decode()}", file=sys.stderr)
        return None
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

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
    print("📝 Transcribing audio via ASR...")
    try:
        wav_obj.seek(0)
        files = {'audio_file': ('rec.wav', wav_obj.read(), 'audio/wav')}
        r = requests.post(f"{ASR_URL}/asr", params={'task': 'transcribe'}, files=files, timeout=30)
        r.raise_for_status()
        text = r.json().get("text", "").strip()
        print(f"🗣 ASR → '{text}'")
        return text
    except Exception as e:
        print(f"❌ ASR request failed: {e}", file=sys.stderr)
        return None

def query_llm(prompt_text: str):
    """Manages the LLM model lifecycle (load, query, unload) to conserve VRAM."""
    if not prompt_text:
        return None
    response_content = None
    try:
        print("🚀 Requesting LLM model load...")
        requests.post(f"{LLM_URL}/load", timeout=90).raise_for_status()
        
        print("🤖 Querying LLM...")
        payload = {"prompt": f"User: {prompt_text}\nAssistant:"}
        r = requests.post(f"{LLM_URL}/completion", json=payload, timeout=60)
        r.raise_for_status()
        response_content = r.json().get("content", "").strip()
    except Exception as e:
        print(f"❌ LLM workflow failed: {e}", file=sys.stderr)
    finally:
        # Crucially, always attempt to unload the model to free VRAM
        print("🛑 Requesting LLM model unload...")
        try:
            requests.post(f"{LLM_URL}/unload", timeout=60)
        except Exception as unload_e:
            print(f"⚠️ Failed to unload LLM model: {unload_e}", file=sys.stderr)
    return response_content

def speak_text(text: str, voice_path="/voices/my_voice.wav"):
    """Manages the TTS model lifecycle (load, synthesize, unload) to conserve VRAM."""
    if not text:
        return
    try:
        print("🚀 Requesting TTS model load...")
        requests.post(f"{TTS_URL}/load", timeout=120).raise_for_status()
        
        print(f"🔊 Requesting TTS for: '{text}'")
        r = requests.post(f"{TTS_URL}/api/tts", json={"text": text, "speaker_wav": voice_path}, timeout=120)
        r.raise_for_status()
        tts_play_wav_bytes(r.content)
    except Exception as e:
        print(f"❌ TTS workflow failed: {e}", file=sys.stderr)
    finally:
        # Crucially, always attempt to unload the model to free VRAM
        print("🛑 Requesting TTS model unload...")
        try:
            requests.post(f"{TTS_URL}/unload", timeout=60)
        except Exception as unload_e:
            print(f"⚠️ Failed to unload TTS model: {unload_e}", file=sys.stderr)

def main():
    """Main application loop."""
    print("🚀 Orchestrator starting.")
    # Wait for all services to be responsive before starting the main loop
    if not all([
        wait_for_health("ASR", ASR_URL),
        wait_for_health("LLM", LLM_URL),
        wait_for_health("TTS", TTS_URL)
    ]):
        print("🔥 One or more services are not healthy. Exiting.", file=sys.stderr)
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
                
            reply = query_llm(user_text)
            print(f"🤖 LLM → '{reply}'")
            speak_text(reply)
            
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
        except Exception as e:
            print(f"⚠ An unexpected error occurred in the main loop: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
