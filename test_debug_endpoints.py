#!/usr/bin/env python3
"""
Test script to verify both endpoints generate identical audio
"""

import sys
from pathlib import Path
import requests
import json
import wave
import soundfile as sf
import numpy as np
# Add the skyrimnet-xtts directory to the path
sys.path.insert(0, str(Path(__file__).parent / "skyrimnet-xtts"))

def test_endpoints(base_url="http://localhost:7860"):
    """Test both endpoints with the same input"""
    
    # Test data
    test_text = "This is a test of both endpoints to verify they generate identical audio output. This just a second sentence to increase the length a bit."
    test_speaker = "malebrute"
    test_language = "en"
    
    print(f"Testing both endpoints with:")
    print(f"  Text: {test_text}")
    print(f"  Speaker: {test_speaker}")
    print(f"  Language: {test_language}")
    print()
    
    # Test 1: tts_to_audio endpoint (FastAPI)
    print("Testing /tts_to_audio/ endpoint...")
    try:
        response1 = requests.post(
            f"{base_url}/tts_to_audio/",
            json={
                "text": test_text,
                "speaker_wav": test_speaker,
                "language": test_language
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response1.status_code == 200:
            print(f"✅ tts_to_audio: Success ({len(response1.content)} bytes)")
            audio_data = np.frombuffer(response1.content, dtype=np.int16)
            print(f"   tts_to_audio: Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
            print(f"   tts_to_audio: Audio length: {len(audio_data)/24000.0} seconds")
            # Save the file
            with open("test_tts_to_audio.wav", "wb") as f:
                f.write(response1.content)
        else:
            print(f"❌ tts_to_audio: Failed with status {response1.status_code}")
            print(f"   Response: {response1.text}")
            
    except Exception as e:
        print(f"❌ tts_to_audio: Error - {e}")
    
    # Test 2: generate_audio endpoint (Gradio API auto-exposed)
    print("\nTesting /gradio_api/call/generate_audio endpoint...")
    try:
        # Gradio API format - use the same test parameters
        response2 = requests.post(
            f"{base_url}/gradio_api/call/generate_audio",
            json={
                "data": ["Zyphra/Zonos-v0.1-transformer", test_text, "en-us", {
            "meta": {
                "_type": "gradio.FileData"
            },
            "mime_type": "audio/wav",
            "orig_name": "malebrute.wav",
            "path": "assets\\malebrute.wav",
            "url": "http://localhost:7860/gradio_api/file=assets\\malebrute.wav"
        }, {
            "meta": {
                "_type": "gradio.FileData"
            },
            "mime_type": "audio/wav",
            "orig_name": "empty_100ms.wav",
            "path": "assets\\silent_100ms.wav",
            "url": "http://localhost:7860/gradio_api/file=assets\\silent_100ms.wav"
        }, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.7799999713897705, 24000, 45.0, 15.0, 1, "true", 0.30000001192092896, 1.0, 0.0, 0.03999999910593033, 0.8999999761581421, 2.0, 0.6000000238418579, 3246887291427558572, "false", ["emotion"]]
                
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response2.status_code == 200:
            result = response2.json()
            if "data" in result and len(result["data"]) > 0:
                audio_path = result["data"][0]
                print(f"✅ generate_audio: Success - {audio_path}")
                
                # Try to read the generated file
                if audio_path and Path(audio_path).exists():
                    file_size = Path(audio_path).stat().st_size
                    print(f"   Generated file size: {file_size} bytes")
                else:
                    print(f"   ⚠️  Generated file not found: {audio_path}")
            else:
                print(f"❌ generate_audio: No data in response - {result}")
        else:
            print(f"❌ generate_audio: Failed with status {response2.status_code}")
            print(f"   Response: {response2.text}")
        if "event_id" in result:
            response3 = requests.get(f"{base_url}/gradio_api/call/generate_audio/{result['event_id']}")
            print(f"   Event stream response: {response3.status_code}")
            if response3.status_code == 200:
                # Parse the server-sent events format
                content_text = response3.text
                print(f"   Event stream content: {content_text}")
                
                # Extract the data line from server-sent events
                if "data: [" in content_text:
                    data_line = content_text.split("data: ")[1].split("\n")[0]
                    try:
                        event_data = json.loads(data_line)
                        if len(event_data) > 0 and isinstance(event_data[0], dict) and "url" in event_data[0]:
                            file_url = event_data[0]["url"]
                            file_path = event_data[0]["path"]
                            print(f"✅ generate_audio: Success - Generated file at {file_path}")
                            
                            # Download the actual audio file to check size
                            try:
                                response4 = requests.get(file_url)
                                if response4.status_code == 200:
                                    print(f"   Downloaded audio file: {len(response4.content)} bytes")
                                    audio_data = np.frombuffer(response4.content, dtype=np.int16)
                                    print(f"  generate_audio: Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
                                    print(f"  generate_audio: Audio length: {len(audio_data)/24000.0} seconds")
                                    # Save for comparison
                                    with open("test_generate_audio.wav", "wb") as f:
                                        f.write(response4.content)
                                else:
                                    print(f"   Failed to download audio: {response4.status_code}")
                            except Exception as e:
                                print(f"   Error downloading audio: {e}")
                        else:
                            print(f"   Unexpected event data format: {event_data}")
                    except json.JSONDecodeError as e:
                        print(f"   Failed to parse event data: {e}")
                else:
                    print(f"   No data found in event stream")
    except Exception as e:
        print(f"❌ generate_audio: Error - {e}")

if __name__ == "__main__":
    test_endpoints()