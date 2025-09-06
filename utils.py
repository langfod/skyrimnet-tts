from pathlib import Path
import functools
import json
import threading
import warnings
import psutil
import torch
import datetime
import time
import torchaudio
import os
from loguru import logger

latent_cache = {}


@functools.cache
def get_process_creation_time():
    """Get the process creation time as a datetime object"""
    p = psutil.Process(os.getpid())
    creation_timestamp = p.create_time()
    return datetime.datetime.fromtimestamp(creation_timestamp)

@functools.cache
def get_latent_dir(language: str="en") -> Path:
    """Get or create the conditionals cache directory"""
    cache_dir = Path("latents").joinpath(language)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

@functools.cache
def get_cache_key(audio_path, uuid: int=0):
    """Generate a cache key based on audio file, UUID, and exaggeration"""
    if audio_path is None:
        return None

    # Extract just the filename without extension as prefix
    try:
        filename = Path(audio_path).stem  # Gets filename without extension
        # Remove any temp directory prefixes, just keep the actual filename
        cache_prefix = filename
    except Exception:
        cache_prefix = "unknown"

    # Convert UUID to hex string for readability
    try:
        uuid_hex = hex(uuid)[2:]  # Remove '0x' prefix
    except (TypeError, ValueError):
        uuid_hex = str(uuid)


    cache_key = f"{cache_prefix}_{uuid_hex}"

    return cache_key

def _save_as_json_to_disk(filename, data):
    """Non-blocking worker function to save conditionals to disk"""
    try:
        with open(filename, "w") as new_file:
            json.dump(data, new_file)

    except Exception as e:
        print(f"Failed to save data: {e}")

def get_latent_from_audio(model, language, speaker_audio, speaker_audio_uuid):
    global latent_cache


    latent_dir = get_latent_dir(language=language)
    latent_filename = latent_dir.joinpath(f"{get_cache_key(speaker_audio, speaker_audio_uuid)}.json")
    if latent_cache.get(language) and latent_cache[language].get(get_cache_key(speaker_audio, speaker_audio_uuid)):
        cached = latent_cache[language][get_cache_key(speaker_audio, speaker_audio_uuid)]
        logger.info(f"Using in-memory cached latents for {Path(speaker_audio).stem} with UUID {speaker_audio_uuid}")
        return cached["gpt_cond_latent"], cached["speaker_embedding"]
    
    if latent_filename.is_file():
        logger.info(f"Loading cached latents from {latent_filename}")
        with open(latent_filename, "r") as latent_file:
            latents = json.load(latent_file)
        speaker_embedding = (torch.tensor(latents["speaker_embedding"]).unsqueeze(0).unsqueeze(-1))
        gpt_cond_latent = (torch.tensor(latents["gpt_cond_latent"]).reshape((-1, 1024)).unsqueeze(0))  
        return gpt_cond_latent, speaker_embedding
    
    print(f"Computing latents for {speaker_audio} and caching to {latent_filename}")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_audio])

    latents = {
                "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
                "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
              }
    
    threading.Thread(
                target=_save_as_json_to_disk,
                args=(latent_filename, latents),
                daemon=True
    ).start()
    latent_cache.setdefault(language, {})[get_cache_key(speaker_audio, speaker_audio_uuid)] = {
        "speaker_embedding": speaker_embedding,
        "gpt_cond_latent": gpt_cond_latent
    }
    return gpt_cond_latent, speaker_embedding

def init_latent_cache():
    """Initialize latent cache directories for supported languages."""
    global latent_cache
    supported_languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"]
    for lang in supported_languages:
        latent_dir = get_latent_dir(language=lang)
        for filename in latent_dir.glob("*.json"):
            with open(filename, "r") as latent_file:
                latents = json.load(latent_file)
                speaker_embedding = (torch.tensor(latents["speaker_embedding"]).unsqueeze(0).unsqueeze(-1))
                gpt_cond_latent = (torch.tensor(latents["gpt_cond_latent"]).reshape((-1, 1024)).unsqueeze(0))  
            latent_cache.setdefault(lang, {})[filename.stem] = {
                "speaker_embedding": speaker_embedding,
                "gpt_cond_latent": gpt_cond_latent
            }

def get_latent_cache_keys():
    """Return a list of all cached latent keys."""
    global latent_cache
    keys = []
    for lang, lang_cache in latent_cache.items():
        for key in lang_cache.keys():
            keys.append((lang, key))
    return keys

@functools.cache
def get_wavout_dir():
    formatted_start_time = get_process_creation_time().strftime("%Y%m%d_%H%M%S")
    wavout_dir = Path("output_temp").joinpath(formatted_start_time)
    wavout_dir.mkdir(parents=True, exist_ok=True)
    return wavout_dir

def save_torchaudio_wav(wav_tensor, sr, audio_path, uuid):
    """Save a tensor as a WAV file using torchaudio"""

    formatted_now_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")

    filename = f"{formatted_now_time}_{get_cache_key(audio_path, uuid)}"
    path = get_wavout_dir() / f"{filename}.wav"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torchaudio.save(path, wav_tensor.to("cpu"), sr, encoding="PCM_S")
    return path.resolve()