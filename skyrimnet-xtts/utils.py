import functools
import threading
import psutil
import torch
import datetime
import torchaudio
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from loguru import logger


class LatentCacheManager:
    """in-memory cache manager for latent embeddings."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested operations

    def get(self, language: str, cache_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._cache.get(language, {}).get(cache_key)

    def set(self, language: str, cache_key: str, latents: Dict[str, Any]) -> None:
        with self._lock:
            if language not in self._cache:
                self._cache[language] = {}
            self._cache[language][cache_key] = latents

    def get_all_keys(self) -> List[Tuple[str, str]]:
        with self._lock:
            keys = []
            for lang, lang_cache in self._cache.items():
                for key in lang_cache.keys():
                    keys.append((lang, key))
            return keys

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            total_entries = sum(len(lang_cache)
                                for lang_cache in self._cache.values())
            return {
                'total_entries': total_entries,
                'languages': len(self._cache),
                'languages_list': list(self._cache.keys())
            }


# Global cache manager instance
cache_manager = LatentCacheManager()


@functools.cache
def get_process_creation_time():
    """Get the process creation time as a datetime object"""
    p = psutil.Process(os.getpid())
    creation_timestamp = p.create_time()
    return datetime.datetime.fromtimestamp(creation_timestamp)


@functools.cache
def get_latent_dir(language: str = "en") -> Path:
    """Get or create the conditionals cache directory"""
    cache_dir = Path("latents_pt").joinpath(language)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

@functools.cache
def get_speakers_dir(language: str = "en") -> Path:
    """Get or create the speakers directory"""
    speakers_dir = Path("speakers").joinpath(language)
    speakers_dir.mkdir(parents=True, exist_ok=True)
    return speakers_dir

@functools.cache
def get_cache_key(audio_path, uuid: int = None) -> Optional[str]:
    """Generate a cache key based on audio file, UUID"""
    if audio_path is None:
        return None

    cache_prefix = Path(audio_path).stem
    if uuid:
        # Convert UUID to hex string for readability
        try:
            uuid_hex = hex(uuid)[2:]  # Remove '0x' prefix
        except (TypeError, ValueError):
            uuid_hex = str(uuid)

        cache_key = f"{cache_prefix}_{uuid_hex}"
    else:
        cache_key = cache_prefix    
    return cache_key


def load_pt_latents(path, device):
    expected_shapes = {
        "gpt_cond_latent": torch.Size([1, 32, 1024]),
        "speaker_embedding": torch.Size([1, 512, 1]),
    }
    latents = torch.load(path, map_location=device)
    for key, expected_shape in expected_shapes.items():
        actual_shape = latents[key].shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"{key} shape mismatch: expected {expected_shape}, got {actual_shape}")
    return latents


def _save_pt_to_disk(filename, data):
    try:
        torch.save(data, filename)
    except Exception as e:
        logger.error(f"Failed to save data: {e}")


def get_latent_from_audio(model, language: str, speaker_audio: str, speaker_audio_uuid: int = None, latents_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get or compute and cache latents for a given speaker audio file."""
    cache_file_key = get_cache_key(speaker_audio, speaker_audio_uuid)

    # Check in-memory cache first
    cached = cache_manager.get(language, cache_file_key)
    if cached:
        logger.info(
            f"Using in-memory cached latents for {Path(speaker_audio).stem} with UUID {speaker_audio_uuid}")
        return cached["gpt_cond_latent"], cached["speaker_embedding"]

    # Check disk cache

    latent_dir = get_latent_dir(language=language)
    latent_filename = latent_dir.joinpath(f"{cache_file_key}.pt")
    if latent_filename.is_file():
        logger.info(f"Loading cached latents from {latent_filename}")
        latents = load_pt_latents(latent_filename, model.device)
        # Store in memory cache for future use
        if latents:
            cache_manager.set(language, cache_file_key, latents)
            return latents["gpt_cond_latent"], latents["speaker_embedding"]
        
    if not latents_only:
        # Compute new latents
        logger.info(
            f"Computing latents for {speaker_audio} and caching to {latent_filename}")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[
                                                                            speaker_audio])
        logger.info(
            f"Computed latents shapes: gpt_cond_latent={gpt_cond_latent.shape}, speaker_embedding={speaker_embedding.shape}")

        latents = {"gpt_cond_latent": gpt_cond_latent,
                   "speaker_embedding": speaker_embedding}

        # Save to disk asynchronously
        threading.Thread(
            target=_save_pt_to_disk,
            args=(latent_filename, latents),
            daemon=True
        ).start()

        # Store in memory cache
        cache_manager.set(language, cache_file_key, latents)
        return gpt_cond_latent, speaker_embedding
    
    # This really should not happen, but just in case
    return None

def init_latent_cache(model, supported_languages: List[str] = ["en"]) -> None:
    """Initialize latent cache from disk for all supported languages."""
    cached_latents = {}
    for lang in supported_languages:
        latent_dir = get_latent_dir(language=lang)
        for filename in latent_dir.glob("*.pt"):
            try:
                cached_latents[lang] = filename.stem
                latents = load_pt_latents(filename, model.device)
                #logger.info(f"Loaded latent shapes: gpt_cond_latent={latents['gpt_cond_latent'].shape}, speaker_embedding={latents['speaker_embedding'].shape}")
                cache_manager.set(lang, filename.stem, latents)
            except Exception as e:
                logger.error(f"Failed to load latents from {filename}: {e}")
        speaker_dir = get_speakers_dir(language=lang)
        for speaker_wav_wav in speaker_dir.glob("*.wav"):
            if speaker_wav_wav.stem in cached_latents.get(lang, []):
                continue # Already cached from .pt file
            try:
                latents = get_latent_from_audio(model, lang, str(speaker_wav_wav))
                cache_manager.set(lang, speaker_wav_wav.stem, latents)
            except Exception as e:
                logger.error(f"Failed to load latents from speaker{speaker_wav_wav}: {e}")

    stats = cache_manager.get_stats()
    logger.info(
        f"Initialized latent cache with {stats['total_entries']} entries across languages: {stats['languages_list']}")


def get_latent_cache_keys() -> List[Tuple[str, str]]:
    """Return a list of all cached latent keys."""
    return cache_manager.get_all_keys()


@functools.cache
def get_wavout_dir():
    formatted_start_time = get_process_creation_time().strftime("%Y%m%d_%H%M%S")
    wavout_dir = Path("output_temp").joinpath(formatted_start_time)
    wavout_dir.mkdir(parents=True, exist_ok=True)
    return wavout_dir


def save_torchaudio_wav(wav_tensor, sr, audio_path, uuid: int = None) -> Path:
    """Save a tensor as a WAV file using torchaudio"""

    formatted_now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{formatted_now_time}_{get_cache_key(audio_path, uuid)}"
    path = Path(get_wavout_dir(), f"{filename}.wav")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torchaudio.save(path, wav_tensor, sr, encoding="PCM_S")
    return path.resolve()
