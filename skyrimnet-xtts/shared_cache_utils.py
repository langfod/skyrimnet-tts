import functools
import threading
import psutil
import torch
from datetime import datetime
import torchaudio
import os
import warnings
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from loguru import logger
from COQUI_AI_TTS.tts.models.xtts import Xtts


def get_model_device(model):
    """Safely get the device from a model, handling different model types."""
    # Cache the device on the model object to avoid repeated parameter iteration
    if hasattr(model, '_cached_device'):
        return model._cached_device
    
    try:
        # Try the device property first (works with our BaseTTS models)
        if hasattr(model, 'device'):
            device = model.device
        else:
            # Fall back to getting device from parameters (only once!)
            device = next(model.parameters()).device
        
        # Cache the result for future calls
        model._cached_device = device
        return device
        
    except Exception:
        # Final fallback to CPU
        logger.warning("Could not determine model device, falling back to CPU")
        device = torch.device('cpu')
        model._cached_device = device
        return device


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
    return datetime.fromtimestamp(p.create_time())


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
def get_cache_key(audio_path) -> Optional[str]:
    """Generate a cache key based on audio file"""
    if audio_path is None:
        return None

    cache_prefix = Path(audio_path).stem
    return cache_prefix


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


def load_json_latents(path, device):
    """Load and convert legacy JSON latents to proper tensor format."""
    expected_shapes = {
        "gpt_cond_latent": torch.Size([1, 32, 1024]),
        "speaker_embedding": torch.Size([1, 512, 1]),
    }
    
    try:
        with open(path, 'r') as f:
            json_data = json.load(f)
        
        # Convert lists back to tensors with proper shapes based on xtts-api-server format
        gpt_cond_latent = torch.tensor(json_data["gpt_cond_latent"], dtype=torch.float32).to(device)
        speaker_embedding = torch.tensor(json_data["speaker_embedding"], dtype=torch.float32).to(device)
        
        # Check if we need reshaping (if loaded from flattened lists)
        if gpt_cond_latent.dim() == 2:  # Flattened format [32, 1024]
            gpt_cond_latent = gpt_cond_latent.unsqueeze(0)  # Add batch dimension [1, 32, 1024]
        elif gpt_cond_latent.dim() == 1:  # Completely flattened format
            gpt_cond_latent = gpt_cond_latent.reshape((-1, 1024)).unsqueeze(0)
            
        if speaker_embedding.dim() == 2:  # Flattened format [512, 1] or [1, 512]
            if speaker_embedding.shape[0] == 1:  # [1, 512] 
                speaker_embedding = speaker_embedding.unsqueeze(-1)  # [1, 512, 1]
            elif speaker_embedding.shape[1] == 1:  # [512, 1]
                speaker_embedding = speaker_embedding.unsqueeze(0)  # [1, 512, 1]
        elif speaker_embedding.dim() == 1:  # Completely flattened [512]
            speaker_embedding = speaker_embedding.unsqueeze(0).unsqueeze(-1)  # [1, 512, 1]
        
        latents = {
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding
        }
        
        # Validate shapes
        for key, expected_shape in expected_shapes.items():
            actual_shape = latents[key].shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"{key} shape mismatch: expected {expected_shape}, got {actual_shape}")
        
        return latents
        
    except (KeyError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to load JSON latents from {path}: {e}")
        raise


def _save_pt_to_disk(filename, data):
    try:
        torch.save(data, filename)
    except Exception as e:
        logger.error(f"Failed to save data: {e}")


def get_latent_from_audio(model:Xtts, language: str, speaker_audio: str, latents_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get or compute and cache latents for a given speaker audio file."""
    if speaker_audio is None:
        return None, None

    cache_file_key = get_cache_key(speaker_audio)
    cached = cache_manager.get(language, cache_file_key)
    if cached:
        logger.info(
            f"Using in-memory cached latents for {Path(speaker_audio).stem}")
        gpt_cond_latent = cached["gpt_cond_latent"].to(dtype=model.gpt.dtype)
        speaker_embedding = cached["speaker_embedding"].to(dtype=torch.float32)
        return gpt_cond_latent, speaker_embedding

    # Check disk cache
    latent_dir = get_latent_dir(language=language)
    latent_filename = latent_dir.joinpath(f"{cache_file_key}.pt")
    if latent_filename.is_file():
        logger.info(f"Loading cached latents from {latent_filename}")
        latents = load_pt_latents(latent_filename, get_model_device(model))
        latents["gpt_cond_latent"] = latents["gpt_cond_latent"].to(dtype=model.gpt.dtype)
        latents["speaker_embedding"] = latents["speaker_embedding"].to(dtype=torch.float32)
        # Store in memory cache for future use
        if latents:
            cache_manager.set(language, cache_file_key, latents)
            return latents["gpt_cond_latent"], latents["speaker_embedding"]
        
    if not latents_only:
        logger.info(
            f"Computing latents for {speaker_audio} and caching to {latent_filename}")
        
        # get_conditioning_latents returns a tuple (gpt_cond_latent, speaker_embedding)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            latent_result = model.get_conditioning_latents(audio_path=[speaker_audio])
        
        # Handle both tuple and potential dict return types for compatibility
        if isinstance(latent_result, tuple):
            gpt_cond_latent, speaker_embedding = latent_result
        elif isinstance(latent_result, dict):
            gpt_cond_latent = latent_result["gpt_cond_latent"]
            speaker_embedding = latent_result["speaker_embedding"]
        else:
            raise TypeError(f"Unexpected return type from get_conditioning_latents: {type(latent_result)}")
        
        latents = {"gpt_cond_latent": gpt_cond_latent,
                   "speaker_embedding": speaker_embedding}

        threading.Thread(
            target=_save_pt_to_disk,
            args=(latent_filename, latents),
            daemon=True
        ).start()

        cache_manager.set(language, cache_file_key, latents)
        return gpt_cond_latent, speaker_embedding
    
    # This really should not happen, but just in case
    return None, None


def init_latent_cache(model, supported_languages: List[str] = ["en"]) -> None:
    """Initialize latent cache from disk for all supported languages."""
    cached_latents = {}
    for lang in supported_languages:
        latent_dir = get_latent_dir(language=lang)
        cached_latents[lang] = []  # Initialize as empty list for each language
        
        # Load existing .pt files from latents directory
        for filename in latent_dir.glob("*.pt"):
            try:
                cached_latents[lang].append(filename.stem)
                latents = load_pt_latents(filename, get_model_device(model))
                latents["gpt_cond_latent"] = latents["gpt_cond_latent"].to(dtype=model.gpt.dtype)
                latents["speaker_embedding"] = latents["speaker_embedding"].to(dtype=torch.float32)
                cache_manager.set(lang, filename.stem, latents)
            except Exception as e:
                logger.error(f"Failed to load latents from {filename}: {e}")
        
        speaker_dir = get_speakers_dir(language=lang)
        
        # Get all speaker files and organize by base name
        speaker_files = {}
        for file_path in speaker_dir.iterdir():
            if file_path.is_file() and file_path.suffix in ['.wav', '.json']:
                base_name = file_path.stem
                if base_name not in speaker_files:
                    speaker_files[base_name] = {}
                speaker_files[base_name][file_path.suffix] = file_path
        
        # Process each speaker, preferring .wav over .json
        for base_name, files in speaker_files.items():
            if base_name in cached_latents.get(lang, []):
                continue  # Already cached from .pt file
            
            try:
                if '.wav' in files:
                    # Prefer .wav files - compute latents from audio
                    speaker_wav_path = files['.wav']
                    logger.info(f"Processing .wav file: {speaker_wav_path}")
                    gpt_cond_latent, speaker_embedding = get_latent_from_audio(model, lang, str(speaker_wav_path))
                    latents = {"gpt_cond_latent": gpt_cond_latent, "speaker_embedding": speaker_embedding}
                    cache_manager.set(lang, base_name, latents)
                    
                elif '.json' in files:
                    # Use legacy .json files and convert to .pt format
                    json_path = files['.json']
                    logger.info(f"Loading legacy JSON latents from: {json_path}")
                    
                    latents = load_json_latents(json_path, get_model_device(model))
                    # Convert to correct dtypes
                    latents["gpt_cond_latent"] = latents["gpt_cond_latent"].to(dtype=model.gpt.dtype)
                    latents["speaker_embedding"] = latents["speaker_embedding"].to(dtype=torch.float32)
                    
                    pt_filename = latent_dir.joinpath(f"{base_name}.pt")
                    threading.Thread(
                        target=_save_pt_to_disk,
                        args=(pt_filename, latents),
                        daemon=True
                    ).start()
                    logger.info(f"Converting JSON latents to .pt format: {pt_filename}")
                    
                    cache_manager.set(lang, base_name, latents)
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to load latents for speaker {base_name}: {e}")

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


def save_torchaudio_wav(wav_tensor, sr, audio_path) -> Path:
    """Save a tensor as a WAV file using torchaudio"""

    if wav_tensor.device.type != 'cpu':
        wav_tensor = wav_tensor.cpu()

    formatted_now_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{formatted_now_time}_{get_cache_key(audio_path)}"
    path = Path(get_wavout_dir(), f"{filename}.wav")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torchaudio.save(path, wav_tensor, sr, encoding="PCM_S")
    del wav_tensor
    return path #.resolve()
