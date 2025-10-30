#!/usr/bin/env python3
"""
Shared Configuration Module for SkyrimNet TTS Applications
Contains common environment setup, constants, and configuration settings
"""

import os
import sys


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_environment():
    """Setup common environment variables and system configuration"""
    
    # Fix torch.compile C++ compilation issues on Windows
    if sys.platform == "win32":
        os.environ["TORCH_COMPILE_CPP_FORCE_X64"] = "1"
        os.environ["DISTUTILS_USE_SDK"] = "1" 
        os.environ["MSSdk"] = "1"

    # TTS Library configuration
    os.environ["COQUI_TOS_AGREED"] = "1"
    os.environ["TTS_HOME"] = "models"

    import warnings
    warnings.filterwarnings("ignore", module='Setuptools.*', append=True)
    warnings.filterwarnings("ignore", module='jieba.*', append=True)
    warnings.filterwarnings("ignore", module='ko_speech_tools.*', append=True)

# =============================================================================
# COMMON CONSTANTS
# =============================================================================

# Supported language codes for TTS
SUPPORTED_LANGUAGE_CODES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]

# Cache configuration defaults
DEFAULT_CACHE_CONFIG = {
    "ENABLE_DISK_CACHE": True,
    "ENABLE_MEMORY_CACHE": True
}

# Default TTS inference parameters
DEFAULT_TTS_PARAMS = {
    "TEMPERATURE": 0.9,
    "TOP_P": 1.0,
    "TOP_K": 50,
    "SPEED": 1.0,
    "REPETITION_PENALTY": 2.1
}

# Text splitting threshold per language (characters)
DEFAULT_CHAR_LIMITS = 250

# Config file path
_CONFIG_FILE_PATH = "skyrimnet_config.txt"
_CONFIG_CACHE = None

# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def validate_language(language: str) -> str:
    """
    Validate and normalize language code
    
    Args:
        language: Language code (may include region, e.g., "en-US")
        
    Returns:
        Normalized language code (e.g., "en")
        
    Raises:
        ValueError: If language is not supported
    """
    language = language.lower()
    if language is None:
        normalized = "en"
    elif language in ["zh-cn", "zh", "cn"]:
        normalized = "zh-cn"
    else:
        normalized = language.split("-")[0] if language else "en"
    
    if normalized not in SUPPORTED_LANGUAGE_CODES:
        raise ValueError(f"Language '{language}' not supported. "
                        f"Supported languages: {SUPPORTED_LANGUAGE_CODES}")
    
    return normalized


def get_cache_config(enable_disk=None, enable_memory=None):
    """
    Get cache configuration with optional overrides
    
    Args:
        enable_disk: Override for disk cache (None to use default)
        enable_memory: Override for memory cache (None to use default)
        
    Returns:
        dict: Cache configuration
    """
    config = DEFAULT_CACHE_CONFIG.copy()
    
    if enable_disk is not None:
        config["ENABLE_DISK_CACHE"] = enable_disk
    
    if enable_memory is not None:
        config["ENABLE_MEMORY_CACHE"] = enable_memory
    
    return config


def load_skyrimnet_config():
    """
    Load configuration from skyrimnet_config.txt with caching.
    
    Returns:
        dict: Configuration overrides from file. Keys can have:
            - numeric values (float/int)
            - "api" string indicating to use API-provided values
            - "default" string (treated as no override)
    """
    global _CONFIG_CACHE
    
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    
    from pathlib import Path
    from loguru import logger
    
    config_overrides = {}
    
    if Path(_CONFIG_FILE_PATH).exists():
        try:
            with open(_CONFIG_FILE_PATH, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip().lower()
                            value = value.strip()
                            
                            # Parse TTS parameters
                            if key in ['temperature', 'top_p', 'speed', 'repetition_penalty']:
                                if value.lower() == "api":
                                    config_overrides[key] = "api"
                                elif value.lower() != "default":
                                    try:
                                        config_overrides[key] = float(value)
                                    except ValueError:
                                        logger.warning(f"Invalid value for {key}: {value}")
                            elif key == 'top_k':
                                if value.lower() == "api":
                                    config_overrides[key] = "api"
                                elif value.lower() != "default":
                                    try:
                                        config_overrides[key] = int(value)
                                    except ValueError:
                                        logger.warning(f"Invalid value for {key}: {value}")
        except Exception as e:
            logger.warning(f"Error loading config file {_CONFIG_FILE_PATH}: {e}")
    
    _CONFIG_CACHE = config_overrides
    return _CONFIG_CACHE


def get_tts_params(payload_params=None, override_flag=False):
    """
    Resolve TTS parameters based on priority:
    1. DEFAULT_TTS_PARAMS (base defaults)
    2. Config file values (if set and not "api")
    3. Config file "api" mode + payload values (if config says "api")
    4. Payload override flag + payload values (if override=True in payload)
    
    Args:
        payload_params: dict with optional keys: temperature, top_p, top_k, speed, repetition_penalty
        override_flag: bool, if True payload values override everything
        
    Returns:
        dict: Resolved TTS parameters with all 5 keys
    """
    config = load_skyrimnet_config()
    params = DEFAULT_TTS_PARAMS.copy()
    
    # Map DEFAULT_TTS_PARAMS keys to standard names
    result = {
        'temperature': params['TEMPERATURE'],
        'top_p': params['TOP_P'],
        'top_k': params['TOP_K'],
        'speed': params['SPEED'],
        'repetition_penalty': params['REPETITION_PENALTY']
    }
    
    if payload_params is None:
        payload_params = {}
    
    # Process each parameter
    for param_name in result.keys():
        config_value = config.get(param_name)
        payload_value = payload_params.get(param_name)
        
        # Priority logic:
        if override_flag and payload_value is not None:
            # Highest priority: override flag + payload value
            result[param_name] = payload_value
        elif config_value == "api" and payload_value is not None:
            # Config says "api" and we have a payload value
            result[param_name] = payload_value
        elif config_value is not None and config_value != "api":
            # Config file has a numeric value
            result[param_name] = config_value
        # else: keep the default from DEFAULT_TTS_PARAMS
    
    return result
