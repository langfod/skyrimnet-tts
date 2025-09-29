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
    warnings.filterwarnings("ignore", module='numbpysbd.*', append=True)
    warnings.filterwarnings("ignore", module='jamo.*', append=True)
    warnings.filterwarnings("ignore", module='g2pkk.*', append=True)

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
    "TEMPERATURE": 0.7,
    "TOP_P": 1.0,
    "TOP_K": 50,
    "SPEED": 1.0,
    "REPETITION_PENALTY": 2.0
}

# Text splitting threshold per language (characters)
DEFAULT_CHAR_LIMITS = 250


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