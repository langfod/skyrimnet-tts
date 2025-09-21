#!/usr/bin/env python3
"""
SkyrimNet TTS Package
A unified Text-to-Speech application with FastAPI backend and Gradio UI
"""

__version__ = "1.0.0"
__author__ = "SkyrimNet Team"
__description__ = "Unified XTTS application with API and web interface"

# Import key components for external use
from .shared_config import SUPPORTED_LANGUAGE_CODES, setup_environment
from .shared_models import load_model
from .shared_args import parse_api_args, parse_gradio_args

__all__ = [
    "SUPPORTED_LANGUAGE_CODES",
    "setup_environment", 
    "load_model",
    "parse_api_args",
    "parse_gradio_args",
]
