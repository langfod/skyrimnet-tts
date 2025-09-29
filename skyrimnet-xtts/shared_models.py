#!/usr/bin/env python3
"""
Shared Model Management Module for SkyrimNet TTS Applications
Contains common model loading, initialization, and management functions
"""

import torch
from loguru import logger
from typing import Optional, Any

# Local imports - Use absolute imports from the package
import os
import sys

# Add current package directory to Python path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# TTS imports - Now these will work consistently
from COQUI_AI_TTS.utils.manage import ModelManager
from COQUI_AI_TTS.tts.configs.xtts_config import XttsConfig
from COQUI_AI_TTS.tts.models.xtts import Xtts
from shared_config import SUPPORTED_LANGUAGE_CODES


# =============================================================================
# MODEL LOADING AND MANAGEMENT
# =============================================================================

def load_model(model_name="xtts_v2", use_cpu=False):
    """
    Load XTTS model with configuration
    
    Args:
        model_name: Name of the model to load (default: "xtts_v2")
        use_cpu: Whether to force CPU mode instead of CUDA
        
    Returns:
        Xtts: Loaded and configured model
        
    Raises:
        Exception: If model loading fails
    """
    logger.info(f"Loading model: {model_name}, use_cpu: {use_cpu}")
    
    try:
        # Download/locate model files
        output_model_path, output_config_path, model_item = ModelManager(progress_bar=True).download_model(model_name)
        
        # Load configuration using the fixed method
        config = XttsConfig.load_from_json(output_config_path)
        
        # Initialize model
        model = Xtts.init_from_config(config)
        
        # Load checkpoint and set device
        if use_cpu:
            model.load_checkpoint(config, checkpoint_dir=output_model_path)
            model.cpu()
            logger.info("Model loaded on CPU")

        else:
            model.load_checkpoint(config, checkpoint_dir=output_model_path)
            model.cuda()
            logger.info("Model loaded on CUDA")
        
        logger.info("Model loading completed successfully")
        return model
        
    except Exception as e:
        import traceback
        traceback.print_stack()
        logger.error(f"Failed to load model '{model_name}': {str(e)}")
        raise


def setup_model_seed(seed=None, randomize=False):
    """
    Set up random seed for reproducible model inference
    
    Args:
        seed: Random seed (int). If None, generates random seed
        
    Returns:
        int: The seed that was set
    """
    if randomize:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    elif seed is None:
        seed = 20250527

    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    logger.debug(f"Model seed set to: {seed}")
    return seed


def get_model_device(model):
    """
    Get the device where the model is currently loaded
    
    Args:
        model: The loaded model
        
    Returns:
        torch.device: Device where model is located
    """
    # Check if model has parameters and get device from first parameter
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def validate_model_state(model):
    """
    Validate that model is properly loaded and ready for inference
    
    Args:
        model: The model to validate
        
    Returns:
        bool: True if model is ready
        
    Raises:
        RuntimeError: If model is not properly initialized
    """
    if model is None:
        raise RuntimeError("Model is None - not loaded")
    
    try:
        device = get_model_device(model)
        logger.debug(f"Model validation: device={device}")
        return True
    except Exception as e:
        raise RuntimeError(f"Model validation failed: {str(e)}")


# =============================================================================
# MODEL INFERENCE HELPERS
# =============================================================================

def check_text_length(text, model, language="en", char_limit=None):
    """
    Check if text exceeds model limits and recommend text splitting
    
    Args:
        text: Input text to check
        model: Loaded model with tokenizer
        language: Language code for text
        char_limit: Custom character limit (uses model default if None)
        
    Returns:
        tuple: (should_split, actual_limit)
    """
    if char_limit is None:
        char_limit = getattr(model.tokenizer, 'char_limits', {}).get(language, 250)
    
    should_split = len(text) > char_limit
    
    return should_split, char_limit


def prepare_inference_params(temperature=0.7, top_p=1.0, top_k=50, speed=1.0, 
                           enable_text_splitting=None, **kwargs):
    """
    Prepare and validate inference parameters
    
    Args:
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter  
        speed: Generation speed multiplier
        enable_text_splitting: Whether to enable text splitting
        **kwargs: Additional parameters
        
    Returns:
        dict: Validated inference parameters
    """
    params = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "speed": float(speed)
    }
    
    if enable_text_splitting is not None:
        params["enable_text_splitting"] = bool(enable_text_splitting)
    
    # Add any additional parameters
    params.update(kwargs)
    
    logger.debug(f"Inference parameters: {params}")
    return params


def initialize_model_with_cache(
    use_cpu: bool = False, 
    seed: Optional[int] = None,
    validate: bool = True
) -> Any:
    """
    Complete model initialization with caching setup.
    
    Args:
        use_cpu: Whether to use CPU instead of CUDA
        seed: Random seed for reproducibility (optional)
        validate: Whether to validate model state after loading
        
    Returns:
        Loaded and initialized model
        
    Raises:
        Exception: If model loading fails
    """
    logger.info("Starting model initialization...")
    
    try:
        # Load model
        model = load_model(use_cpu=use_cpu)
        
        setup_model_seed(seed)
        
        # Validate model state
        if validate:
            validate_model_state(model)
        
        # Initialize latent cache import here to avoid circular imports
        from shared_cache_utils import init_latent_cache
        init_latent_cache(model=model, supported_languages=SUPPORTED_LANGUAGE_CODES)
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise