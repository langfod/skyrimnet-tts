#!/usr/bin/env python3
"""
SkyrimNet TTS Unified Application Entry Point
Phase 3: Combined API and Gradio UI in a single application
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path for relative imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import uvicorn
from loguru import logger
from gradio.routes import mount_gradio_app

# Import shared modules
from shared_config import setup_environment, SUPPORTED_LANGUAGE_CODES
from shared_models import load_model, setup_model_seed, validate_model_state
from shared_args import parse_api_args
from utils import init_latent_cache, get_wavout_dir, get_latent_dir, get_speakers_dir

import skyrimnet_api
import skyrimnet_xtts as skyrimnet_gradio




def initialize_logging():
    """Initialize logging configuration"""
    # Remove default logger to avoid conflicts
    logger.remove()
    logger.add(
        sys.stdout, 
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", 
        level="INFO", 
        enqueue=True
    )
    
    # Optional file logging
    LOG_TO_FILE = os.getenv('LOG_TO_FILE') == 'true'
    LOG_FILE_PATH = os.getenv('LOG_FILE_PATH', 'logs/skyrimnet_unified.log')
    
    if LOG_TO_FILE:
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        logger.add(
            LOG_FILE_PATH,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level="INFO",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True
        )
        logger.info(f"File logging enabled. Logs will be written to: {LOG_FILE_PATH}")


def initialize_configuration():
    """Initialize environment and configuration"""
    setup_environment()
    output_temp = get_wavout_dir().parent.absolute()
    latents_dir = get_latent_dir().parent.absolute()
    speakers_dir = get_speakers_dir().parent.absolute()

    os.environ["GRADIO_ALLOWED_PATHS"] = f'"{output_temp}","{latents_dir}","{speakers_dir}"'
    logger.info("Environment initialized")


def initialize_model(use_cpu=False):
    """Initialize and load the TTS model"""
    logger.info("Starting model initialization...")
    
    try:
        # Load model
        model = load_model(use_cpu=use_cpu)
        
        # Setup model seed for reproducibility
        setup_model_seed(20250527)
        
        # Validate model state
        validate_model_state(model)
        
        # Initialize latent cache
        init_latent_cache(model=model, supported_languages=SUPPORTED_LANGUAGE_CODES)
        
        logger.info("Model loaded and initialized successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def create_unified_app(model, args):
    """Create unified application with both API and Gradio UI"""
    
    # Set the global model in both applications
    
    skyrimnet_api.CURRENT_MODEL = model
    skyrimnet_gradio.CURRENT_MODEL = model
    
    # Build Gradio interface
    logger.info("Building Gradio interface...")
    demo = skyrimnet_gradio.build_interface()
    
    # Mount Gradio on FastAPI app
    logger.info("Mounting Gradio interface on FastAPI application...")
    unified_app = mount_gradio_app(skyrimnet_api.app, demo, path="/")
    
    return unified_app


if __name__ == "__main__":
    logger.info("Starting SkyrimNet TTS Unified Application...")
    
    # Parse command line arguments
    extra_args = {
        "--ui-path": {
            "type": str,
            "default": "/",
            "help": "Path where Gradio UI will be mounted (default: /)"
        }
    }
    args = parse_api_args("SkyrimNet TTS Unified Application (API + Gradio UI)", extra_args)
    
    # Initialize logging
    initialize_logging()
    
    # Initialize configuration
    initialize_configuration()
    
    # Initialize model
    try:
        model = initialize_model(use_cpu=args.use_cpu)
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        sys.exit(1)
    
    # Create unified application
    try:
        app = create_unified_app(model, args)
        logger.info("Unified application created successfully")
    except Exception as e:
        logger.error(f"Failed to create unified application: {e}")
        sys.exit(1)
    
    # Start server
    logger.info(f"Starting unified server on {args.server}:{args.port}")
    logger.info(f"API endpoints available at: http://{args.server}:{args.port}/")
    logger.info(f"Gradio UI available at: http://{args.server}:{args.port}{getattr(args, 'ui_path', '/')}")
    logger.info("Available API endpoints:")
    logger.info("  POST /tts_to_audio")
    logger.info("  POST /create_and_store_latents") 
    logger.info("  GET  /health")
    logger.info("  GET  /docs (Swagger API documentation)")
    
    try:
        uvicorn.run(app, host=args.server, port=args.port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Application shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)