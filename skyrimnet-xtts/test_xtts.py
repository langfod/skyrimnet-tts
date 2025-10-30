#!/usr/bin/env python3
"""
SkyrimNet TTS Unified Application Entry Point
Phase 3: Combined API and Gradio UI in a single application
"""

import shutil
import sys
import os
from pathlib import Path
import traceback

# Add current directory to Python path for relative imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import uvicorn
from loguru import logger
from gradio.routes import mount_gradio_app

# Import shared modules - Handle both direct execution and module execution
try:
    # Try relative imports first (for module execution: python -m skyrimnet-xtts)
    from .shared_config import SUPPORTED_LANGUAGE_CODES
    from .shared_args import parse_api_args
    from .shared_app_utils import setup_application_logging, initialize_application_environment
    from .shared_models import initialize_model_with_cache
    from .shared_cache_utils import get_wavout_dir, get_latent_dir, get_speakers_dir
    from .shared_audio_utils import generate_audio_file
    from . import skyrimnet_api
    from . import skyrimnet_xtts as skyrimnet_gradio
except ImportError:
    # Fall back to absolute imports (for direct execution or PyInstaller)
    from shared_config import SUPPORTED_LANGUAGE_CODES
    from shared_args import parse_api_args
    from shared_app_utils import setup_application_logging, initialize_application_environment
    from shared_models import initialize_model_with_cache
    from shared_cache_utils import get_wavout_dir, get_latent_dir, get_speakers_dir
    from shared_audio_utils import generate_audio_file

    import skyrimnet_api
    import skyrimnet_xtts as skyrimnet_gradio




def initialize_logging():
    """Initialize logging configuration using shared utility"""
    setup_application_logging(log_file_path='logs/skyrimnet_unified.log', log_to_file=False)
    # Also initialize API logging to ensure consistent format
    skyrimnet_api.initialize_api_logging()


def initialize_configuration():
    """Initialize environment and configuration"""
    output_temp = get_wavout_dir().parent.absolute()
    latents_dir = get_latent_dir().parent.absolute()
    speakers_dir = get_speakers_dir().parent.absolute()

    os.environ["GRADIO_ALLOWED_PATHS"] = f'""assets","{output_temp}","{latents_dir}","{speakers_dir}"'



def initialize_model(use_cpu=False, use_deepspeed=False, use_bfloat16=False):
    """Initialize and load the TTS model using shared utility"""
    return initialize_model_with_cache(
        use_cpu=use_cpu,
        seed=20250527,
        validate=True,
        use_deepspeed=use_deepspeed,
        use_bfloat16=use_bfloat16
    )


def create_unified_app(model, args):
    """Create unified application with both API and Gradio UI"""
    
    # Set the global model in both applications
    skyrimnet_api.CURRENT_MODEL = model
    skyrimnet_gradio.CURRENT_MODEL = model
    
    # Set up API-only catch-all route (for /api/* paths only)
    # This avoids conflicts with Gradio routing while still providing API debugging
    #skyrimnet_api.setup_api_only_catch_all_route()
    
    # Build Gradio interface
    logger.info("Building Gradio interface...")
    demo = skyrimnet_gradio.build_interface()
    
    # Mount Gradio on FastAPI app
    logger.info("Mounting Gradio interface on FastAPI application...")
    unified_app = mount_gradio_app(skyrimnet_api.app, demo, path="/")
    #skyrimnet_api.setup_catch_all_route()  # Setup catch-all route for undefined API paths
    return unified_app


if __name__ == "__main__":
    # Parse command line arguments
    extra_args = {
        "--ui-path": {
            "type": str,
            "default": "/",
            "help": "Path where Gradio UI will be mounted (default: /)"
        }
    }
    args = parse_api_args("SkyrimNet TTS Unified Application (API + Gradio UI)", extra_args)
 
    speaker_embeddings_cache_dir = get_latent_dir("en")
    if speaker_embeddings_cache_dir.exists():
        shutil.rmtree(speaker_embeddings_cache_dir, ignore_errors=True)
    
    # Initialize application environment using shared utility
    initialize_application_environment("SkyrimNet TTS Unified Application")
    
    # Initialize logging using shared utility
    initialize_logging()
    
    # Initialize configuration (Gradio-specific setup)
    initialize_configuration()
    
    # Initialize model using shared utility
    try:
        model = initialize_model(use_cpu=args.use_cpu, use_deepspeed=args.deepspeed, use_bfloat16=args.use_bfloat16)
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
    

    inference_kwargs = {}
    inference_kwargs['temperature'] = 0.9
    inference_kwargs['top_p'] = 1.0
    inference_kwargs['top_k'] = 50
    inference_kwargs['speed'] = 1.0
    inference_kwargs['repetition_penalty'] = 2.1
    try:
        text="The Silver Bloods. They have a whole mine filled with prisoners to dig up silver ore, get smelted by workers they pay, and they own half the city."
        speaker_audio="malebrute"
        language="en"
        speaker_audio_uuid=None
        #print(wav)

        wav_out_path = generate_audio_file(
        model=model,
        language=language,
        speaker_wav=speaker_audio,
        text=text,
        stream=True,
        **inference_kwargs
        )

        wav_out_path = generate_audio_file(
        model=model,
        language=language,
        speaker_wav=speaker_audio,
        text=text,
        stream=True,
        **inference_kwargs
        )
        

        logger.info(f"Audio generated and saved to: {wav_out_path}")
    except Exception as e:
        traceback.print_exc()

        logger.error(f"Audio generation failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Application shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)