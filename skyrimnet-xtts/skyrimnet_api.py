
#!/usr/bin/env python3
"""
SkyrimNet Simplified FastAPI TTS Service
Simplified FastAPI service modeling APIs from xtts_api_server but using methodology from skyrimnet-xtts.py
"""

# Standard library imports
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Third-party imports
import uvicorn
import time
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

# Local imports - Handle both direct execution and module execution
try:
    # Try relative imports first (for module execution: python -m skyrimnet-xtts)
    from .shared_cache_utils import get_latent_from_audio
    from .shared_config import SUPPORTED_LANGUAGE_CODES, validate_language
    from .shared_args import parse_api_args
    from .shared_audio_utils import generate_audio_file
    from .shared_app_utils import setup_application_logging, initialize_application_environment
    from .shared_models import initialize_model_with_cache, setup_model_seed
except ImportError:
    # Fall back to absolute imports (for direct execution: python skyrimnet_api.py)
    from shared_cache_utils import get_latent_from_audio
    from shared_config import SUPPORTED_LANGUAGE_CODES, validate_language
    from shared_args import parse_api_args
    from shared_audio_utils import generate_audio_file
    from shared_app_utils import setup_application_logging, initialize_application_environment
    from shared_models import initialize_model_with_cache, setup_model_seed
# =============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# =============================================================================

# Global model state
CURRENT_MODEL = None
IGNORE_PING = False
CACHED_TEMP_DIR = None
# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

args = parse_api_args("SkyrimNet Simplified TTS API")

# =============================================================================
# LOGGING SETUP
# =============================================================================

# Global flag to track if logging has been initialized
_LOGGING_INITIALIZED = False

def initialize_api_logging():
    """Initialize logging for the API module"""
    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        # Setup standardized logging (only when not already configured)
        setup_application_logging()
        _LOGGING_INITIALIZED = True

# Only setup logging when running as standalone script
if __name__ == "__main__":
    initialize_api_logging()

# =============================================================================
# PYDANTIC REQUEST/RESPONSE MODELS
# =============================================================================

class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: Optional[str] = None
    language: Optional[str] = "en"
    accent: Optional[str] = None
    save_path: Optional[str] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_cached_temp_dir():
    """Get or create the cached temporary directory"""
    global CACHED_TEMP_DIR
    
    if CACHED_TEMP_DIR is None:
        CACHED_TEMP_DIR = Path(tempfile.mkdtemp(prefix="skyrimnet_tts_"))
        logger.info(f"Created cached temp directory: {CACHED_TEMP_DIR}")
    elif not CACHED_TEMP_DIR.exists():
        # Recreate if it was somehow deleted
        CACHED_TEMP_DIR = Path(tempfile.mkdtemp(prefix="skyrimnet_tts_"))
        logger.info(f"Recreated cached temp directory: {CACHED_TEMP_DIR}")
    
    return CACHED_TEMP_DIR

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

app = FastAPI(title="SkyrimNet TTS API", description="Simplified TTS API service", version="1.0.0")

# Request logging middleware (logs ALL requests, even undefined endpoints)
#@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log the incoming request
    logger.info(f"📥 INCOMING REQUEST: {request.method} {request.url}")
    logger.info(f"   Headers: {dict(request.headers)}")
    logger.info(f"   Client: {request.client.host if request.client else 'unknown'}")
    
    # Log query parameters if any
    if request.query_params:
        logger.info(f"   Query params: {dict(request.query_params)}")
    
    # Try to log request body for POST requests (be careful with large files)
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                # For JSON requests, we can log the body
                body = await request.body()
                if len(body) < 2000:  # Only log small bodies
                    logger.info(f"   Body: {body.decode('utf-8')}")
                else:
                    logger.info(f"   Body: <large body {len(body)} bytes>")
            elif "multipart/form-data" in content_type:
                logger.info(f"   Body: <multipart form data>")
            else:
                logger.info(f"   Body: <{content_type}>")
        except Exception as e:
            logger.warning(f"   Body: <failed to read body: {e}>")
    
    # Process the request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log the response
        logger.info(f"📤 RESPONSE: {response.status_code} for {request.method} {request.url.path}")
        logger.info(f"   Processing time: {process_time:.4f}s")
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"❌ REQUEST FAILED: {request.method} {request.url.path}")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Processing time: {process_time:.4f}s")
        raise

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API ENDPOINTS
# =============================================================================


##@app.post("/tts_to_audio")
@app.post("/tts_to_audio/")
async def tts_to_audio(request: SynthesisRequest, background_tasks: BackgroundTasks):
    """
    Generate TTS audio from text with specified speaker voice
    """
    global IGNORE_PING
    try:
        logger.info(f"Post tts_to_audio - Processing TTS to audio with request: "
                   f"text='{request.text}' speaker_wav='{request.speaker_wav}' "
                   f"language='{request.language}' accent={request.accent} save_path='{request.save_path}'")
        
        if not CURRENT_MODEL:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")
        

        if request.text == "ping" and (request.speaker_wav == 'maleeventoned' or request.speaker_wav == 'player voice') and not IGNORE_PING:
            IGNORE_PING = True
            return FileResponse(
                path="assets/silence_100ms.wav",
                filename=request.save_path,
                media_type="audio/wav"
            )
        IGNORE_PING = True
        
        try:
            language = validate_language(request.language or "en")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Get latents from speaker
        speaker_wav = request.speaker_wav or "malecommoner"
        text = request.text

        wav_out_path = generate_audio_file(
            model=CURRENT_MODEL,
            language=language,
            speaker_wav=speaker_wav,
            text=text
        )
                    
        return FileResponse(
            path=str(wav_out_path),
            filename=request.save_path,
            media_type="audio/wav"
        )            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /tts_to_audio - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/create_and_store_latents")
async def create_and_store_latents(
    speaker_name: str = Form(...),
    language: str = Form("en"),
    wav_file: UploadFile = File(...)
):
    setup_model_seed()
    """
    Create and store latent embeddings from uploaded audio file
    """    
    try:
        logger.info(f"POST /create_and_store_latents - Creating and storing latents for speaker: {speaker_name}, language: {language}, file: {wav_file.filename}")
        
        if not CURRENT_MODEL:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Validate language
        try:
            language = validate_language(language)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Validate file type
        if not wav_file.filename.endswith('.wav'):
            raise HTTPException(status_code=400, detail="Only WAV files are supported")
        
        # Use cached temp directory and create unique filename
        temp_dir = get_cached_temp_dir()
        #unique_filename = f"{uuid.uuid4().hex}_{wav_file.filename}"
        temp_audio_path = temp_dir.joinpath(wav_file.filename)
        
        try:
            with open(temp_audio_path, "wb") as buffer:
                content = await wav_file.read()
                buffer.write(content)            
            
            # Get latents from audio
            gpt_cond_latent, speaker_embedding = get_latent_from_audio(
                CURRENT_MODEL, language, str(temp_audio_path))
            
            logger.info(f"Successfully created latents for speaker: {speaker_name}")
            
            return {
                "message": f"Latents created and stored for speaker '{speaker_name}' in language '{language}'",
                "speaker_name": speaker_name,
                "language": language,
                "latent_shapes": {
                    "gpt_cond_latent": list(gpt_cond_latent.shape),
                    "speaker_embedding": list(speaker_embedding.shape)
                }
            }
            
        finally:
            temp_audio_path.unlink(missing_ok=True)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /create_and_store_latents - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": CURRENT_MODEL is not None,
        "supported_languages": SUPPORTED_LANGUAGE_CODES
    }


# =============================================================================
# CATCH-ALL ROUTE CONFIGURATION
# =============================================================================

def setup_catch_all_route():
    """
    Set up catch-all route for undefined API endpoints.
    This should only be called when NOT mounting Gradio UI.
    """
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def catch_undefined_endpoints(request: Request, path: str):
        """
        Catch-all route to log attempts to access undefined endpoints
        This helps with debugging missing routes and API discovery
        """
        logger.warning(f"🚫 UNDEFINED ENDPOINT: {request.method} /{path}")
        logger.warning(f"   Full URL: {request.url}")
        logger.warning(f"   Available endpoints:")
        logger.warning(f"     POST /tts_to_audio")
        logger.warning(f"     POST /create_and_store_latents") 
        logger.warning(f"     GET  /health")
        logger.warning(f"     GET  /docs (Swagger UI)")
        logger.warning(f"     GET  /redoc (ReDoc)")
        
        raise HTTPException(
            status_code=404, 
            detail={
                "error": f"Endpoint not found: {request.method} /{path}",
                "available_endpoints": [
                    "POST /tts_to_audio",
                    "POST /create_and_store_latents",
                    "GET /health",
                    "GET /docs",
                    "GET /redoc"
                ]
            }
        )


def setup_api_only_catch_all_route():
    """
    Set up a limited catch-all route that only catches API paths when Gradio is mounted.
    This avoids conflicts with Gradio's routing while still providing API endpoint discovery.
    """
    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def catch_undefined_api_endpoints(request: Request, path: str):
        """
        Catch-all route for undefined /api/* endpoints only
        This helps with API debugging without interfering with Gradio
        """
        logger.warning(f"🚫 UNDEFINED API ENDPOINT: {request.method} /api/{path}")
        logger.warning(f"   Full URL: {request.url}")
        logger.warning(f"   Available API endpoints:")
        logger.warning(f"     POST /tts_to_audio")
        logger.warning(f"     POST /create_and_store_latents") 
        logger.warning(f"     GET  /health")
        logger.warning(f"     GET  /docs (Swagger UI)")
        logger.warning(f"     GET  /redoc (ReDoc)")
        
        raise HTTPException(
            status_code=404, 
            detail={
                "error": f"API endpoint not found: {request.method} /api/{path}",
                "available_endpoints": [
                    "POST /tts_to_audio",
                    "POST /create_and_store_latents",
                    "GET /health",
                    "GET /docs",
                    "GET /redoc"
                ],
                "note": "For the Gradio UI, visit the root path '/'"
            }
        )

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize application environment
    initialize_application_environment("SkyrimNet TTS API")
    
    # Set up full catch-all route for standalone API mode
    setup_catch_all_route()
    
    # Load model with standardized initialization
    try:
        CURRENT_MODEL = initialize_model_with_cache(
            use_cpu=args.use_cpu,
            validate=True
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Start server
    logger.info(f"Starting server on {args.server}:{args.port}")
    uvicorn.run(
        app, 
        host=args.server, 
        port=args.port, 
        log_level="info",
        access_log=False,  # Disable uvicorn's access logging to use our format
        log_config=None    # Use default Python logging instead of uvicorn's custom format
    )
