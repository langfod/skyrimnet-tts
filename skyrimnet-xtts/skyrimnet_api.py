
#!/usr/bin/env python3
"""
SkyrimNet Simplified FastAPI TTS Service
Simplified FastAPI service modeling APIs from xtts_api_server but using methodology from skyrimnet-xtts.py
"""

# Standard library imports
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# Third-party imports
import uvicorn
import torch
import time
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

# Local imports
from utils import get_latent_from_audio, get_speakers_dir, init_latent_cache, save_torchaudio_wav
from shared_config import setup_environment, SUPPORTED_LANGUAGE_CODES, DEFAULT_TTS_PARAMS, validate_language
from shared_models import load_model, setup_model_seed, validate_model_state, check_text_length
from shared_args import parse_api_args

# =============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# =============================================================================

# Initialize environment
setup_environment()

# Global model state
CURRENT_MODEL = None

# Hardcoded constants for Phase 1 (using shared defaults)
DEFAULT_TEMPERATURE = DEFAULT_TTS_PARAMS["TEMPERATURE"]
DEFAULT_TOP_P = DEFAULT_TTS_PARAMS["TOP_P"]
DEFAULT_TOP_K = DEFAULT_TTS_PARAMS["TOP_K"]
DEFAULT_SPEED = DEFAULT_TTS_PARAMS["SPEED"]

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

args = parse_api_args("SkyrimNet Simplified TTS API")

# =============================================================================
# LOGGING SETUP
# =============================================================================

# Remove default logger to avoid conflicts
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO", enqueue=True)

# Optional file logging
LOG_TO_FILE = os.getenv('LOG_TO_FILE') == 'true'
LOG_FILE_PATH = os.getenv('LOG_FILE_PATH', 'logs/skyrimnet_api.log')

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

# =============================================================================
# PYDANTIC REQUEST/RESPONSE MODELS
# =============================================================================

class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: Optional[str] = None
    language: Optional[str] = "en"
    accent: Optional[str] = None
    save_path: Optional[str] = None


class CreateLatentsRequest(BaseModel):
    speaker_name: str
    language: str = "en"

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_latents_from_speaker_path(speaker_wav: str, language: str, model) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get latents from speaker path - either from cache or by computing from audio file"""
    if not speaker_wav:
        raise HTTPException(status_code=400, detail="speaker_wav is required")

    # First, try to load from cache using the speaker name as cache key
    cached_latents = get_latent_from_audio(model, language, speaker_wav, latents_only=True)
    if cached_latents:
        return cached_latents

    # Check in speakers folder for actual audio file
    speaker_wav_wav = get_speakers_dir(language).joinpath(f"{speaker_wav}.wav")
    if speaker_wav_wav.exists():
        # Compute latents from the audio file
        logger.info(f"Computing latents from audio file: {speaker_wav_wav}")
        return get_latent_from_audio(model, language, str(speaker_wav_wav))

    raise HTTPException(
        status_code=404, 
        detail=f"Speaker '{speaker_wav}' not found in speakers folder or cache."
    )


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
                if len(body) < 1000:  # Only log small bodies
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
    try:
        logger.info(f"Post tts_to_audio - Processing TTS to audio with request: "
                   f"text='{request.text}' speaker_wav='{request.speaker_wav}' "
                   f"language='{request.language}' accent={request.accent} save_path='{request.save_path}'")
        
        if not CURRENT_MODEL:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Validate inputs
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        try:
            language = validate_language(request.language or "en")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if request.text == "ping" and (request.speaker_wav == 'maleeventoned' or request.speaker_wav == 'player voice'):
            return FileResponse(
                path="assets/silence_100ms.wav",
                filename=request.save_path,
                media_type="audio/wav"
            )
        
        # Get latents from speaker
        gpt_cond_latent, speaker_embedding = get_latents_from_speaker_path(request.speaker_wav, language, CURRENT_MODEL)
        
        # Check text length and enable splitting if needed
        enable_text_splitting, char_limit = check_text_length(request.text, CURRENT_MODEL, language)
        if enable_text_splitting:
            logger.info(f"Text length {len(request.text)} exceeds limit {char_limit}, enabling text splitting")
        
        # Generate audio
        logger.info("Running model inference...")
        wav_out = CURRENT_MODEL.inference(
            text=request.text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            speed=DEFAULT_SPEED,
            top_p=DEFAULT_TOP_P,
            top_k=DEFAULT_TOP_K,
            temperature=DEFAULT_TEMPERATURE,
            enable_text_splitting=enable_text_splitting,
        )
        
        # Save audio file
        wav_out_path = save_torchaudio_wav(
            wav_tensor=torch.tensor(wav_out["wav"]).unsqueeze(0),
            sr=24000,
            audio_path=request.speaker_wav,
        )
        
        logger.info(f"Generated audio saved to: {wav_out_path}")       
            
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
        
        # Save uploaded file temporarily
        temp_dir = Path(tempfile.mkdtemp())
        temp_audio_path = temp_dir / wav_file.filename
        
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
            # Cleanup temporary file
            try:
                if temp_audio_path.exists():
                    temp_audio_path.unlink()
                if temp_dir.exists():
                    temp_dir.rmdir()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {e}")
                
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


### Catch-all route for undefined endpoints (must be last)
#@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
#async def catch_undefined_endpoints(request: Request, path: str):
#    """
#    Catch-all route to log attempts to access undefined endpoints
#    This helps with debugging missing routes and API discovery
#    """
#    logger.warning(f"🚫 UNDEFINED ENDPOINT: {request.method} /{path}")
#    logger.warning(f"   Full URL: {request.url}")
#    logger.warning(f"   Available endpoints:")
#    logger.warning(f"     POST /tts_to_audio")
#    logger.warning(f"     POST /create_and_store_latents") 
#    logger.warning(f"     GET  /health")
#    logger.warning(f"     GET  /docs (Swagger UI)")
#    logger.warning(f"     GET  /redoc (ReDoc)")
#    
#    raise HTTPException(
#        status_code=404, 
#        detail={
#            "error": f"Endpoint not found: {request.method} /{path}",
#            "available_endpoints": [
#                "POST /tts_to_audio",
#                "POST /create_and_store_latents",
#                "GET /health",
#                "GET /docs",
#                "GET /redoc"
#            ]
#        }
#    )

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting SkyrimNet TTS API...")
    
    # Load model
    try:
        CURRENT_MODEL = load_model(use_cpu=args.use_cpu)
        
        # Setup model seed for reproducibility
        setup_model_seed(5272025)
        
        # Validate model state
        validate_model_state(CURRENT_MODEL)
        
        # Initialize latent cache
        init_latent_cache(model=CURRENT_MODEL, supported_languages=SUPPORTED_LANGUAGE_CODES)
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Start server
    logger.info(f"Starting server on {args.server}:{args.port}")
    uvicorn.run(app, host=args.server, port=args.port, log_level="info")
