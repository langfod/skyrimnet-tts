# SkyrimNet TTS Module

A unified Text-to-Speech application with FastAPI backend and Gradio UI.

## Installation

1. Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

## Usage

### As a Python Module

Run the unified application (API + Gradio UI):
```bash
python -m skyrimnet-xtts
```

### Command Line Options

```bash
python -m skyrimnet-xtts [OPTIONS]
```

**Options:**
- `--server HOST`: Server host address (default: localhost)
- `--port PORT`: Server port number (default: 8020)
- `--use_cpu`: Use CPU instead of CUDA for model inference
- `--deepspeed`: Use DeepSpeed optimization for faster inference

### Examples

**CPU Mode:**
```bash
python -m skyrimnet-xtts --use_cpu --port 8021
```

**CUDA with DeepSpeed:**
```bash
python -m skyrimnet-xtts --deepspeed --port 8020
```

**Custom Host/Port:**
```bash
python -m skyrimnet-xtts --server 0.0.0.0 --port 8030
```

## Endpoints

Once running, the application provides:

### API Endpoints
- `POST /tts_to_audio` - Generate TTS audio from text
- `POST /create_and_store_latents` - Create and store speaker latents
- `GET /health` - Health check endpoint
- `GET /docs` - Swagger API documentation

### Web Interface
- `/ui` - Gradio web interface for interactive TTS

## Accessing the Application

After starting the module, you can access:
- **API**: `http://localhost:8020/`
- **Web UI**: `http://localhost:8020/ui`
- **API Docs**: `http://localhost:8020/docs`

## Module Structure

- `__main__.py` - Main entry point for unified application
- `skyrimnet-api.py` - FastAPI backend implementation
- `skyrimnet-xtts.py` - Gradio UI implementation
- `shared_config.py` - Common configuration and constants
- `shared_models.py` - Model loading and management
- `shared_args.py` - Command line argument parsing
- `utils.py` - Utility functions for TTS operations

## Features

- **Unified Application**: Both REST API and web interface in one process
- **Model Sharing**: Single model instance shared between API and UI
- **Flexible Deployment**: Can run on CPU or CUDA with optional DeepSpeed
- **Comprehensive Logging**: Structured logging with optional file output
- **Caching System**: Intelligent latent caching for improved performance