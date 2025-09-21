#!/usr/bin/env python3
"""
Zonos Text-to-Speech Application with Gradio Interface
Enhanced with disk and memory caching for speaker embeddings
"""

# Standard library imports
import functools
from pathlib import Path
import sys
import time


# Third-party imports
import gradio as gr
import time
import torch
from loguru import logger
from utils import get_latent_from_audio, init_latent_cache, save_torchaudio_wav, get_wavout_dir, get_latent_dir, get_speakers_dir, get_cache_key

# Shared module imports
from shared_config import setup_environment, SUPPORTED_LANGUAGE_CODES, DEFAULT_TTS_PARAMS, DEFAULT_CACHE_CONFIG, validate_language
from shared_models import load_model, check_text_length
from shared_args import parse_gradio_args

# =============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# =============================================================================

# Initialize environment
setup_environment()

# Global model state
CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None
SPEAKER_EMBEDDING = None
SPEAKER_AUDIO_PATH = None

# Cache flags - defaults that can be overridden by skyrimnet_config.txt
ENABLE_DISK_CACHE = DEFAULT_CACHE_CONFIG["ENABLE_DISK_CACHE"]
ENABLE_MEMORY_CACHE = DEFAULT_CACHE_CONFIG["ENABLE_MEMORY_CACHE"]
_CONFIG_CACHE = None
_CONFIG_FILE_PATH = "skyrimnet_config.txt"
# Testing flag - when True, bypasses config loading and uses all API values
_USE_API_MODE = False

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

args = parse_gradio_args("Zonos Text-to-Speech Application with Gradio Interface")

# =============================================================================
# Support Functions
# =============================================================================

def load_skyrimnet_config():
    """Load configuration from skyrimnet_config.txt with error handling"""
    global _CONFIG_CACHE, ENABLE_MEMORY_CACHE, ENABLE_DISK_CACHE
    
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    
    # Default configuration - using shared defaults from shared_config
    default_config = {
        'temperature': DEFAULT_TTS_PARAMS["TEMPERATURE"],
        'top_p': DEFAULT_TTS_PARAMS["TOP_P"],
        'top_k': DEFAULT_TTS_PARAMS["TOP_K"],
        'speed': DEFAULT_TTS_PARAMS["SPEED"],
        'repetition_penalty': DEFAULT_TTS_PARAMS["REPETITION_PENALTY"],
    }
    
    global_flags = {
        'enable_memory_cache': ENABLE_MEMORY_CACHE,
        'enable_disk_cache': ENABLE_DISK_CACHE
    }
    
    config_mode = {
        'temperature': 'default',
        'top_p': 'default',
        'top_k': 'default',
        'speed': 'default',
        'repetition_penalty': 'default'
    }
    
    try:
        config_path = Path(_CONFIG_FILE_PATH)
        if not config_path.exists():
            logger.warning(f"Config file {_CONFIG_FILE_PATH} not found, using hardcoded defaults")
            _CONFIG_CACHE = (default_config, config_mode, global_flags)
            return _CONFIG_CACHE
            
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
                
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle global boolean flags
                if key in global_flags:
                    if value.lower() in ['true', 'yes', '1', 'on']:
                        global_flags[key] = True
                        # Update global variables
                        if key == 'enable_memory_cache':
                            ENABLE_MEMORY_CACHE = True
                        elif key == 'enable_disk_cache':
                            ENABLE_DISK_CACHE = True
                        logger.info(f"Setting {key} to True")
                    elif value.lower() in ['false', 'no', '0', 'off']:
                        global_flags[key] = False
                        # Update global variables
                        if key == 'enable_memory_cache':
                            ENABLE_MEMORY_CACHE = False
                        elif key == 'enable_disk_cache':
                            ENABLE_DISK_CACHE = False
                        logger.info(f"Setting {key} to False")
                    else:
                        logger.warning(f"Invalid boolean value '{value}' for {key}, using default")
                
                # Handle parameter modes
                elif key in config_mode:
                    if value.lower() == 'default':
                        config_mode[key] = 'default'
                    elif value.lower() == 'api':
                        config_mode[key] = 'api'
                    else:
                        try:
                            custom_value = float(value)
                            config_mode[key] = 'custom'
                            default_config[key] = custom_value
                            logger.info(f"Using custom {key} value: {custom_value}")
                        except ValueError:
                            logger.warning(f"Invalid value '{value}' for {key}, using default")
                            
        logger.info(f"Loaded config: {config_mode}")
        logger.info(f"Global flags: {global_flags}")
        _CONFIG_CACHE = (default_config, config_mode, global_flags)
        return _CONFIG_CACHE
        
    except Exception as e:
        logger.error(f"Error reading config file {_CONFIG_FILE_PATH}: {e}, using hardcoded defaults")
        _CONFIG_CACHE = (default_config, config_mode, global_flags)
        return _CONFIG_CACHE

def get_config_value(param_name, api_value, defaults, modes, bypass_config=False):
    """Get the appropriate value based on configuration mode"""
    if bypass_config:
        # API mode: use API value with fallback to shared defaults
        fallback_defaults = {
            'temperature': DEFAULT_TTS_PARAMS["TEMPERATURE"],
            'top_p': DEFAULT_TTS_PARAMS["TOP_P"],
            'top_k': DEFAULT_TTS_PARAMS["TOP_K"],
            'speed': DEFAULT_TTS_PARAMS["SPEED"],
            'repetition_penalty': DEFAULT_TTS_PARAMS["REPETITION_PENALTY"],
        }
        return api_value if api_value is not None else fallback_defaults.get(param_name, 0.0)
    
    mode = modes.get(param_name, 'default')
    
    if mode == 'api':
        return api_value if api_value is not None else defaults[param_name]
    else:  # 'default' or 'custom'
        return defaults[param_name]


def reload_config():
    """Force reload of configuration file"""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
    return load_skyrimnet_config()


def set_seed(seed: int):
    """
    Set random seeds for reproducible generation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@functools.cache
def cpp_uuid_to_seed(uuid_64: int) -> int:
    """
    Convert a 64-bit UUID to a valid PyTorch seed (0 to 2^32 - 1).
    Uses hash() for better distribution across the seed space.
    """
    return abs(hash(uuid_64)) % (2**32)


# =============================================================================
# MAIN APPLICATION FUNCTIONS
# =============================================================================


def generate_audio(model_choice=None, text=None, language="en", speaker_audio=None, prefix_audio=None, e1=None, e2=None, e3=None, e4=None, e5=None, e6=None, e7=None, e8=None,
                  vq_single=None, fmax=None, pitch_std=None, speaking_rate=None, dnsmos_ovrl=None, speaker_noised=None, cfg_scale=None, top_p=None,
                  top_k=None, min_p=None, linear=None, confidence=None, quadratic=None, seed=None, randomize_seed=None, unconditional_keys=None,
                  ):
    """
    Generates audio based on the provided UI parameters with enhanced caching.
    """
    language = validate_language(language)
    logger.info(f"inputs: text={text}, language={language}, speaker_audio={Path(speaker_audio).stem if speaker_audio else 'None'}, seed={seed}")
    # Start timing the entire function
    func_start_time = time.perf_counter()

    # Load config (or use empty values for API mode)
    if _USE_API_MODE:
        defaults, modes = {}, {}
    else:
        defaults, modes, flags = load_skyrimnet_config()

    # Convert parameters to appropriate types
    #speaker_noised_bool = bool(speaker_noised)
    #fmax = float(fmax)
    #pitch_std = float(pitch_std)
    #speaking_rate = float(speaking_rate)
    #dnsmos_ovrl = float(dnsmos_ovrl)
    #cfg_scale = float(cfg_scale)
    #top_p = float(top_p)
    #top_k = int(top_k)
    #min_p = float(min_p)
    #linear = float(linear)
    #confidence = float(confidence)
    #quadratic = float(quadratic)
    #seed = int(seed)

    # Handle speaker audio caching
    global SPEAKER_AUDIO_PATH, SPEAKER_EMBEDDING

    speaker_audio_uuid = seed

    seed = torch.randint(0, 2**32 - 1, (1,)).item() if seed is None or randomize_seed else cpp_uuid_to_seed(seed)
    torch.manual_seed(seed)
    
    enable_text_splitting, char_limit = check_text_length(text, CURRENT_MODEL, language)
    if enable_text_splitting:
        logger.warning(f"Text length {len(text)} exceeds limit {char_limit} for language '{language}'. Enabling text splitting.")
    
    if speaker_audio is None or speaker_audio.strip() == "":
        speaker_audio = "malebrute.wav"

    gpt_cond_latent, speaker_embedding = get_latent_from_audio(CURRENT_MODEL, language, speaker_audio, speaker_audio_uuid)
    
    # Get effective parameter values using config system
    effective_temperature = get_config_value('temperature', None, defaults, modes, _USE_API_MODE)
    effective_top_p = get_config_value('top_p', top_p, defaults, modes, _USE_API_MODE)
    effective_top_k = get_config_value('top_k', top_k, defaults, modes, _USE_API_MODE)
    effective_speed = get_config_value('speed', speaking_rate, defaults, modes, _USE_API_MODE)
    effective_repetition_penalty = get_config_value('repetition_penalty', confidence, defaults, modes, _USE_API_MODE)
    
    wav_out = CURRENT_MODEL.inference(
    text=text, language=language,
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    speed=effective_speed,
    top_p=effective_top_p,
    top_k=effective_top_k,
    temperature=effective_temperature,
    repetition_penalty=effective_repetition_penalty,
    enable_text_splitting=enable_text_splitting,
    )
    wav_out_path = save_torchaudio_wav(wav_tensor=torch.tensor(wav_out["wav"]).unsqueeze(0), sr=24000, audio_path=speaker_audio, uuid=speaker_audio_uuid)

    #save_torchaudio_wav(wav_tensor, sr, audio_path, uuid):
    # Log execution time
    func_end_time = time.perf_counter()
    total_duration_s = func_end_time - func_start_time
    if speaker_audio:
        logger.info(f"Total 'generate_audio' for {speaker_audio.split('\\')[-1]} execution time: {total_duration_s:.2f} seconds")
    else:
        logger.info(f"Total 'generate_audio' execution time: {total_duration_s:.2f} seconds")
    sys.stdout.flush()

    return wav_out_path, speaker_audio_uuid


def build_interface():
    """Build and return the Gradio interface with cache management."""
    output_temp = get_wavout_dir().parent.absolute()
    latents_dir = get_latent_dir().parent.absolute()
    speakers_dir = get_speakers_dir().parent.absolute()

    gr.set_static_paths([output_temp, latents_dir, speakers_dir])
    with gr.Blocks(analytics_enabled=False, title="XTTS") as demo:
        gr.Markdown("# XTTS with Speaker Embedding Cache")

        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text to Synthesize",
                    value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                    lines=4)
                language = gr.Dropdown(choices=SUPPORTED_LANGUAGE_CODES, value="en", label="Language Code", allow_custom_value=True)

            with gr.Column():
                speaker_audio = gr.Audio(label="Optional Speaker Audio (for cloning)", type="filepath",sources=["upload", "microphone"], value="assets/malebrute.wav")
                enable_text_splitting = gr.Checkbox(value=True, label="Enable Text Splitting")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Conditioning Parameters")
                temperature = gr.Slider(0, 2, value=1, step=0.01, label="Temperature")
                top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
                top_p = gr.Slider(0, 1, value=1, step=0.01, label="Top-P")
                speed = gr.Slider(0.5, 2, value=1.0, step=0.01, label="Speed")

        with gr.Column():
            generate_button = gr.Button("Generate Audio", variant="primary")
            output_audio = gr.Audio(label="Generated Audio", type="filepath", autoplay=True)

        model_choice = gr.Textbox(visible=False)
        #language = gr.Textbox(visible=False)
        #speaker_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None, visible=False)
        prefix_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None, visible=False)
        emotion1 = gr.Number(visible=False)
        emotion2 = gr.Number(visible=False)
        emotion3 = gr.Number(visible=False)
        emotion4 = gr.Number(visible=False)
        emotion5 = gr.Number(visible=False)
        emotion6 = gr.Number(visible=False)
        emotion7 = gr.Number(visible=False)
        emotion8 = gr.Number(visible=False)
        vq_single = gr.Number(visible=False)
        fmax = gr.Number(visible=False)
        pitch_std = gr.Number(visible=False)
        speaking_rate = gr.Number(visible=False)
        dnsmos = gr.Number(visible=False)
        speaker_noised_checkbox = gr.Checkbox(visible=False)
        cfg_scale = gr.Number(visible=False)
        min_k = gr.Number(visible=False)
        min_p = gr.Number(visible=False)
        linear = gr.Number(visible=False)
        confidence = gr.Number(visible=False)
        quadratic = gr.Number(visible=False)
        randomize_seed_toggle = gr.Checkbox(visible=False)
        unconditional_keys = gr.Textbox(visible=False)
        seed_number = gr.Number(visible=False)

        generate_button.click(fn=generate_audio,
            inputs=[model_choice, text, language, speaker_audio, prefix_audio, emotion1, emotion2, emotion3, emotion4,
                emotion5, emotion6, emotion7, emotion8, vq_single, fmax, pitch_std,
                speaking_rate, dnsmos, speaker_noised_checkbox, cfg_scale, top_p,
                min_k, min_p, linear, confidence, quadratic, seed_number,
                randomize_seed_toggle, unconditional_keys, ], outputs=[output_audio, seed_number], )



    return demo


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    CURRENT_MODEL = load_model(use_cpu=args.use_cpu)

    init_latent_cache(model=CURRENT_MODEL, supported_languages=SUPPORTED_LANGUAGE_CODES)
    wav, _ = generate_audio(text="This is a test.", speaker_audio="assets/malebrute.wav", language="en")

    demo = build_interface()
    demo.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser)