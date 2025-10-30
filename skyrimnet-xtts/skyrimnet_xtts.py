#!/usr/bin/env python3
"""
Zonos Text-to-Speech Application with Gradio Interface
Enhanced with disk and memory caching for speaker embeddings
"""

# Standard library imports
import os
from pathlib import Path
import uuid


# Third-party imports
import gradio as gr
from loguru import logger

# Handle both direct execution and module execution
try:
    # Try relative imports first (for module execution: python -m skyrimnet-xtts)
    from .shared_cache_utils import get_wavout_dir, get_latent_dir, get_speakers_dir
    from .shared_config import SUPPORTED_LANGUAGE_CODES, DEFAULT_CACHE_CONFIG, validate_language, load_skyrimnet_config
    from .shared_args import parse_gradio_args
    from .shared_audio_utils import generate_audio_file
    from .shared_app_utils import initialize_application_environment
    from .shared_models import initialize_model_with_cache, setup_model_seed
except ImportError:
    # Fall back to absolute imports (for direct execution: python skyrimnet_xtts.py)
    from shared_cache_utils import get_wavout_dir, get_latent_dir, get_speakers_dir
    from shared_config import SUPPORTED_LANGUAGE_CODES, DEFAULT_CACHE_CONFIG, validate_language, load_skyrimnet_config
    from shared_args import parse_gradio_args
    from shared_audio_utils import generate_audio_file
    from shared_app_utils import initialize_application_environment
    from shared_models import initialize_model_with_cache, setup_model_seed

# =============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# =============================================================================

# Global model state
CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None
IGNORE_PING = None
SILENCE_AUDIO_PATH = "assets/silence_100ms.wav"
# Cache flags - defaults that can be overridden by skyrimnet_config.txt
ENABLE_DISK_CACHE = DEFAULT_CACHE_CONFIG["ENABLE_DISK_CACHE"]
ENABLE_MEMORY_CACHE = DEFAULT_CACHE_CONFIG["ENABLE_MEMORY_CACHE"]
# Testing flag - when True, bypasses config loading and uses all API values
_USE_API_MODE = False
_FROM_GRADIO = False
STREAM = False
# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

args = parse_gradio_args("XTTS Text-to-Speech Application with Gradio Interface")

# =============================================================================
# Support Functions
# =============================================================================

def get_config_override(param_name, api_value):
    """Get config override value, only using API value in API mode"""
    config_overrides = load_skyrimnet_config()
    config_value = config_overrides.get(param_name)
    
    # Only use API values if explicitly in API mode or from Gradio web interface
    if (_USE_API_MODE or _FROM_GRADIO) and api_value is not None:
        return api_value
    
    # Otherwise, only return config file value (ignore API value)
    return config_value


# =============================================================================
# MAIN APPLICATION FUNCTIONS
# =============================================================================

def generate_audio(model_choice:str=None, text:str=None, language:str="en", speaker_audio:str=None, prefix_audio:str=None,
                    e1:float=None, e2:float=None, e3:float=None, e4:float=None, e5:float=None, e6:float=None, e7:float=None, e8:float=None,
                  vq_single:float=None, fmax:int=None, pitch_std:float=None, speaking_rate:float=None, dnsmos_ovrl:float=None, speaker_noised:float=None, cfg_scale:float=None, top_p:float=None,
                  min_k:float=None, min_p:float=None, linear:float=None, confidence:float=None, quadratic:float=None, seed:int=None, randomize_seed:bool=None, unconditional_keys:float=None
                  ) -> tuple[Path, int]:
    """
    Generates audio based on the provided UI parameters with enhanced caching.
    """
    global IGNORE_PING
    job_id = seed

    language = validate_language(language)

    if isinstance(speaker_audio, dict) and 'path' in speaker_audio:
        speaker_audio = speaker_audio['path']
    logger.info(f"inputs: text={text}, language={language}, speaker_audio={Path(speaker_audio).stem if speaker_audio else 'None'}, seed={seed}")

    if text == "ping":
       if IGNORE_PING is None:
          IGNORE_PING = "pending"
       else:
          logger.info("Ping request received, sending silence audio.")
          return SILENCE_AUDIO_PATH, job_id
    
    setup_model_seed(randomize=randomize_seed)

    if speaker_audio is None or speaker_audio.strip() == "":
        speaker_audio = "malebrute"

    # Get parameter overrides - only pass non-None values
    inference_kwargs = {}   

    # Only use API parameters when explicitly from Gradio web interface or API mode is enabled
    use_api_params = _FROM_GRADIO or _USE_API_MODE

    # Convert parameters if we're using API mode 
    if use_api_params and not _FROM_GRADIO:
        # Convert parameters for Gradio API calls
        speaking_rate = float(speaking_rate) if speaking_rate is not None else None
        top_p = float(top_p) if top_p is not None else None
        min_k = int(min_k) if min_k is not None else None
        linear = float(linear) if linear is not None else None
        confidence = float(confidence) if confidence is not None else None
    elif _FROM_GRADIO or _USE_API_MODE:
        # Convert parameters for web UI calls
        speaking_rate = float(speaking_rate)
        top_p = float(top_p)
        min_k = int(min_k)
        linear = float(linear)
        confidence = float(confidence)
    
    if use_api_params:
        # Use API parameters directly when in API mode (Gradio web interface)
        if linear is not None:
            inference_kwargs['temperature'] = linear
        if top_p is not None:
            inference_kwargs['top_p'] = top_p
        if min_k is not None:
            inference_kwargs['top_k'] = min_k
        if speaking_rate is not None:
            inference_kwargs['speed'] = speaking_rate
        if confidence is not None:
            inference_kwargs['repetition_penalty'] = confidence
        logger.info(f"Using API parameters: temp={linear}, top_p={top_p}, top_k={min_k}, speed={speaking_rate}, rep_penalty={confidence}")
    else:
        temp_override = get_config_override('temperature', linear)
        if temp_override is not None:
            inference_kwargs['temperature'] = temp_override

        top_p_override = get_config_override('top_p', top_p)
        if top_p_override is not None:
            inference_kwargs['top_p'] = top_p_override

        top_k_override = get_config_override('top_k', min_k)
        if top_k_override is not None:
            inference_kwargs['top_k'] = top_k_override

        speed_override = get_config_override('speed', speaking_rate)
        if speed_override is not None:
            inference_kwargs['speed'] = speed_override

        repetition_penalty_override = get_config_override('repetition_penalty', confidence)
        if repetition_penalty_override is not None:
            inference_kwargs['repetition_penalty'] = repetition_penalty_override
    
    # Always pass the stream parameter
    logger.debug(f"Inference kwargs: {inference_kwargs}")
    # Use shared audio generation function with only necessary kwargs
    wav_out_path = generate_audio_file(
        model=CURRENT_MODEL,
        language=language,
        speaker_wav=speaker_audio,
        text=text,
        stream=STREAM,
        **inference_kwargs
    )

    if IGNORE_PING == "pending":
        IGNORE_PING = True
        Path(wav_out_path).unlink(missing_ok=True)
        wav_out_path = SILENCE_AUDIO_PATH
    
    return wav_out_path, job_id


def generate_gradio_audio(model_choice, text, language, speaker_audio, prefix_audio, 
                speed, top_p,top_k, temperature, repetition_penalty, job_id) -> tuple[Path, int]:
    global _FROM_GRADIO
    _FROM_GRADIO = True
    wav_out_path, job_id = generate_audio(model_choice=model_choice, text=text, language=language, speaker_audio=speaker_audio, prefix_audio=prefix_audio,
                           speaking_rate=speed, top_p=top_p, min_k=top_k, linear=temperature, confidence=repetition_penalty, seed=job_id)
    _FROM_GRADIO = False
    return wav_out_path, job_id

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
                prefix_audio = gr.Audio( label="Optional Reference Audio (for style)", type="filepath",sources=["upload", "microphone"])
                speaker_audio = gr.Audio(label="Optional Speaker Audio (for cloning)", type="filepath",sources=["upload", "microphone"])

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Conditioning Parameters")
                temperature = gr.Slider(0, 2, value=1, step=0.01, label="Temperature")
                repetition_penalty = gr.Slider(1, 3, value=2.1, step=0.1, label="Repetition Penalty")
                top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K", precision=0)
                top_p = gr.Slider(0, 1, value=1, step=0.01, label="Top-P")
                speed = gr.Slider(0.5, 2, value=1.0, step=0.01, label="Speed")

        with gr.Column():
            generate_button = gr.Button("Generate Audio", variant="primary")
            output_audio = gr.Audio(label="Generated Audio", type="filepath", autoplay=True)

        model_choice = gr.Textbox(visible=False)
        #prefix_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None, visible=False)
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
        dnsmos_ovrl = gr.Number(visible=False)
        speaker_noised = gr.Checkbox(visible=False)
        cfg_scale = gr.Number(visible=False)
        min_k = gr.Number(visible=False)
        min_p = gr.Number(visible=False)
        linear = gr.Number(visible=False)
        confidence = gr.Number(visible=False)
        quadratic = gr.Number(visible=False)
        randomize_seed = gr.Checkbox(visible=False)
        unconditional_keys = gr.Textbox(visible=False)
        job_id = gr.Number(visible=False, value=uuid.uuid4())
        speed_input = gr.Number(visible=False)
        top_p_input = gr.Number(visible=False)
        top_k_input = gr.Number(visible=False)
        temperature_input = gr.Number(visible=False)
        repetition_penalty_input = gr.Number(visible=False)

        # Web UI button - uses visible sliders with generate_gradio_audio
        generate_button.click(fn=generate_gradio_audio,
            inputs=[model_choice, text, language, speaker_audio, prefix_audio, 
                speed, top_p, top_k, temperature, repetition_penalty, job_id],
                 outputs=[output_audio, job_id])
        
        # API-only button - uses hidden Number components with generate_audio
        # This is the endpoint that external API calls should use
        api_button = gr.Button(visible=False)
        api_button.click(fn=generate_audio,
            inputs=[model_choice, text, language, speaker_audio, prefix_audio, emotion1, emotion2, emotion3, emotion4, emotion5, emotion6, emotion7, emotion8,
                  vq_single, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale, top_p,
                  min_k, min_p, linear, confidence, quadratic, job_id, randomize_seed, unconditional_keys],
                  outputs=[output_audio, job_id])
        
        # Expose only the API function for external calls
        gr.api(fn=generate_audio, api_name="generate_audio")
    return demo


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize application environment
    initialize_application_environment("XTTS Text-to-Speech Application with Gradio Interface")
    
    # Load model with standardized initialization
    CURRENT_MODEL = initialize_model_with_cache(use_cpu=args.use_cpu, use_deepspeed=args.deepspeed, use_bfloat16=args.use_bfloat16)


    demo = build_interface()
    demo.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser, debug=True, show_api=True)