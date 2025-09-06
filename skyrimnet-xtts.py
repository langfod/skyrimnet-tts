#!/usr/bin/env python3
"""
Zonos Text-to-Speech Application with Gradio Interface
Enhanced with disk and memory caching for speaker embeddings
"""

# Standard library imports
import os
from pathlib import Path
import sys
import time
import argparse
from loguru import logger

# Third-party imports
import gradio as gr
import os
import time

from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
from utils import get_latent_from_audio, init_latent_cache, save_torchaudio_wav

# =============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# =============================================================================
# Fix torch.compile C++ compilation issues on Windows
if sys.platform == "win32":
    os.environ["TORCH_COMPILE_CPP_FORCE_X64"] = "1"
    # Alternative approach - force specific compiler architecture
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"

os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TTS_HOME"] = "models"

# Global model state
CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None
SPEAKER_EMBEDDING = None
SPEAKER_AUDIO_PATH = None
SPEAKER_AUDIO_PATH_DICT = {}
SUPPORTED_LANGUAGE_CODES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"]

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--use_cpu", action='store_true')
parser.add_argument("--deepspeed", action='store_true')

args = parser.parse_args()


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
    language = language.split("-")[0] if language else "en"
    logger.info(f"inputs: text={text}, language={language}, speaker_audio={Path(speaker_audio).stem if speaker_audio else 'None'}, seed={seed}")
    # Start timing the entire function
    func_start_time = time.perf_counter()

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
    global SPEAKER_AUDIO_PATH, SPEAKER_AUDIO_PATH_DICT, SPEAKER_EMBEDDING

    speaker_audio_uuid = seed
    #if randomize_seed:
    #    seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
    #torch.manual_seed(seed)

    gpt_cond_latent, speaker_embedding = get_latent_from_audio(CURRENT_MODEL, language, speaker_audio, speaker_audio_uuid)

    wav_out = CURRENT_MODEL.inference(
    text=text, language=language,
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    speed=1.0,  # speaking_rate if speaking_rate else 1.0,
    top_p=1.0,  # top_p if top_p else 1.0,
    top_k=50,   # top_k if top_k else 50,
    temperature=0.7,
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
    with gr.Blocks(analytics_enabled=False, title="XTTS") as demo:
        gr.Markdown("# XTTS with Speaker Embedding Cache")

        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text to Synthesize",
                    value="Zonos uses eSpeak for text to phoneme conversion!",
                    lines=4, max_length=500)
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

def load_model(model_name="xtts_v2", use_deepspeed=False, use_cpu=False):
    output_model_path, output_config_path, model_item = ModelManager().download_model(model_name)
    config = XttsConfig()
    config.load_json(output_config_path)
    model = Xtts.init_from_config(config)
    if use_cpu:
        model.load_checkpoint(config, checkpoint_dir=output_model_path, use_deepspeed=False)
        model.cpu()
    else:
        model.load_checkpoint(config, checkpoint_dir=output_model_path, use_deepspeed=use_deepspeed)
        model.cuda()
    return model

if __name__ == "__main__":
    CURRENT_MODEL = load_model(use_cpu=args.use_cpu, use_deepspeed=args.deepspeed)

    init_latent_cache(model=CURRENT_MODEL, supported_languages=SUPPORTED_LANGUAGE_CODES)
    wav, _ = generate_audio(text="This is a test.", speaker_audio="assets/malebrute.wav", language="en")

    demo = build_interface()
    demo.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser)