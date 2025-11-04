# =============================================================================
import torch
import time
from loguru import logger
from typing import Optional
from pathlib import Path
try:
    from .shared_cache_utils import get_latent_from_audio, save_torchaudio_wav
    from .shared_config import DEFAULT_TTS_PARAMS
    from .shared_models import check_text_length
    from .COQUI_AI_TTS.tts.models.xtts import Xtts
except ImportError:
    from shared_cache_utils import get_latent_from_audio, save_torchaudio_wav
    from shared_config import DEFAULT_TTS_PARAMS
    from shared_models import check_text_length
    from COQUI_AI_TTS.tts.models.xtts import Xtts

# =============================================================================

def generate_audio_file(
    model: Xtts,
    language: str,
    speaker_wav: str,
    text: str,
    stream: bool = False,
    **inference_kwargs
) -> Path:
    """
    Generate audio file using streaming inference with CUDA optimization to avoid CPU transfers.
    
    Args:
        model: The TTS model to use for inference
        language: Language code for synthesis
        speaker_wav: Speaker reference (file path or speaker name)
        text: Text to synthesize
        **inference_kwargs: Additional parameters for model.inference_stream()
            Supported kwargs:
            - temperature: Controls randomness (default from DEFAULT_TTS_PARAMS)
            - top_p: Nucleus sampling parameter (default from DEFAULT_TTS_PARAMS)
            - top_k: Top-k sampling parameter (default from DEFAULT_TTS_PARAMS)
            - speed: Speech speed (default from DEFAULT_TTS_PARAMS)
            - repetition_penalty: Penalty for repetition (default from DEFAULT_TTS_PARAMS)
            - enable_text_splitting: Whether to enable text splitting (auto-detected if not provided)
            - stream_chunk_size: Chunk size for streaming (default: 20)
            - overlap_wav_len: Overlap length for streaming (default: 1024)
    
    Returns:
        tuple: (Path to the generated audio file, audio length in seconds)
    """
    
    # Start timing the entire function
    func_start_time = time.perf_counter()

    output_sample_rate = model.args.output_sample_rate

    logger.info(f"Generating audio for text='{text[:50]}...', speaker='{Path(speaker_wav).stem}', language='{language}', stream={stream}")

    # Get latents from speaker
    gpt_cond_latent, speaker_embedding = get_latent_from_audio(
        model, language, speaker_wav
    )
    
    # Check text length and enable splitting if needed
    enable_text_splitting, char_limit = check_text_length(text, model, language)
    if enable_text_splitting:
        logger.info(f"Text length {len(text)} exceeds limit {char_limit}, enabling text splitting")
    
    # Prepare inference parameters with defaults
    inference_params = {
        'text': text,
        'language': language,
        'gpt_cond_latent': gpt_cond_latent,
        'speaker_embedding': speaker_embedding,
        'temperature': DEFAULT_TTS_PARAMS["TEMPERATURE"],
        'top_p': DEFAULT_TTS_PARAMS["TOP_P"],
        'top_k': DEFAULT_TTS_PARAMS["TOP_K"],
        'speed': DEFAULT_TTS_PARAMS["SPEED"],
        'repetition_penalty': DEFAULT_TTS_PARAMS["REPETITION_PENALTY"],
        'enable_text_splitting': enable_text_splitting,
    }
    for key, value in inference_kwargs.items():
        if key in inference_params and value is not None:
            inference_params[key] = value
        elif key not in inference_params:
            logger.warning(f"Ignoring unknown inference parameter: {key}={value}")
       
    # Override enable_text_splitting if explicitly provided
    if 'enable_text_splitting' in inference_kwargs:
        inference_params['enable_text_splitting'] = inference_kwargs['enable_text_splitting']    

    if stream:
        wav_chunks = []
        for chunk in model.inference_stream(**inference_params):
            wav_chunks.append(chunk)
        
        wav_out = torch.cat(wav_chunks, dim=0)
    else:
        wav_result = model.inference(**inference_params, return_as_tensor=True)
        wav_out = wav_result["wav"]
        wav_chunks = None
    
    wav_out_path = save_torchaudio_wav(
        wav_tensor=wav_out.unsqueeze(0),
        sr=output_sample_rate,
        audio_path=speaker_wav,
    )
    wav_length_s = wav_out.shape[0] / output_sample_rate

    func_end_time = time.perf_counter()
    total_duration_s = func_end_time - func_start_time
    if speaker_wav:
        input_wav = speaker_wav.split('\\')[-1]
        logger.info(f"Total 'generate_audio' output of {wav_out_path} for {input_wav} length: {wav_length_s:.2f}s execution time: {total_duration_s:.2f}s Speed: {wav_length_s/total_duration_s:.2f}x")
    else:
        logger.info(f"Total 'generate_audio' execution time: {total_duration_s:.2f} seconds")

    del wav_out
    if wav_chunks is not None:
        del wav_chunks
    return wav_out_path
