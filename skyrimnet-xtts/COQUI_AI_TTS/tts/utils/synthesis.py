def inv_spectrogram(postnet_output, ap, CONFIG):
    # Tacotron removed - using mel spectrogram inversion for all models
    wav = ap.inv_melspectrogram(postnet_output.T)
    return wav


# Note: apply_griffin_lim function removed - XTTS uses HifiGAN decoder, not Griffin-Lim vocoder
