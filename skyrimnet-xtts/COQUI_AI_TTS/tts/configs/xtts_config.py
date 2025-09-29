from dataclasses import dataclass, field
import json

from COQUI_AI_TTS.tts.configs.shared_configs import BaseTTSConfig
from COQUI_AI_TTS.tts.models.xtts import XttsArgs, XttsAudioConfig


@dataclass
class XttsConfig(BaseTTSConfig):
    """Defines parameters for XTTS TTS model.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (XttsArgs):
            Model architecture arguments. Defaults to `XttsArgs()`.

        audio (XttsAudioConfig):
            Audio processing configuration. Defaults to `XttsAudioConfig()`.

        model_dir (str):
            Path to the folder that has all the XTTS models. Defaults to None.

        temperature (float):
            Temperature for the autoregressive model inference. Larger values makes predictions more creative sacrificing stability. Defaults to `0.2`.

        length_penalty (float):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length,
            which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative),
            length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.

        repetition_penalty (float):
            The parameter for repetition penalty. 1.0 means no penalty. Defaults to `2.0`.

        top_p (float):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            Defaults to `0.8`.

        num_gpt_outputs (int):
            Number of samples taken from the autoregressive model, all of which are filtered using CLVP.
            As XTTS is a probabilistic model, more samples means a higher probability of creating something "great".
            Defaults to `16`.

        gpt_cond_len (int):
            Length of the audio used as conditioning for the autoregressive  model. If audio is shorter,
            then audio length is used else the first `gpt_cond_len` secs is used. Defaults to 12 seconds.

        gpt_cond_chunk_len (int):
            Audio chunk size in seconds. Audio is split into chunks and latents are extracted for each chunk. Then
            the latents are averaged. Chunking improves the stability. It must be <= gpt_cond_len.
            If gpt_cond_len == gpt_cond_chunk_len, no chunking. Defaults to `4` seconds.

        max_ref_len (int):
            Maximum number of seconds of audio to be used as conditioning for the decoder. Defaults to `10`.

        sound_norm_refs (bool):
            Whether to normalize the conditioning audio. Defaults to `False`.

    Note:
        Check :class:`TTS.tts.configs.shared_configs.BaseTTSConfig` for the inherited parameters.

    Example:

        >>> from COQUI_AI_TTS.tts.configs.xtts_config import XttsConfig
        >>> config = XttsConfig()
    """

    model: str = "xtts"
    _supports_cloning: bool = True
    # model specific params
    model_args: XttsArgs = field(default_factory=XttsArgs)
    audio: XttsAudioConfig = field(default_factory=XttsAudioConfig)
    model_dir: str = None
    languages: list[str] = field(
        default_factory=lambda: [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "pl",
            "tr",
            "ru",
            "nl",
            "cs",
            "ar",
            "zh-cn",
            "hu",
            "ko",
            "ja",
            "hi",
        ]
    )

    # inference params
    temperature: float = 0.85
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    top_k: int = 50
    top_p: float = 0.85
    num_gpt_outputs: int = 1

    # cloning
    gpt_cond_len: int = 12
    gpt_cond_chunk_len: int = 4
    max_ref_len: int = 10
    sound_norm_refs: bool = False

    @classmethod
    def load_from_json(cls, json_path: str):
        """Load config from JSON file with proper nested object handling.
        
        This method works around coqpit's inability to properly deserialize
        nested objects like XttsArgs and XttsAudioConfig from JSON dicts.
        """
        with open(json_path, 'r') as f:
            config_data = json.load(f)
        
        # Create base config
        config = cls()
        
        # Set simple fields that exist in both JSON and config
        simple_fields = [
            'model', 'languages', 'temperature', 'length_penalty', 'repetition_penalty',
            'top_k', 'top_p', 'num_gpt_outputs', 'gpt_cond_len', 'gpt_cond_chunk_len',
            'max_ref_len', 'sound_norm_refs', 'model_dir'
        ]
        
        for field_name in simple_fields:
            if field_name in config_data and config_data[field_name] is not None:
                setattr(config, field_name, config_data[field_name])
        
        # Handle model_args (dict -> XttsArgs)
        if 'model_args' in config_data and config_data['model_args'] is not None:
            model_args_dict = config_data['model_args']
            model_args = XttsArgs()
            for key, value in model_args_dict.items():
                if hasattr(model_args, key) and value is not None:
                    setattr(model_args, key, value)
            config.model_args = model_args
        
        # Handle audio (dict -> XttsAudioConfig) 
        if 'audio' in config_data and config_data['audio'] is not None:
            audio_dict = config_data['audio']
            audio_config = XttsAudioConfig()
            for key, value in audio_dict.items():
                if hasattr(audio_config, key) and value is not None:
                    setattr(audio_config, key, value)
            config.audio = audio_config
        
        return config
