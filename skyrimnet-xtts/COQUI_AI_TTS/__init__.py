import importlib.metadata

from COQUI_AI_TTS.utils.generic_utils import is_pytorch_at_least_2_4

# Use hardcoded version for extracted TTS library
__version__ = "0.27.1-extracted"

# Skip coqpit check for extracted library
# if "coqpit" in importlib.metadata.packages_distributions().get("coqpit", []):
#     msg = (
#         "coqui-tts switched to a forked version of Coqpit, but you still have the original "
#         "package installed. Run the following to avoid conflicts:\n"
#         "  pip uninstall coqpit\n"
#         "  pip install coqpit-config"
#     )
#     raise ImportError(msg)


if is_pytorch_at_least_2_4():
    import _codecs
    from collections import defaultdict

    import numpy as np
    import torch
    from packaging import version

    from COQUI_AI_TTS.config.shared_configs import BaseDatasetConfig
    from COQUI_AI_TTS.tts.configs.xtts_config import XttsConfig
    from COQUI_AI_TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
    from COQUI_AI_TTS.utils.radam import RAdam

    torch.serialization.add_safe_globals([dict, defaultdict, RAdam])

    # XTTS
    torch.serialization.add_safe_globals([BaseDatasetConfig, XttsConfig, XttsAudioConfig, XttsArgs])
