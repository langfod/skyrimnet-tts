import importlib
import logging
import re

from COQUI_AI_TTS.vc.configs.shared_configs import BaseVCConfig
from COQUI_AI_TTS.vc.models.base_vc import BaseVC

logger = logging.getLogger(__name__)


def setup_model(config: BaseVCConfig) -> BaseVC:
    logger.info("Using model: %s", config.model)
    # No voice conversion models are supported in this minimal XTTS build
    msg = f"Voice conversion model {config.model} is not supported in this build!"
    raise ValueError(msg)
