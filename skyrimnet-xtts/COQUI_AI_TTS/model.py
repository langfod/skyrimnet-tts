import os
from abc import abstractmethod
from typing import Any

import torch
from coqpit import Coqpit

# pylint: skip-file


# Minimal load_fsspec replacement for inference-only deployment  
def load_fsspec(checkpoint_path: str | os.PathLike[Any], map_location: str = "cpu", cache: bool = False) -> dict[str, Any]:
    """Load checkpoint file using torch.load (minimal fsspec replacement)."""
    import sys
    
    # Create module alias for compatibility with existing model checkpoints
    # Model files were saved with 'TTS' namespace, but we renamed to 'COQUI_AI_TTS'
    if 'TTS' not in sys.modules:
        import COQUI_AI_TTS as TTS_alias
        sys.modules['TTS'] = TTS_alias
        sys.modules['TTS.tts'] = TTS_alias.tts
        sys.modules['TTS.tts.configs'] = TTS_alias.tts.configs
        sys.modules['TTS.tts.configs.xtts_config'] = TTS_alias.tts.configs.xtts_config
    
    try:
        # Try with weights_only=True first (PyTorch 2.6+ default)
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception:
        # Fall back to weights_only=False for compatibility with existing model files
        # This is safe for trusted model checkpoints
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


class BaseTrainerModel(torch.nn.Module):
    """Minimal base model for inference-only TTS deployment.
    
    Replaces the full training model with minimal implementation
    that provides the same interface without training dependencies.
    """

    @staticmethod
    @abstractmethod
    def init_from_config(config: Coqpit) -> "BaseTrainerModel":
        """Init the model and all its attributes from the given config.

        Override this depending on your model.
        """
        ...

    @abstractmethod
    def inference(self, input: torch.Tensor, aux_input: dict[str, Any] = {}) -> dict[str, Any]:
        """Forward pass for inference.

        It must return a dictionary with the main model output and all the auxiliary outputs. The key ```model_outputs```
        is considered to be the main output and you can add any other auxiliary outputs as you want.

        We don't use `*kwargs` since it is problematic with the TorchScript API.

        Args:
            input (torch.Tensor): [description]
            aux_input (Dict): Auxiliary inputs like speaker embeddings, durations etc.

        Returns:
            Dict: [description]
        """
        outputs_dict = {"model_outputs": None}
        ...
        return outputs_dict

    def load_checkpoint(
        self,
        config: Coqpit,
        checkpoint_path: str | os.PathLike[Any],
        *,
        eval: bool = False,
        strict: bool = True,
        cache: bool = False,
    ) -> None:
        """Load a model checkpoint file and get ready for training or inference.

        Args:
            config (Coqpit): Model configuration.
            checkpoint_path (str | os.PathLike): Path to the model checkpoint file.
            eval (bool, optional): If true, init model for inference else for training. Defaults to False.
            strict (bool, optional): Match all checkpoint keys to model's keys. Defaults to True.
            cache (bool, optional): If True, cache the file locally for subsequent calls.
                It is cached under `trainer.io.get_user_data_dir()/tts_cache`. Defaults to False.
        """
        state = load_fsspec(checkpoint_path, map_location="cpu", cache=cache)
        self.load_state_dict(state["model"], strict=strict)
        if eval:
            self.eval()

    @property
    def device(self) -> torch.device:
        # Cache the device to avoid repeated parameter iteration
        if hasattr(self, '_cached_device'):
            return self._cached_device
        
        device = next(self.parameters()).device
        self._cached_device = device
        return device

    def _apply(self, fn):
        """Override to invalidate device cache when model is moved between devices."""
        # Clear device cache when parameters are transformed (e.g., .to(), .cuda(), .cpu())
        if hasattr(self, '_cached_device'):
            delattr(self, '_cached_device')
        return super()._apply(fn)


# Minimal function replacements for inference-only deployment
def set_partial_state_dict(*args, **kwargs):
    """Minimal stub - use standard load_state_dict for inference"""
    raise NotImplementedError("set_partial_state_dict not available in inference-only mode. Use load_state_dict instead.")


# Minimal class stubs for training-only type annotations
class BaseDashboardLogger:
    """Minimal stub for training logger type annotations"""
    pass


class DistributedSampler:
    """Minimal stub for training sampler type annotations"""
    pass


class DistributedSamplerWrapper:
    """Minimal stub for training sampler type annotations"""  
    pass
