# Runtime hook to restrict Triton to NVIDIA backends only
# This prevents FileNotFoundError when Triton tries to discover AMD backends

import sys
import os

# Set environment variable before triton imports to restrict backend discovery
os.environ['TRITON_BACKENDS'] = 'nvidia'

"""
PyInstaller runtime hook to restrict Triton to NVIDIA backend only.

This prevents Triton from trying to discover and load AMD backends,
which aren't included in the frozen executable.

CRITICAL: This must run BEFORE any triton imports happen.
"""
import os
import sys

# Set environment variable to disable Triton backend auto-discovery
# This prevents the import-time backend loading that causes FileNotFoundError
os.environ['TRITON_INTERPRET'] = '1'  # Interpretation mode - no JIT compilation

# Set TRITON_CACHE_DIR to use PyInstaller temp directory
if hasattr(sys, '_MEIPASS'):
    os.environ['TRITON_CACHE_DIR'] = os.path.join(sys._MEIPASS, 'triton_cache')
def _patch_triton_backends():
    """Patch Triton to only discover NVIDIA backends in frozen app."""
    try:
        import triton.backends
        
        original_discover = triton.backends._discover_backends
        
        def _discover_nvidia_only():
            """Only discover NVIDIA backend, ignore AMD/others."""
            backends = original_discover()
            # Filter to only nvidia backend
            return {k: v for k, v in backends.items() if k == 'nvidia'}
        
        triton.backends._discover_backends = _discover_nvidia_only
    except (ImportError, AttributeError):
        # Triton not available or API changed, ignore
        pass

# Apply patch when this hook runs
_patch_triton_backends()
