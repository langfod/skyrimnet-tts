#!/usr/bin/env python3
"""
Shared Application Utilities Module for SkyrimNet TTS Applications
Contains common application initialization, setup, and configuration functions
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
from loguru import logger

# Local imports - Handle both direct and module execution
try:
    from .shared_config import setup_environment, SUPPORTED_LANGUAGE_CODES, DEFAULT_TTS_PARAMS
except ImportError:
    from shared_config import setup_environment, SUPPORTED_LANGUAGE_CODES, DEFAULT_TTS_PARAMS


# =============================================================================
# LOGGING CONFIGURATION UTILITIES
# =============================================================================

# Global flag to prevent multiple logging initializations
_LOGGING_CONFIGURED = False

def setup_application_logging(
    log_to_file: bool = None,
    log_file_path: str = None,
    console_level: str = "INFO",
    file_level: str = "INFO"
) -> None:
    """
    Setup standardized logging configuration for SkyrimNet applications.
    
    Args:
        log_to_file: Whether to enable file logging (checks env var if None)
        log_file_path: Path to log file (uses env var or default if None)
        console_level: Logging level for console output
        file_level: Logging level for file output
    """
    global _LOGGING_CONFIGURED
    
    # Only configure logging once to prevent conflicts
    if _LOGGING_CONFIGURED:
        return
        
    # Remove ALL existing loggers to avoid conflicts
    logger.remove()
    
    # Setup console logging with consistent format
    logger.add(
        sys.stdout, 
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", 
        level=console_level, 
        enqueue=True,
        catch=True
    )
    
    # Determine file logging settings
    if log_to_file is None:
        log_to_file = os.getenv('LOG_TO_FILE', 'false').lower() == 'true'
    
    if log_file_path is None:
        log_file_path = os.getenv('LOG_FILE_PATH', 'logs/skyrimnet.log')
    
    # Setup file logging if enabled
    if log_to_file:
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        logger.add(
            log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level=file_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True
        )
        logger.info(f"File logging enabled. Logs will be written to: {log_file_path}")
    
    # Set the flag to prevent reinitialization
    _LOGGING_CONFIGURED = True


# =============================================================================
# CONFIGURATION FILE UTILITIES
# =============================================================================

def load_config_file(
    config_file_path: str,
    default_config: Dict[str, Any],
    supported_modes: Dict[str, str] = None,
    global_flags: Dict[str, bool] = None
) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, bool]]:
    """
    Generic configuration file loader with error handling.
    
    Args:
        config_file_path: Path to configuration file
        default_config: Default configuration values
        supported_modes: Supported configuration modes for each parameter
        global_flags: Global boolean flags with defaults
        
    Returns:
        Tuple of (config_values, config_modes, global_flags)
    """
    if supported_modes is None:
        supported_modes = {key: 'default' for key in default_config.keys()}
    
    if global_flags is None:
        global_flags = {}
    
    config_modes = supported_modes.copy()
    final_config = default_config.copy()
    final_flags = global_flags.copy()
    
    try:
        config_path = Path(config_file_path)
        if not config_path.exists():
            logger.warning(f"Config file {config_file_path} not found, using defaults")
            return final_config, config_modes, final_flags
            
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
                if key in final_flags:
                    if value.lower() in ['true', 'yes', '1', 'on']:
                        final_flags[key] = True
                        logger.info(f"Setting {key} to True")
                    elif value.lower() in ['false', 'no', '0', 'off']:
                        final_flags[key] = False
                        logger.info(f"Setting {key} to False")
                    else:
                        logger.warning(f"Invalid boolean value '{value}' for {key}, using default")
                
                # Handle parameter modes and values
                elif key in config_modes:
                    if value.lower() == 'default':
                        config_modes[key] = 'default'
                    elif value.lower() == 'api':
                        config_modes[key] = 'api'
                    else:
                        try:
                            custom_value = float(value)
                            config_modes[key] = 'custom'
                            final_config[key] = custom_value
                            logger.info(f"Using custom {key} value: {custom_value}")
                        except ValueError:
                            logger.warning(f"Invalid value '{value}' for {key}, using default")
                            
        logger.info(f"Loaded config modes: {config_modes}")
        logger.info(f"Global flags: {final_flags}")
        return final_config, config_modes, final_flags
        
    except Exception as e:
        logger.error(f"Error reading config file {config_file_path}: {e}, using defaults")
        return final_config, config_modes, final_flags


def get_effective_config_value(
    param_name: str,
    api_value: Any,
    defaults: Dict[str, Any],
    modes: Dict[str, str],
    fallback_defaults: Dict[str, Any] = None,
    bypass_config: bool = False
) -> Any:
    """
    Get the effective configuration value based on mode and available values.
    
    Args:
        param_name: Name of the parameter
        api_value: Value from API/UI input
        defaults: Default configuration values
        modes: Configuration modes for each parameter
        fallback_defaults: Fallback defaults if not in config
        bypass_config: Whether to bypass config system (API mode)
        
    Returns:
        The effective value to use
    """
    if bypass_config:
        # API mode: use API value with fallback to shared defaults
        if fallback_defaults is None:
            fallback_defaults = DEFAULT_TTS_PARAMS
        return api_value if api_value is not None else fallback_defaults.get(param_name, 0.0)
    
    mode = modes.get(param_name, 'default')
    
    if mode == 'api':
        return api_value if api_value is not None else defaults[param_name]
    else:  # 'default' or 'custom'
        return defaults[param_name]


# =============================================================================
# APPLICATION STARTUP UTILITIES
# =============================================================================

def initialize_application_environment(app_name: str = "SkyrimNet TTS") -> None:
    """
    Initialize common application environment setup.
    
    Args:
        app_name: Name of the application for logging
    """
    logger.info(f"Starting {app_name}...")
    
    # Setup environment
    setup_environment()
    
    logger.info("Environment initialized")


def create_standard_error_response(error_message: str, details: str = None) -> Dict[str, Any]:
    """
    Create standardized error response format.
    
    Args:
        error_message: Main error message
        details: Additional error details
        
    Returns:
        Standardized error response dictionary
    """
    response = {"error": error_message}
    if details:
        response["detail"] = details
    return response