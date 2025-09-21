#!/usr/bin/env python3
"""
Shared Command Line Arguments Module for SkyrimNet TTS Applications
Contains common argument parsing with extensibility for app-specific arguments
"""

import argparse
from typing import Optional, List, Dict, Any


# =============================================================================
# COMMON ARGUMENT DEFINITIONS
# =============================================================================

COMMON_ARGS = {
    "--server": {
        "type": str,
        "help": "Server host address",
        "metavar": "HOST"
    },
    "--port": {
        "type": int,
        "help": "Server port number",
        "metavar": "PORT"
    },
    "--use_cpu": {
        "action": "store_true",
        "help": "Use CPU instead of CUDA for model inference"
    }
}


# Default values per application type
DEFAULT_VALUES = {
    "api": {
        "server": "0.0.0.0",
        "port": 7860
    },
    "gradio": {
        "server": "0.0.0.0",
        "port": 7860
    }
}

# Application-specific arguments
APP_SPECIFIC_ARGS = {
    "gradio": {
        "--share": {
            "action": "store_true",
            "help": "Create a public Gradio share link"
        },
        "--inbrowser": {
            "action": "store_true",
            "help": "Automatically open browser after launching"
        }
    },
    "api": {
        # API-specific args can be added here if needed
    }
}


# =============================================================================
# ARGUMENT PARSER CREATION
# =============================================================================

def create_base_parser(app_type: str, description: str = None) -> argparse.ArgumentParser:
    """
    Create base argument parser with common arguments
    
    Args:
        app_type: Type of application ("api" or "gradio")
        description: Parser description
        
    Returns:
        ArgumentParser: Configured parser with common arguments
    """
    if description is None:
        description = f"SkyrimNet TTS {app_type.title()} Application"
    
    parser = argparse.ArgumentParser(description=description)
    
    # Add common arguments with app-specific defaults
    defaults = DEFAULT_VALUES.get(app_type, {})
    
    for arg_name, arg_config in COMMON_ARGS.items():
        config = arg_config.copy()
        
        # Set default value if available
        if arg_name.lstrip('-') in defaults:
            config["default"] = defaults[arg_name.lstrip('-')]
        
        parser.add_argument(arg_name, **config)
    
    return parser


def add_app_specific_args(parser: argparse.ArgumentParser, app_type: str) -> argparse.ArgumentParser:
    """
    Add application-specific arguments to parser
    
    Args:
        parser: Base parser to extend
        app_type: Type of application ("api" or "gradio")
        
    Returns:
        ArgumentParser: Extended parser with app-specific arguments
    """
    app_args = APP_SPECIFIC_ARGS.get(app_type, {})
    
    for arg_name, arg_config in app_args.items():
        parser.add_argument(arg_name, **arg_config)
    
    return parser


def create_parser(app_type: str, description: str = None, 
                 extra_args: Optional[Dict[str, Dict[str, Any]]] = None) -> argparse.ArgumentParser:
    """
    Create complete argument parser for specified application type
    
    Args:
        app_type: Type of application ("api" or "gradio")
        description: Parser description
        extra_args: Additional custom arguments in format {arg_name: arg_config}
        
    Returns:
        ArgumentParser: Complete configured parser
        
    Example:
        parser = create_parser("api", extra_args={
            "--log-level": {"choices": ["DEBUG", "INFO"], "default": "INFO"}
        })
    """
    # Create base parser
    parser = create_base_parser(app_type, description)
    
    # Add app-specific arguments
    parser = add_app_specific_args(parser, app_type)
    
    # Add any extra custom arguments
    if extra_args:
        for arg_name, arg_config in extra_args.items():
            parser.add_argument(arg_name, **arg_config)
    
    return parser


# =============================================================================
# ARGUMENT VALIDATION
# =============================================================================

def validate_args(args: argparse.Namespace, app_type: str) -> argparse.Namespace:
    """
    Validate parsed arguments and apply any necessary corrections
    
    Args:
        args: Parsed arguments namespace
        app_type: Type of application for validation context
        
    Returns:
        Namespace: Validated arguments
        
    Raises:
        ValueError: If validation fails
    """
    # Validate port range
    if hasattr(args, 'port') and args.port is not None:
        if not (1 <= args.port <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got: {args.port}")
    
    # Validate server address
    if hasattr(args, 'server') and args.server:
        # Basic validation - could be enhanced
        if not args.server.replace('.', '').replace(':', '').replace('-', '').replace('_', '').isalnum():
            if args.server not in ['localhost', '0.0.0.0']:
                raise ValueError(f"Invalid server address: {args.server}")
    
    # App-specific validation
    if app_type == "api":
        # Ensure API has a port specified
        if not hasattr(args, 'port') or args.port is None:
            raise ValueError("API applications must specify a port")
    return args


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_api_args(description: str = None, extra_args: Optional[Dict[str, Dict[str, Any]]] = None):
    """
    Parse command line arguments for API application
    
    Args:
        description: Parser description
        extra_args: Additional custom arguments
        
    Returns:
        Namespace: Parsed and validated arguments
    """
    parser = create_parser("api", description, extra_args)
    args = parser.parse_args()
    return validate_args(args, "api")


def parse_gradio_args(description: str = None, extra_args: Optional[Dict[str, Dict[str, Any]]] = None):
    """
    Parse command line arguments for Gradio application
    
    Args:
        description: Parser description  
        extra_args: Additional custom arguments
        
    Returns:
        Namespace: Parsed and validated arguments
    """
    parser = create_parser("gradio", description, extra_args)
    args = parser.parse_args()
    return validate_args(args, "gradio")


def get_common_arg_names() -> List[str]:
    """
    Get list of common argument names (without dashes)
    
    Returns:
        List[str]: Argument names
    """
    return [name.lstrip('-') for name in COMMON_ARGS.keys()]


def get_app_specific_arg_names(app_type: str) -> List[str]:
    """
    Get list of app-specific argument names (without dashes)
    
    Args:
        app_type: Application type
        
    Returns:
        List[str]: App-specific argument names
    """
    app_args = APP_SPECIFIC_ARGS.get(app_type, {})
    return [name.lstrip('-') for name in app_args.keys()]