"""
Logging configuration for the Migration-Accelerators platform.
"""

import logging
import sys
from typing import Optional
import structlog
from structlog.stdlib import LoggerFactory

from config.settings import settings, get_langsmith_config


def configure_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    enable_langsmith: bool = True
) -> None:
    """
    Configure structured logging for the platform.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json, console)
        enable_langsmith: Enable LangSmith integration
    """
    
    # Use settings if not provided
    log_level = log_level or settings.log_level
    log_format = log_format or settings.log_format
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        stream=sys.stdout
    )
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add format-specific processor
    if log_format.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure LangSmith if enabled
    if enable_langsmith:
        _configure_langsmith()
    
    # Log configuration
    logger = structlog.get_logger()
    logger.info(
        "Logging configured",
        log_level=log_level,
        log_format=log_format,
        langsmith_enabled=enable_langsmith
    )


def _configure_langsmith() -> None:
    """Configure LangSmith integration."""
    try:
        langsmith_config = get_langsmith_config()
        
        if not langsmith_config.api_key:
            logger = structlog.get_logger()
            logger.warning("LangSmith API key not configured, skipping LangSmith setup")
            return
        
        # Set environment variables for LangSmith
        import os
        os.environ["LANGCHAIN_API_KEY"] = langsmith_config.api_key
        os.environ["LANGCHAIN_PROJECT"] = langsmith_config.project
        os.environ["LANGCHAIN_TRACING_V2"] = "true" if langsmith_config.tracing_v2 else "false"
        
        if langsmith_config.endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = langsmith_config.endpoint
        
        # Import and configure LangSmith
        try:
            import langsmith
            from langsmith import Client
            
            # Initialize LangSmith client
            client = Client(
                api_key=langsmith_config.api_key,
                api_url=langsmith_config.endpoint
            )
            
            logger = structlog.get_logger()
            logger.info(
                "LangSmith configured successfully",
                project=langsmith_config.project,
                tracing_v2=langsmith_config.tracing_v2
            )
            
        except ImportError:
            logger = structlog.get_logger()
            logger.warning("LangSmith not installed, skipping LangSmith configuration")
            
    except Exception as e:
        logger = structlog.get_logger()
        logger.error("Failed to configure LangSmith", error=str(e))


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        structlog.BoundLogger: Structured logger
    """
    return structlog.get_logger(name)


class LoggingContext:
    """Context manager for logging configuration."""
    
    def __init__(
        self,
        log_level: Optional[str] = None,
        log_format: Optional[str] = None,
        enable_langsmith: bool = True
    ):
        self.log_level = log_level
        self.log_format = log_format
        self.enable_langsmith = enable_langsmith
        self.original_config = None
    
    def __enter__(self):
        # Store original configuration
        self.original_config = {
            "log_level": settings.log_level,
            "log_format": settings.log_format
        }
        
        # Apply new configuration
        configure_logging(
            log_level=self.log_level,
            log_format=self.log_format,
            enable_langsmith=self.enable_langsmith
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original configuration
        if self.original_config:
            configure_logging(
                log_level=self.original_config["log_level"],
                log_format=self.original_config["log_format"],
                enable_langsmith=self.enable_langsmith
            )


# Initialize logging on module import
configure_logging()
