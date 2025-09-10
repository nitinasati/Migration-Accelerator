"""
LLM Provider wrapper for LLM-API-Service integration.

This module provides a unified interface for LLM operations using the LLM-API-Service.
"""

import structlog
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from llm.llm_api_client import LLMAPIClient, create_llm_client
from config.settings import LLMConfig

logger = structlog.get_logger(__name__)

# Constants
PROVIDER_NOT_INITIALIZED_MSG = "Provider not initialized. Call initialize() first."


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig, agent_name: str = "unknown"):
        """
        Initialize the LLM provider.
        
        Args:
            config: LLM configuration
            agent_name: Name of the agent using this provider
        """
        self.config = config
        self.agent_name = agent_name
        self.logger = structlog.get_logger(__name__).bind(
            provider=self.__class__.__name__,
            agent=agent_name
        )
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using the LLM."""
        pass
    
    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        output_schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate structured output using the LLM."""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion using the LLM."""
        pass


class LLMServiceProvider(BaseLLMProvider):
    """LLM provider that uses the LLM-API-Service."""
    
    def __init__(self, config: LLMConfig, agent_name: str = "unknown"):
        super().__init__(config, agent_name)
        self.client: Optional[LLMAPIClient] = None
    
    async def initialize(self) -> None:
        """Initialize the LLM service client."""
        try:
            from config.settings import settings
            
            self.client = create_llm_client(
                base_url=settings.llm_api_service_url,
                api_key=settings.llm_api_service_key,
                timeout=settings.llm_api_timeout
            )
            
            # Test the connection
            health = await self.client.health_check()
            self.logger.info(
                "LLM service provider initialized",
                service_url=settings.llm_api_service_url,
                health_status=health.get("status", "unknown")
            )
            
        except Exception as e:
            self.logger.error("Failed to initialize LLM service provider", error=str(e))
            raise
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using the LLM-API-Service."""
        if not self.client:
            raise RuntimeError(PROVIDER_NOT_INITIALIZED_MSG)
        
        try:
            result = await self.client.generate(
                prompt=prompt,
                provider=self.config.provider,
                model=self.config.model,
                system_message=system_message,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens
            )
            
            self.logger.debug(
                "Text generation completed",
                provider=self.config.provider,
                model=self.config.model,
                prompt_length=len(prompt)
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Text generation failed", error=str(e))
            raise
    
    async def generate_structured(
        self,
        prompt: str,
        output_schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate structured output using the LLM-API-Service."""
        if not self.client:
            raise RuntimeError(PROVIDER_NOT_INITIALIZED_MSG)
        
        try:
            result = await self.client.generate_structured(
                prompt=prompt,
                output_schema=output_schema,
                provider=self.config.provider,
                model=self.config.model,
                system_prompt=system_prompt,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens
            )
            
            self.logger.debug(
                "Structured generation completed",
                provider=self.config.provider,
                model=self.config.model,
                prompt_length=len(prompt)
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Structured generation failed", error=str(e))
            raise
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion using the LLM-API-Service."""
        if not self.client:
            raise RuntimeError(PROVIDER_NOT_INITIALIZED_MSG)
        
        try:
            result = await self.client.chat(
                messages=messages,
                provider=self.config.provider,
                model=self.config.model,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens
            )
            
            self.logger.debug(
                "Chat completion completed",
                provider=self.config.provider,
                model=self.config.model,
                message_count=len(messages)
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Chat completion failed", error=str(e))
            raise


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create(config: LLMConfig, agent_name: str = "unknown") -> BaseLLMProvider:
        """
        Create an LLM provider based on configuration.
        
        Args:
            config: LLM configuration
            agent_name: Name of the agent using this provider
            
        Returns:
            Configured LLM provider instance
        """
        # For now, we only support the LLM-API-Service provider
        return LLMServiceProvider(config, agent_name)


def initialize_langsmith() -> None:
    """
    Initialize LangSmith tracing for the Migration-Accelerator workflows.
    
    This function sets up LangSmith tracing to monitor LLM calls and agent workflows.
    It should be called early in the application lifecycle.
    """
    try:
        from config.settings import get_langsmith_config
        import os
        
        langsmith_config = get_langsmith_config()
        
        if langsmith_config.api_key:
            # Set LangSmith environment variables
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_API_KEY"] = langsmith_config.api_key
            os.environ["LANGCHAIN_PROJECT"] = langsmith_config.project
            
            logger.info(
                "✅ LangSmith tracing initialized for Migration-Accelerator workflows",
                api_key_set=bool(langsmith_config.api_key),
                project=langsmith_config.project
            )
        else:
            logger.warning(
                "⚠️ LangSmith API key not found - tracing disabled",
                hint="Set LANGSMITH_API_KEY or LANGCHAIN_API_KEY environment variable"
            )
            
    except Exception as e:
        logger.error("Failed to initialize LangSmith", error=str(e))
        # Don't raise - LangSmith is optional
