"""
LLM provider abstraction for the Migration-Accelerators platform.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import structlog

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler

from config.settings import LLMConfig, LLMProvider


class LLMCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for LLM operations."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = structlog.get_logger().bind(agent=agent_name)
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts."""
        self.logger.info("LLM operation started", prompts_count=len(prompts))
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when LLM ends."""
        self.logger.info("LLM operation completed", response_length=len(str(response)))
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Called when LLM encounters an error."""
        self.logger.error("LLM operation failed", error=str(error))


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, config: LLMConfig, agent_name: str = "llm_provider"):
        self.config = config
        self.agent_name = agent_name
        self.logger = structlog.get_logger().bind(agent=agent_name)
        self.callback_handler = LLMCallbackHandler(agent_name)
        self._client = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM provider."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            system_message: Optional system message
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            str: Generated text
        """
        pass
    
    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured output using the LLM.
        
        Args:
            prompt: Input prompt
            schema: Output schema
            system_message: Optional system message
            
        Returns:
            Dict[str, Any]: Structured output
        """
        pass
    
    async def health_check(self) -> bool:
        """
        Check if the LLM provider is healthy.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            response = await self.generate("Hello", max_tokens=10)
            return len(response) > 0
        except Exception as e:
            self.logger.error("LLM health check failed", error=str(e))
            return False


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    async def initialize(self) -> None:
        """Initialize OpenAI provider."""
        try:
            self._client = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                openai_api_key=self.config.api_key,
                callbacks=[self.callback_handler]
            )
            self.logger.info("OpenAI provider initialized", model=self.config.model)
        except Exception as e:
            self.logger.error("Failed to initialize OpenAI provider", error=str(e))
            raise
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using OpenAI."""
        try:
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            response = await self._client.ainvoke(messages)
            return response.content
        except Exception as e:
            self.logger.error("OpenAI generation failed", error=str(e))
            raise
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured output using OpenAI."""
        try:
            structured_prompt = f"""
            {system_message or ""}
            
            {prompt}
            
            Please respond with a JSON object that matches this schema:
            {schema}
            
            Return only the JSON object, no additional text.
            """
            
            response = await self.generate(structured_prompt)
            
            # Parse JSON response
            import json
            return json.loads(response)
        except Exception as e:
            self.logger.error("OpenAI structured generation failed", error=str(e))
            raise


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider."""
    
    async def initialize(self) -> None:
        """Initialize Anthropic provider."""
        try:
            self._client = ChatAnthropic(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                anthropic_api_key=self.config.api_key,
                callbacks=[self.callback_handler]
            )
            self.logger.info("Anthropic provider initialized", model=self.config.model)
        except Exception as e:
            self.logger.error("Failed to initialize Anthropic provider", error=str(e))
            raise
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using Anthropic."""
        try:
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            response = await self._client.ainvoke(messages)
            return response.content
        except Exception as e:
            self.logger.error("Anthropic generation failed", error=str(e))
            raise
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured output using Anthropic."""
        try:
            structured_prompt = f"""
            {system_message or ""}
            
            {prompt}
            
            Please respond with a JSON object that matches this schema:
            {schema}
            
            Return only the JSON object, no additional text.
            """
            
            response = await self.generate(structured_prompt)
            
            # Parse JSON response
            import json
            return json.loads(response)
        except Exception as e:
            self.logger.error("Anthropic structured generation failed", error=str(e))
            raise


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock LLM provider."""
    
    async def initialize(self) -> None:
        """Initialize Bedrock provider."""
        try:
            self._client = ChatBedrock(
                model_id=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                region_name=self.config.region,
                callbacks=[self.callback_handler]
            )
            self.logger.info("Bedrock provider initialized", model=self.config.model)
        except Exception as e:
            self.logger.error("Failed to initialize Bedrock provider", error=str(e))
            raise
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using Bedrock."""
        try:
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            response = await self._client.ainvoke(messages)
            return response.content
        except Exception as e:
            self.logger.error("Bedrock generation failed", error=str(e))
            raise
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured output using Bedrock."""
        try:
            structured_prompt = f"""
            {system_message or ""}
            
            {prompt}
            
            Please respond with a JSON object that matches this schema:
            {schema}
            
            Return only the JSON object, no additional text.
            """
            
            response = await self.generate(structured_prompt)
            
            # Parse JSON response
            import json
            return json.loads(response)
        except Exception as e:
            self.logger.error("Bedrock structured generation failed", error=str(e))
            raise


class VertexAIProvider(BaseLLMProvider):
    """Google Vertex AI LLM provider."""
    
    async def initialize(self) -> None:
        """Initialize Vertex AI provider."""
        try:
            self._client = ChatVertexAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                callbacks=[self.callback_handler]
            )
            self.logger.info("Vertex AI provider initialized", model=self.config.model)
        except Exception as e:
            self.logger.error("Failed to initialize Vertex AI provider", error=str(e))
            raise
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using Vertex AI."""
        try:
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            response = await self._client.ainvoke(messages)
            return response.content
        except Exception as e:
            self.logger.error("Vertex AI generation failed", error=str(e))
            raise
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured output using Vertex AI."""
        try:
            structured_prompt = f"""
            {system_message or ""}
            
            {prompt}
            
            Please respond with a JSON object that matches this schema:
            {schema}
            
            Return only the JSON object, no additional text.
            """
            
            response = await self.generate(structured_prompt)
            
            # Parse JSON response
            import json
            return json.loads(response)
        except Exception as e:
            self.logger.error("Vertex AI structured generation failed", error=str(e))
            raise


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.BEDROCK: BedrockProvider,
        LLMProvider.VERTEXAI: VertexAIProvider,
    }
    
    @classmethod
    def create(cls, config: LLMConfig, agent_name: str = "llm_provider") -> BaseLLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            config: LLM configuration
            agent_name: Name of the agent using the provider
            
        Returns:
            BaseLLMProvider: LLM provider instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        if config.provider not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
        
        provider_class = cls._providers[config.provider]
        return provider_class(config, agent_name)
    
    @classmethod
    def get_supported_providers(cls) -> List[LLMProvider]:
        """Get list of supported LLM providers."""
        return list(cls._providers.keys())
