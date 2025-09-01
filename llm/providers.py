"""
LLM provider abstraction for the Migration-Accelerators platform.
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import structlog

from config.settings import LLMConfig, LLMProvider, get_langsmith_config


def initialize_langsmith():
    """Initialize LangSmith tracing."""
    try:
        # Load environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        # Get LangSmith configuration
        langsmith_config = get_langsmith_config()
        
        # Check for LangSmith API key in multiple possible locations
        api_key = (
            langsmith_config.api_key or 
            os.getenv("LANGSMITH_API_KEY") or 
            os.getenv("LANGCHAIN_API_KEY")
        )
        
        project = (
            langsmith_config.project or 
            os.getenv("LANGSMITH_PROJECT") or 
            os.getenv("LANGCHAIN_PROJECT", "migration-accelerators")
        )
        
        if api_key:
            # Set environment variables for LangSmith
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = api_key
            os.environ["LANGCHAIN_PROJECT"] = project
            
            if langsmith_config.endpoint:
                os.environ["LANGCHAIN_ENDPOINT"] = langsmith_config.endpoint
            
            # Import and initialize LangSmith
            try:
                from langsmith import Client
                client = Client(api_key=api_key)
                print(f"✅ LangSmith initialized - Project: {project}")
                return True
            except ImportError:
                print("⚠️ LangSmith package not installed. Install with: pip install langsmith")
                return False
        else:
            print("⚠️ LangSmith API key not configured. Set LANGSMITH_API_KEY or LANGCHAIN_API_KEY environment variable.")
            return False
            
    except Exception as e:
        print(f"⚠️ Failed to initialize LangSmith: {e}")
        return False


class LLMCallbackHandler:
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


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing and development."""
    
    async def initialize(self) -> None:
        """Initialize mock provider."""
        self.logger.info("Mock LLM provider initialized", model=self.config.model)
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate mock text response."""
        try:
            self.callback_handler.on_llm_start({}, [prompt])
            
            # Simple mock response based on prompt content
            if "file" in prompt.lower() and "read" in prompt.lower():
                response = """{
                    "success": true,
                    "data": [
                        {"policy_number": "POL123456", "employee_id": "EMP001234", "status": "active"},
                        {"policy_number": "POL789012", "employee_id": "EMP005678", "status": "pending"}
                    ],
                    "metadata": {"format": "csv", "records_count": 2}
                }"""
            elif "map" in prompt.lower() and "select" in prompt.lower():
                response = """{
                    "selected_mapping": "disability_mapping",
                    "record_type": "disability",
                    "confidence": 0.95
                }"""
            elif "transform" in prompt.lower():
                response = """{
                    "success": true,
                    "transformed_data": [
                        {"policyId": "POL123456", "employeeId": "EMP001234", "policyStatus": "ACTIVE"},
                        {"policyId": "POL789012", "employeeId": "EMP005678", "policyStatus": "PENDING"}
                    ],
                    "metadata": {"transformation_type": "llm", "records_processed": 2}
                }"""
            elif "validate" in prompt.lower():
                response = "Data validation completed successfully. All required fields are present."
            elif "api" in prompt.lower():
                response = "API integration completed successfully. Data has been sent to target system."
            else:
                response = f"Mock response for: {prompt[:50]}..."
            
            self.callback_handler.on_llm_end(response)
            return response
            
        except Exception as e:
            self.callback_handler.on_llm_error(e)
            self.logger.error("Mock LLM generation failed", error=str(e))
            raise
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate mock structured output."""
        try:
            # Generate a simple mock response based on the schema
            mock_response = {}
            
            if "validation_results" in schema.get("properties", {}):
                mock_response = {
                    "validation_results": [
                        {
                            "field": "sample_field",
                            "status": "valid",
                            "message": "Field validation passed"
                        }
                    ],
                    "total_errors": 0,
                    "total_warnings": 0
                }
            elif "mapping_results" in schema.get("properties", {}):
                mock_response = {
                    "mapping_results": [
                        {
                            "source_field": "old_field",
                            "target_field": "new_field",
                            "transformation": "direct",
                            "status": "success"
                        }
                    ],
                    "total_mapped": 1
                }
            elif "transformation_results" in schema.get("properties", {}):
                mock_response = {
                    "transformation_results": {
                        "format": "json",
                        "records_processed": 1,
                        "status": "success"
                    }
                }
            elif "api_results" in schema.get("properties", {}):
                mock_response = {
                    "api_results": [
                        {
                            "record_id": "123",
                            "status": "success",
                            "response_code": 200
                        }
                    ],
                    "total_success": 1,
                    "total_failed": 0
                }
            else:
                # Generic mock response
                mock_response = {
                    "status": "success",
                    "message": "Mock structured response generated",
                    "data": {"sample": "value"}
                }
            
            return mock_response
            
        except Exception as e:
            self.logger.error("Mock structured generation failed", error=str(e))
            raise


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider (requires langchain-openai package)."""
    
    async def initialize(self) -> None:
        """Initialize OpenAI provider."""
        try:
            # Try to import OpenAI provider
            from langchain_openai import ChatOpenAI
            from langchain.schema import BaseMessage, HumanMessage, SystemMessage
            
            self._client = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                openai_api_key=self.config.api_key
            )
            self.logger.info("OpenAI provider initialized", model=self.config.model)
        except ImportError:
            self.logger.warning("langchain-openai not available, falling back to mock provider")
            raise ImportError("langchain-openai package not installed")
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
            from langchain.schema import HumanMessage, SystemMessage
            
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
            {json.dumps(schema, indent=2)}
            
            Return only the JSON object, no additional text.
            """
            
            response = await self.generate(structured_prompt)
            return json.loads(response)
        except Exception as e:
            self.logger.error("OpenAI structured generation failed", error=str(e))
            raise


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider (requires langchain-anthropic package)."""
    
    async def initialize(self) -> None:
        """Initialize Anthropic provider."""
        try:
            from langchain_anthropic import ChatAnthropic
            
            self._client = ChatAnthropic(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                anthropic_api_key=self.config.api_key
            )
            self.logger.info("Anthropic provider initialized", model=self.config.model)
        except ImportError:
            self.logger.warning("langchain-anthropic not available, falling back to mock provider")
            raise ImportError("langchain-anthropic package not installed")
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
            from langchain.schema import HumanMessage, SystemMessage
            
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
            {json.dumps(schema, indent=2)}
            
            Return only the JSON object, no additional text.
            """
            
            response = await self.generate(structured_prompt)
            return json.loads(response)
        except Exception as e:
            self.logger.error("Anthropic structured generation failed", error=str(e))
            raise


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock LLM provider (requires langchain-aws package)."""
    
    async def initialize(self) -> None:
        """Initialize Bedrock provider."""
        try:
            from langchain_aws import ChatBedrock
            
            self._client = ChatBedrock(
                model_id=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                region_name=self.config.region
            )
            self.logger.info("Bedrock provider initialized", model=self.config.model)
        except ImportError:
            self.logger.warning("langchain-aws not available, falling back to mock provider")
            raise ImportError("langchain-aws package not installed")
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
            from langchain.schema import HumanMessage, SystemMessage
            
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
            {json.dumps(schema, indent=2)}
            
            Return only the JSON object, no additional text.
            """
            
            response = await self.generate(structured_prompt)
            return json.loads(response)
        except Exception as e:
            self.logger.error("Bedrock structured generation failed", error=str(e))
            raise


class VertexAIProvider(BaseLLMProvider):
    """Google Vertex AI LLM provider (requires langchain-google-vertexai package)."""
    
    async def initialize(self) -> None:
        """Initialize Vertex AI provider."""
        try:
            from langchain_google_vertexai import ChatVertexAI
            
            self._client = ChatVertexAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            self.logger.info("Vertex AI provider initialized", model=self.config.model)
        except ImportError:
            self.logger.warning("langchain-google-vertexai not available, falling back to mock provider")
            raise ImportError("langchain-google-vertexai package not installed")
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
            from langchain.schema import HumanMessage, SystemMessage
            
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
            {json.dumps(schema, indent=2)}
            
            Return only the JSON object, no additional text.
            """
            
            response = await self.generate(structured_prompt)
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
        
        try:
            return provider_class(config, agent_name)
        except ImportError as e:
            # Fall back to mock provider if specific provider package is not available
            logger = structlog.get_logger()
            logger.warning(
                f"Provider {config.provider} not available, using mock provider",
                provider=config.provider.value,
                agent=agent_name,
                error=str(e)
            )
            return MockLLMProvider(config, agent_name)
        except Exception as e:
            # Fall back to mock provider for any other error
            logger = structlog.get_logger()
            logger.warning(
                f"Provider {config.provider} failed to initialize, using mock provider",
                provider=config.provider.value,
                agent=agent_name,
                error=str(e)
            )
            return MockLLMProvider(config, agent_name)
    
    @classmethod
    def get_supported_providers(cls) -> List[LLMProvider]:
        """Get list of supported LLM providers."""
        return list(cls._providers.keys())