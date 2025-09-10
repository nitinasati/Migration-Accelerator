"""
LLM API Client for Migration-Accelerator

This client communicates with the LLM-API-Service to make LLM calls.
It provides a unified interface for all LLM operations.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union
import structlog
import httpx
import asyncio
from abc import ABC, abstractmethod

logger = structlog.get_logger(__name__)

# Constants
UNKNOWN_ERROR_MSG = "Unknown error"
PROVIDER_NOT_INITIALIZED_MSG = "Provider not initialized. Call initialize() first."


class LLMAPIClient(ABC):
    """Abstract base class for LLM API clients."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        provider: str = "openai",
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using the LLM API service."""
        pass
    
    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        output_schema: Optional[Dict[str, Any]] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured output using the LLM API service."""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate chat completion using the LLM API service."""
        pass


class LLMServiceClient(LLMAPIClient):
    """Client for communicating with the LLM-API-Service."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize the LLM API client.
        
        Args:
            base_url: Base URL of the LLM-API-Service
            api_key: API key for authentication (optional if service doesn't require auth)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.logger = structlog.get_logger(__name__).bind(client="llm_api")
        
    async def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to the LLM API service."""
        url = f"{self.base_url}/api/v1/llm/{endpoint}"
        
        # Prepare headers
        request_headers = {"Content-Type": "application/json"}
        if self.api_key:
            request_headers["Authorization"] = f"Bearer {self.api_key}"
        if headers:
            request_headers.update(headers)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                self.logger.debug("Making LLM API request", url=url, endpoint=endpoint)
                
                response = await client.post(
                    url,
                    json=data,
                    headers=request_headers
                )
                
                response.raise_for_status()
                result = response.json()
                
                self.logger.debug("LLM API response received", status_code=response.status_code)
                return result
                
        except httpx.HTTPStatusError as e:
            self.logger.error(
                "LLM API HTTP error",
                status_code=e.response.status_code,
                response_text=e.response.text
            )
            raise Exception(f"LLM API HTTP error {e.response.status_code}: {e.response.text}")
            
        except httpx.TimeoutException:
            self.logger.error("LLM API request timeout")
            raise Exception("LLM API request timeout")
            
        except Exception as e:
            self.logger.error("LLM API request failed", error=str(e))
            raise Exception(f"LLM API request failed: {str(e)}")
    
    async def generate(
        self,
        prompt: str,
        provider: str = "openai",
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the LLM API service.
        
        Args:
            prompt: Input prompt for text generation
            provider: LLM provider to use (openai, anthropic, bedrock, vertex, deepseek, mock)
            model: Model name to use (optional, will use provider default)
            system_message: System message for context (optional)
            temperature: Temperature for randomness (optional)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        # Prepare request data
        request_data = {
            "prompt": prompt,
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Add model if specified
        if model:
            request_data["model"] = model
        
        # Add system message as additional parameter if provided
        if system_message:
            request_data["additional_params"] = request_data.get("additional_params", {})
            request_data["additional_params"]["system_message"] = system_message
        
        try:
            result = await self._make_request("generate", request_data)
            
            if result.get("success", False):
                return result.get("text", "")
            else:
                error_msg = result.get("error", UNKNOWN_ERROR_MSG)
                self.logger.error("LLM generation failed", error=error_msg)
                raise RuntimeError(f"LLM generation failed: {error_msg}")
                
        except Exception as e:
            self.logger.error("Error in generate", error=str(e))
            raise
    
    async def generate_structured(
        self,
        prompt: str,
        output_schema: Optional[Dict[str, Any]] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured output using the LLM API service.
        
        Args:
            prompt: Input prompt for structured generation
            output_schema: JSON schema for output validation (optional)
            provider: LLM provider to use (openai, anthropic, bedrock, vertex, deepseek, mock)
            model: Model name to use (optional, will use provider default)
            system_prompt: System prompt for context (optional)
            temperature: Temperature for randomness (optional)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional parameters
            
        Returns:
            Structured output as dictionary
        """
        # Prepare request data
        request_data = {
            "prompt": prompt,
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Add model if specified
        if model:
            request_data["model"] = model
        
        # Add system prompt if provided
        if system_prompt:
            request_data["system_prompt"] = system_prompt
        
        # Add output schema if provided
        if output_schema:
            request_data["output_schema"] = output_schema
        
        try:
            result = await self._make_request("structured", request_data)
            
            if result.get("success", False):
                return result.get("structured_output", {})
            else:
                error_msg = result.get("error", UNKNOWN_ERROR_MSG)
                self.logger.error("LLM structured generation failed", error=error_msg)
                raise RuntimeError(f"LLM structured generation failed: {error_msg}")
                
        except Exception as e:
            self.logger.error("Error in generate_structured", error=str(e))
            raise
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion using the LLM API service.
        
        Args:
            messages: List of chat messages with 'role' and 'content' keys
            provider: LLM provider to use (openai, anthropic, bedrock, vertex, deepseek, mock)
            model: Model name to use (optional, will use provider default)
            temperature: Temperature for randomness (optional)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional parameters
            
        Returns:
            Chat completion text
        """
        # Prepare request data
        request_data = {
            "messages": messages,
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Add model if specified
        if model:
            request_data["model"] = model
        
        try:
            result = await self._make_request("chat", request_data)
            
            if result.get("success", False):
                return result.get("text", "")
            else:
                error_msg = result.get("error", UNKNOWN_ERROR_MSG)
                self.logger.error("LLM chat completion failed", error=error_msg)
                raise RuntimeError(f"LLM chat completion failed: {error_msg}")
                
        except Exception as e:
            self.logger.error("Error in chat", error=str(e))
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the LLM API service."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            raise Exception(f"Health check failed: {str(e)}")


def create_llm_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 30.0
) -> LLMServiceClient:
    """
    Factory function to create an LLM API client.
    
    Args:
        base_url: Base URL of the LLM-API-Service (defaults to environment variable or localhost)
        api_key: API key for authentication (defaults to environment variable)
        timeout: Request timeout in seconds
        
    Returns:
        Configured LLMServiceClient instance
    """
    # Get configuration from environment variables if not provided
    if base_url is None:
        base_url = os.getenv("LLM_API_SERVICE_URL", "http://localhost:8000")
    
    if api_key is None:
        api_key = os.getenv("LLM_API_SERVICE_KEY")
    
    return LLMServiceClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout
    )
