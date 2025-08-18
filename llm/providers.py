"""
LLM provider abstractions for multiple providers.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json
import re

from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Bedrock
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import LLMConfig, LLMProvider


class LLMProviderBase(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = self._create_llm()
    
    @abstractmethod
    def _create_llm(self) -> BaseLLM:
        """Create the LLM instance."""
        pass
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        try:
            response = await self.llm.ainvoke(prompt, **kwargs)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
    
    async def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured output according to schema."""
        try:
            # Add schema information to prompt
            structured_prompt = f"""
{prompt}

Please provide your response in the following JSON schema:
{json.dumps(schema, indent=2)}

Ensure your response is valid JSON that matches this schema exactly.
"""
            
            response = await self.generate(structured_prompt, temperature=0.1)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            raise Exception(f"Structured generation failed: {str(e)}")
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        try:
            responses = await asyncio.gather(
                *[self.generate(prompt, **kwargs) for prompt in prompts]
            )
            return responses
        except Exception as e:
            raise Exception(f"Batch generation failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }


class OpenAIProvider(LLMProviderBase):
    """OpenAI LLM provider."""
    
    def _create_llm(self) -> BaseLLM:
        return ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            openai_api_key=self.config.api_key,
            base_url=self.config.base_url
        )


class AnthropicProvider(LLMProviderBase):
    """Anthropic LLM provider."""
    
    def _create_llm(self) -> BaseLLM:
        return ChatAnthropic(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            anthropic_api_key=self.config.api_key
        )


class BedrockProvider(LLMProviderBase):
    """AWS Bedrock LLM provider."""
    
    def _create_llm(self) -> BaseLLM:
        return Bedrock(
            model_id=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            region_name=self.config.region,
            aws_access_key_id=self.config.api_key,
            aws_secret_access_key=self.config.api_key  # This should be handled properly
        )


class VertexAIProvider(LLMProviderBase):
    """Google Vertex AI LLM provider."""
    
    def _create_llm(self) -> BaseLLM:
        return ChatGoogleGenerativeAI(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            google_api_key=self.config.api_key
        )


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.BEDROCK: BedrockProvider,
        LLMProvider.VERTEXAI: VertexAIProvider
    }
    
    @classmethod
    def create(cls, config: LLMConfig) -> LLMProviderBase:
        """Create LLM provider instance."""
        if config.provider not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
        
        provider_class = cls._providers[config.provider]
        return provider_class(config)
    
    @classmethod
    def register_provider(cls, provider: LLMProvider, provider_class: type):
        """Register a custom LLM provider."""
        cls._providers[provider] = provider_class


class LLMManager:
    """Manager for LLM operations."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = LLMProviderFactory.create(config)
    
    async def validate_data(self, data: Dict[str, Any], validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data using LLM."""
        prompt = f"""
Validate the following data against the specified rules:

Data:
{json.dumps(data, indent=2)}

Validation Rules:
{json.dumps(validation_rules, indent=2)}

Please provide validation results in JSON format:
{{
    "is_valid": true/false,
    "errors": ["error1", "error2"],
    "warnings": ["warning1", "warning2"],
    "score": 0.85
}}
"""
        
        schema = {
            "type": "object",
            "properties": {
                "is_valid": {"type": "boolean"},
                "errors": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "score": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }
        
        return await self.provider.generate_structured(prompt, schema)
    
    async def transform_data(self, source_data: Dict[str, Any], transformation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data using LLM."""
        prompt = f"""
Transform the following data according to the specified rules:

Source Data:
{json.dumps(source_data, indent=2)}

Transformation Rules:
{json.dumps(transformation_rules, indent=2)}

Please provide the transformed data in JSON format.
"""
        
        response = await self.provider.generate(prompt, temperature=0.1)
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in transformation response")
    
    async def analyze_business_rules(self, data: Dict[str, Any], business_context: str) -> Dict[str, Any]:
        """Analyze data against business rules using LLM."""
        prompt = f"""
Analyze the following insurance data against business rules:

Data:
{json.dumps(data, indent=2)}

Business Context:
{business_context}

Please provide analysis results in JSON format:
{{
    "compliance_score": 0.85,
    "business_rule_violations": ["violation1", "violation2"],
    "recommendations": ["recommendation1", "recommendation2"],
    "risk_level": "low|medium|high"
}}
"""
        
        schema = {
            "type": "object",
            "properties": {
                "compliance_score": {"type": "number", "minimum": 0, "maximum": 1},
                "business_rule_violations": {"type": "array", "items": {"type": "string"}},
                "recommendations": {"type": "array", "items": {"type": "string"}},
                "risk_level": {"type": "string", "enum": ["low", "medium", "high"]}
            }
        }
        
        return await self.provider.generate_structured(prompt, schema)
    
    async def enhance_mapping_rules(self, current_rules: List[Dict[str, Any]], source_format: str, target_format: str) -> List[Dict[str, Any]]:
        """Enhance mapping rules using LLM."""
        prompt = f"""
Analyze and enhance the following field mapping rules for converting from {source_format} to {target_format}:

Current Rules:
{json.dumps(current_rules, indent=2)}

Please suggest additional mapping rules in JSON format:
{{
    "additional_rules": [
        {{
            "source_field": "field_name",
            "target_field": "target_field_name",
            "transformation_type": "direct|date_format|lookup|calculated|conditional",
            "validation": {{
                "required": true/false,
                "pattern": "regex_pattern"
            }}
        }}
    ],
    "improvements": ["improvement1", "improvement2"]
}}
"""
        
        schema = {
            "type": "object",
            "properties": {
                "additional_rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_field": {"type": "string"},
                            "target_field": {"type": "string"},
                            "transformation_type": {"type": "string"},
                            "validation": {"type": "object"}
                        }
                    }
                },
                "improvements": {"type": "array", "items": {"type": "string"}}
            }
        }
        
        return await self.provider.generate_structured(prompt, schema)
