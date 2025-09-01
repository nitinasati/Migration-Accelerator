"""
Configuration settings for the Migration-Accelerators platform.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    BEDROCK = "bedrock"
    ANTHROPIC = "anthropic"
    VERTEXAI = "vertexai"


class FileFormat(str, Enum):
    """Supported file formats."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    XML = "xml"
    FIXED_WIDTH = "fixed_width"


class RecordType(str, Enum):
    """Insurance record types."""
    DISABILITY = "disability"
    ABSENCE = "absence"
    GROUP_POLICY = "group_policy"
    EMPLOYEE = "employee"
    CLAIM = "claim"


class TransformationType(str, Enum):
    """Data transformation types."""
    DIRECT = "direct"
    DATE_FORMAT = "date_format"
    LOOKUP = "lookup"
    CALCULATED = "calculated"
    CONDITIONAL = "conditional"


class ValidationSeverity(str, Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: LLMProvider = Field(..., description="LLM provider to use")
    model: str = Field(..., description="Model name/ID")
    temperature: float = Field(default=0.1, description="Generation temperature")
    max_tokens: int = Field(default=4000, description="Maximum tokens")
    api_key: Optional[str] = Field(None, description="API key")
    base_url: Optional[str] = Field(None, description="Base URL for API")
    region: Optional[str] = Field(None, description="AWS region for Bedrock")


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) configuration."""
    server_url: str = Field(..., description="MCP server URL")
    api_key: Optional[str] = Field(None, description="MCP API key")
    timeout: int = Field(default=30, description="Request timeout")
    max_retries: int = Field(default=3, description="Maximum retries")


class LangSmithConfig(BaseModel):
    """LangSmith configuration."""
    api_key: Optional[str] = Field(None, description="LangSmith API key")
    project: str = Field(default="migration-accelerators", description="Project name")
    tracing_v2: bool = Field(default=True, description="Enable tracing v2")
    endpoint: Optional[str] = Field(None, description="Custom endpoint")


class ValidationRule(BaseModel):
    """Validation rule configuration."""
    required: bool = Field(default=False, description="Field is required")
    pattern: Optional[str] = Field(None, description="Regex pattern")
    min_length: Optional[int] = Field(None, description="Minimum length")
    max_length: Optional[int] = Field(None, description="Maximum length")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    future_date: Optional[bool] = Field(None, description="Allow future dates")
    custom_rule: Optional[str] = Field(None, description="Custom validation rule")


class FieldMappingRule(BaseModel):
    """Field mapping rule configuration."""
    source_field: str = Field(..., description="Source field name")
    target_field: str = Field(..., description="Target field name")
    transformation_type: TransformationType = Field(..., description="Transformation type")
    validation: Optional[ValidationRule] = Field(None, description="Validation rules")
    source_format: Optional[str] = Field(None, description="Source format (for dates)")
    target_format: Optional[str] = Field(None, description="Target format (for dates)")
    lookup_table: Optional[Dict[str, str]] = Field(None, description="Lookup table")
    calculation_formula: Optional[str] = Field(None, description="Calculation formula")
    condition: Optional[str] = Field(None, description="Conditional logic")


class FieldMappingConfig(BaseModel):
    """Field mapping configuration."""
    source_format: FileFormat = Field(..., description="Source file format")
    target_format: FileFormat = Field(..., description="Target file format")
    record_type: RecordType = Field(..., description="Record type")
    version: str = Field(default="1.0", description="Mapping version")
    description: str = Field(default="", description="Mapping description")
    rules: List[FieldMappingRule] = Field(default_factory=list, description="Mapping rules")


class MigrationSettings(BaseSettings):
    """Main application settings."""
    
    # LLM Configuration
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="Default LLM provider")
    llm_model: str = Field(default="gpt-4o-mini", description="Default LLM model")
    llm_temperature: float = Field(default=0.1, description="Default LLM temperature")
    llm_max_tokens: int = Field(default=4000, description="Default max tokens")
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    aws_access_key_id: Optional[str] = Field(None, description="AWS access key ID")
    aws_secret_access_key: Optional[str] = Field(None, description="AWS secret access key")
    aws_region: str = Field(default="us-east-1", description="AWS region")
    
    # LangSmith Configuration
    langsmith_api_key: Optional[str] = Field(None, description="LangSmith API key")
    langchain_api_key: Optional[str] = Field(None, description="LangChain API key (alias for langsmith_api_key)")
    langsmith_project: str = Field(default="migration-accelerators", description="LangSmith project")
    langsmith_tracing_v2: bool = Field(default=True, description="Enable LangSmith tracing")
    
    # MCP Configuration
    mcp_server_url: str = Field(default="http://localhost:3000", description="MCP server URL")
    mcp_api_key: Optional[str] = Field(None, description="MCP API key")
    mcp_timeout: int = Field(default=30, description="MCP timeout")
    mcp_max_retries: int = Field(default=3, description="MCP max retries")
    
    # Database Configuration
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=8810, description="Database port")
    db_name: str = Field(default="migration", description="Database name")
    db_user: str = Field(default="postgres", description="Database username")
    db_password: Optional[str] = Field(None, description="Database password")
    db_schema: str = Field(default="public", description="Database schema")
    
    # File Processing
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    chunk_size: int = Field(default=1000, description="Processing chunk size")
    temp_directory: str = Field(default="./temp", description="Temporary directory")
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = MigrationSettings()


def get_llm_config() -> LLMConfig:
    """Get LLM configuration from settings."""
    return LLMConfig(
        provider=settings.llm_provider,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=getattr(settings, f"{settings.llm_provider.value}_api_key", None),
        region=settings.aws_region if settings.llm_provider == LLMProvider.BEDROCK else None
    )


def get_mcp_config() -> MCPConfig:
    """Get MCP configuration from settings."""
    return MCPConfig(
        server_url=settings.mcp_server_url,
        api_key=settings.mcp_api_key,
        timeout=settings.mcp_timeout,
        max_retries=settings.mcp_max_retries
    )


def get_langsmith_config() -> LangSmithConfig:
    """Get LangSmith configuration from settings."""
    # Use langsmith_api_key first, fallback to langchain_api_key
    api_key = settings.langsmith_api_key or settings.langchain_api_key
    
    return LangSmithConfig(
        api_key=api_key,
        project=settings.langsmith_project,
        tracing_v2=settings.langsmith_tracing_v2
    )
