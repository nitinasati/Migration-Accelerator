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
    """Data record types."""
    CUSTOMER_DATA = "customer_data"
    PRODUCT_DATA = "product_data"
    TRANSACTION_DATA = "transaction_data"
    USER_DATA = "user_data"
    INVENTORY_DATA = "inventory_data"


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
    default_value: Optional[Any] = Field(None, description="Default value")
    calculation_formula: Optional[str] = Field(None, description="Calculation formula")
    condition: Optional[str] = Field(None, description="Conditional logic")


class FieldMappingConfig(BaseModel):
    """Field mapping configuration."""
    source_format: FileFormat = Field(..., description="Source file format")
    target_format: FileFormat = Field(..., description="Target file format")
    record_type: RecordType = Field(..., description="Record type")
    rules: List[FieldMappingRule] = Field(..., description="Mapping rules")
    version: str = Field(default="1.0", description="Mapping version")
    description: Optional[str] = Field(None, description="Mapping description")


class MigrationConfig(BaseModel):
    """Migration configuration."""
    migration_id: str = Field(..., description="Unique migration ID")
    batch_size: int = Field(default=1000, description="Batch size for processing")
    max_errors: int = Field(default=100, description="Maximum allowed errors")
    allow_partial_migration: bool = Field(default=False, description="Allow partial migration")
    strict_validation: bool = Field(default=True, description="Strict validation mode")
    dry_run: bool = Field(default=False, description="Dry run mode")
    target_system: str = Field(..., description="Target system identifier")


class PlatformSettings(BaseSettings):
    """Main platform settings."""
    
    # LLM Configuration
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4000, env="LLM_MAX_TOKENS")
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    aws_access_key_id: Optional[str] = Field(None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    
    # LangSmith Configuration
    langchain_api_key: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="insurance-migration", env="LANGCHAIN_PROJECT")
    langchain_tracing_v2: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    langchain_endpoint: Optional[str] = Field(None, env="LANGCHAIN_ENDPOINT")
    
    # MCP Configuration
    mcp_server_url: str = Field(default="http://localhost:3000", env="MCP_SERVER_URL")
    mcp_api_key: Optional[str] = Field(None, env="MCP_API_KEY")
    mcp_timeout: int = Field(default=30, env="MCP_TIMEOUT")
    mcp_max_retries: int = Field(default=3, env="MCP_MAX_RETRIES")
    
    # File Processing
    input_dir: str = Field(default="data/input", env="INPUT_DIR")
    output_dir: str = Field(default="data/output", env="OUTPUT_DIR")
    temp_dir: str = Field(default="data/temp", env="TEMP_DIR")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/migration.log", env="LOG_FILE")
    
    # Processing
    batch_size: int = Field(default=1000, env="BATCH_SIZE")
    max_errors: int = Field(default=100, env="MAX_ERRORS")
    timeout: int = Field(default=300, env="TIMEOUT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        api_key = None
        if self.llm_provider == LLMProvider.OPENAI:
            api_key = self.openai_api_key
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            api_key = self.anthropic_api_key
        
        return LLMConfig(
            provider=self.llm_provider,
            model=self.llm_model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            api_key=api_key,
            region=self.aws_region
        )
    
    def get_mcp_config(self) -> MCPConfig:
        """Get MCP configuration."""
        return MCPConfig(
            server_url=self.mcp_server_url,
            api_key=self.mcp_api_key,
            timeout=self.mcp_timeout,
            max_retries=self.mcp_max_retries
        )
    
    def get_langsmith_config(self) -> LangSmithConfig:
        """Get LangSmith configuration."""
        return LangSmithConfig(
            api_key=self.langchain_api_key,
            project=self.langchain_project,
            tracing_v2=self.langchain_tracing_v2,
            endpoint=self.langchain_endpoint
        )


# Global settings instance
settings = PlatformSettings()
