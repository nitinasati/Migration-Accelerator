"""
Database configuration and models for LangGraph state persistence.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not available, continue without it
    pass


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=8810, description="Database port")
    database: str = Field(default="migration", description="Database name")
    username: str = Field(default="postgres", description="Database username")
    password: Optional[str] = Field(default=None, description="Database password")
    schema: str = Field(default="public", description="Database schema")
    
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        if self.password:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            return f"postgresql://{self.username}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_string(self) -> str:
        """Generate async PostgreSQL connection string."""
        if self.password:
            return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            return f"postgresql+asyncpg://{self.username}@{self.host}:{self.port}/{self.database}"


def get_database_config() -> DatabaseConfig:
    """Get database configuration from environment variables or defaults."""
    return DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "8810")),
        database=os.getenv("DB_NAME", "migration"),
        username=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        schema=os.getenv("DB_SCHEMA", "public")
    )


class MigrationRun(BaseModel):
    """Model for migration run records."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique run ID")
    file_path: str = Field(description="Path to the source file")
    record_type: str = Field(description="Type of records being processed")
    status: str = Field(default="running", description="Current run status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Run creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    total_duration: Optional[float] = Field(default=None, description="Total duration in seconds")
    success: Optional[bool] = Field(default=None, description="Whether the run was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class WorkflowState(BaseModel):
    """Model for workflow state persistence."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique state ID")
    run_id: str = Field(description="Associated migration run ID")
    step_name: str = Field(description="Name of the workflow step")
    state_data: dict = Field(description="Serialized state data")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="State creation timestamp")
    step_order: int = Field(description="Order of the step in the workflow")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")


class WorkflowCheckpoint(BaseModel):
    """Model for workflow checkpoints."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique checkpoint ID")
    run_id: str = Field(description="Associated migration run ID")
    checkpoint_name: str = Field(description="Name of the checkpoint")
    state_data: dict = Field(description="Complete workflow state at checkpoint")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Checkpoint creation timestamp")
    metadata: Optional[dict] = Field(default=None, description="Additional checkpoint metadata")


class SearchFilters(BaseModel):
    """Model for search filters."""
    
    status: Optional[str] = Field(default=None, description="Filter by status")
    record_type: Optional[str] = Field(default=None, description="Filter by record type")
    file_path: Optional[str] = Field(default=None, description="Filter by file path")
    start_date: Optional[datetime] = Field(default=None, description="Filter by start date")
    end_date: Optional[datetime] = Field(default=None, description="Filter by end date")
    limit: int = Field(default=100, description="Maximum number of results")
    offset: int = Field(default=0, description="Number of results to skip")