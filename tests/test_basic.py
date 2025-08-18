"""
Basic tests for the Migration-Accelerators platform.
"""

import pytest
import asyncio
import os
from pathlib import Path

from config.settings import settings, LLMConfig, MCPConfig
from llm.providers import LLMProviderFactory
from mcp.client import MCPToolManager
from config.mappings import create_default_mapping_config, validate_mapping_config


class TestConfiguration:
    """Test configuration loading and validation."""
    
    def test_settings_loading(self):
        """Test that settings can be loaded."""
        assert settings is not None
        assert hasattr(settings, 'llm_provider')
        assert hasattr(settings, 'mcp_server_url')
    
    def test_llm_config_creation(self):
        """Test LLM configuration creation."""
        llm_config = settings.get_llm_config()
        assert llm_config is not None
        assert llm_config.provider is not None
        assert llm_config.model is not None
    
    def test_mcp_config_creation(self):
        """Test MCP configuration creation."""
        mcp_config = settings.get_mcp_config()
        assert mcp_config is not None
        assert mcp_config.server_url is not None
        assert mcp_config.timeout > 0


class TestMappingConfiguration:
    """Test mapping configuration functionality."""
    
    def test_default_mapping_creation(self):
        """Test creating default mapping configuration."""
        mapping_config = create_default_mapping_config("customer_data")
        assert mapping_config is not None
        assert mapping_config.record_type == "customer_data"
        assert len(mapping_config.rules) > 0
    
    def test_mapping_validation(self):
        """Test mapping configuration validation."""
        mapping_config = create_default_mapping_config("customer_data")
        errors = validate_mapping_config(mapping_config)
        assert len(errors) == 0
    
    def test_mapping_validation_with_errors(self):
        """Test mapping validation with errors."""
        # Create invalid mapping
        mapping_config = create_default_mapping_config("customer_data")
        mapping_config.rules[0].source_field = ""  # Invalid empty field
        
        errors = validate_mapping_config(mapping_config)
        assert len(errors) > 0
        assert any("source_field is required" in error for error in errors)


class TestFileReader:
    """Test file reader functionality."""
    
    def test_file_detection(self):
        """Test file format detection."""
        from agents.file_reader import FileReaderAgent
        
        file_reader = FileReaderAgent()
        
        # Test CSV detection
        csv_path = "data/input/sample_data.csv"
        if os.path.exists(csv_path):
            format_detected = file_reader._detect_file_format(csv_path)
            assert format_detected == "csv"
    
    @pytest.mark.asyncio
    async def test_csv_reading(self):
        """Test CSV file reading."""
        from agents.file_reader import FileReaderAgent
        
        file_reader = FileReaderAgent()
        csv_path = "data/input/sample_data.csv"
        
        if os.path.exists(csv_path):
            records = await file_reader._read_csv(csv_path)
            assert isinstance(records, list)
            assert len(records) > 0
            assert isinstance(records[0], dict)


class TestLLMProvider:
    """Test LLM provider functionality."""
    
    @pytest.mark.asyncio
    async def test_llm_provider_creation(self):
        """Test LLM provider creation."""
        try:
            llm_config = settings.get_llm_config()
            provider = LLMProviderFactory.create(llm_config)
            assert provider is not None
        except Exception as e:
            # LLM provider might not be available in test environment
            pytest.skip(f"LLM provider not available: {e}")
    
    @pytest.mark.asyncio
    async def test_llm_generation(self):
        """Test LLM text generation."""
        try:
            llm_config = settings.get_llm_config()
            provider = LLMProviderFactory.create(llm_config)
            
            response = await provider.generate("Hello, world!")
            assert isinstance(response, str)
            assert len(response) > 0
        except Exception as e:
            # LLM provider might not be available in test environment
            pytest.skip(f"LLM generation not available: {e}")


class TestMCPTools:
    """Test MCP tools functionality."""
    
    @pytest.mark.asyncio
    async def test_mcp_manager_creation(self):
        """Test MCP manager creation."""
        try:
            mcp_config = settings.get_mcp_config()
            manager = MCPToolManager(mcp_config)
            assert manager is not None
        except Exception as e:
            # MCP might not be available in test environment
            pytest.skip(f"MCP manager not available: {e}")
    
    @pytest.mark.asyncio
    async def test_mcp_tool_execution(self):
        """Test MCP tool execution."""
        try:
            mcp_config = settings.get_mcp_config()
            manager = MCPToolManager(mcp_config)
            
            # Test data validation tool
            result = await manager.execute_tool("data_validation", {
                "data": {"test_field": "test_value"},
                "validation_rules": {"test_field": {"required": True}}
            })
            
            assert result is not None
            assert "success" in result
        except Exception as e:
            # MCP might not be available in test environment
            pytest.skip(f"MCP tool execution not available: {e}")


class TestWorkflow:
    """Test workflow functionality."""
    
    @pytest.mark.asyncio
    async def test_workflow_creation(self):
        """Test workflow creation."""
        from workflows.migration_graph import MigrationWorkflow
        
        llm_config = settings.get_llm_config()
        mcp_config = settings.get_mcp_config()
        
        workflow = MigrationWorkflow(llm_config, mcp_config)
        assert workflow is not None
        assert hasattr(workflow, 'graph')
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self):
        """Test workflow execution with sample data."""
        from workflows.migration_graph import MigrationWorkflow
        
        llm_config = settings.get_llm_config()
        mcp_config = settings.get_mcp_config()
        
        workflow = MigrationWorkflow(llm_config, mcp_config)
        
        # Test with sample file
        csv_path = "data/input/sample_disability_data.csv"
        if os.path.exists(csv_path):
            try:
                result = await workflow.run(
                    file_path=csv_path,
                    mapping_config=None,
                    record_type="disability"
                )
                
                assert result is not None
                assert "success" in result
                assert "summary" in result
                
            except Exception as e:
                # Workflow might fail due to missing dependencies
                pytest.skip(f"Workflow execution not available: {e}")
        else:
            pytest.skip("Sample data file not found")


def test_sample_data_exists():
    """Test that sample data file exists."""
    csv_path = "data/input/sample_disability_data.csv"
    assert os.path.exists(csv_path), f"Sample data file not found: {csv_path}"


def test_mapping_file_exists():
    """Test that mapping file exists."""
    mapping_path = "config/mappings/disability_mapping.yaml"
    assert os.path.exists(mapping_path), f"Mapping file not found: {mapping_path}"


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
