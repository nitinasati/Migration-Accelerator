"""
Basic tests for the Migration-Accelerators platform.
"""

import pytest
import asyncio
import os
from pathlib import Path

from config.settings import settings, LLMConfig, MCPConfig
from llm.providers import LLMProviderFactory
from mcp_tools.client import MCPToolManager
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
        from config.settings import get_llm_config
        llm_config = get_llm_config()
        assert llm_config is not None
        assert llm_config.provider is not None
        assert llm_config.model is not None
    
    def test_mcp_config_creation(self):
        """Test MCP configuration creation."""
        from config.settings import get_mcp_config
        mcp_config = get_mcp_config()
        assert mcp_config is not None
        assert mcp_config.server_url is not None
        assert mcp_config.timeout > 0


class TestMappingConfiguration:
    """Test mapping configuration functionality."""
    
    def test_default_mapping_creation(self):
        """Test creating default mapping configuration."""
        mapping_config = create_default_mapping_config("disability")
        assert mapping_config is not None
        assert mapping_config.record_type.value == "disability"
        assert len(mapping_config.rules) > 0
    
    def test_mapping_validation(self):
        """Test mapping configuration validation."""
        mapping_config = create_default_mapping_config("disability")
        errors = validate_mapping_config(mapping_config)
        assert len(errors) == 0
    
    def test_mapping_validation_with_errors(self):
        """Test mapping validation with errors."""
        # Create invalid mapping
        mapping_config = create_default_mapping_config("disability")
        mapping_config.rules[0].source_field = ""  # Invalid empty field
        
        errors = validate_mapping_config(mapping_config)
        assert len(errors) > 0
        assert any("source_field is required" in error for error in errors)


class TestFileReader:
    """Test file reader functionality."""
    
    @pytest.mark.asyncio
    async def test_file_detection(self):
        """Test file format detection."""
        from agents.file_reader import FileReaderAgent
        
        file_reader = FileReaderAgent()
        
        # Test CSV detection
        csv_path = "data/input/sample_disability_data.csv"
        if os.path.exists(csv_path):
            format_detected = await file_reader._detect_file_format(csv_path, None)
            assert format_detected is not None
    
    @pytest.mark.asyncio
    async def test_csv_reading(self):
        """Test CSV file reading."""
        from agents.file_reader import FileReaderAgent
        
        file_reader = FileReaderAgent()
        csv_path = "data/input/sample_disability_data.csv"
        
        if os.path.exists(csv_path):
            context = {"encoding": "utf-8"}
            records = await file_reader._read_csv(csv_path, None, context)
            assert isinstance(records.data, list)
            assert len(records.data) > 0
            assert isinstance(records.data[0], dict)


class TestLLMProvider:
    """Test LLM provider functionality."""
    
    @pytest.mark.asyncio
    async def test_provider_factory(self):
        """Test LLM provider factory."""
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test_key"
        )
        
        provider = LLMProviderFactory.create(llm_config)
        assert provider is not None
        assert provider.config.provider == "openai"
    
    def test_supported_providers(self):
        """Test supported providers list."""
        providers = LLMProviderFactory.get_supported_providers()
        assert len(providers) > 0
        assert "openai" in [p.value for p in providers]


class TestMCPClient:
    """Test MCP client functionality."""
    
    @pytest.mark.asyncio
    async def test_mcp_tool_manager_creation(self):
        """Test MCP tool manager creation."""
        mcp_config = MCPConfig(
            server_url="http://localhost:3000",
            api_key="test_key"
        )
        
        manager = MCPToolManager(mcp_config)
        assert manager is not None
        assert manager.config.server_url == "http://localhost:3000"


class TestValidationAgent:
    """Test validation agent functionality."""
    
    @pytest.mark.asyncio
    async def test_validation_agent_creation(self):
        """Test validation agent creation."""
        from agents.validation import ValidationAgent
        
        agent = ValidationAgent()
        assert agent is not None
        assert agent.agent_name == "validation"
    
    @pytest.mark.asyncio
    async def test_field_validation(self):
        """Test field validation."""
        from agents.validation import ValidationAgent, ValidationRule
        
        agent = ValidationAgent()
        
        # Test required field validation
        rule = ValidationRule(required=True)
        result = await agent.validate_field("test_field", "", rule)
        
        assert not result.is_valid
        assert "required" in result.message.lower()


class TestMappingAgent:
    """Test mapping agent functionality."""
    
    @pytest.mark.asyncio
    async def test_mapping_agent_creation(self):
        """Test mapping agent creation."""
        from agents.mapping import MappingAgent
        
        agent = MappingAgent()
        assert agent is not None
        assert agent.agent_name == "mapping"
    
    @pytest.mark.asyncio
    async def test_direct_transformation(self):
        """Test direct transformation."""
        from agents.mapping import MappingAgent
        from config.settings import FieldMappingRule, TransformationType
        
        agent = MappingAgent()
        
        rule = FieldMappingRule(
            source_field="test_field",
            target_field="targetField",
            transformation_type=TransformationType.DIRECT
        )
        
        result = await agent._direct_transformation("test_value", rule)
        assert result == "test_value"


class TestTransformationAgent:
    """Test transformation agent functionality."""
    
    @pytest.mark.asyncio
    async def test_transformation_agent_creation(self):
        """Test transformation agent creation."""
        from agents.transformation import TransformationAgent
        
        agent = TransformationAgent()
        assert agent is not None
        assert agent.agent_name == "transformation"
    
    @pytest.mark.asyncio
    async def test_csv_to_json_conversion(self):
        """Test CSV to JSON conversion."""
        from agents.transformation import TransformationAgent
        
        agent = TransformationAgent()
        
        test_data = [
            {"field1": "value1", "field2": "value2"},
            {"field1": "value3", "field2": "value4"}
        ]
        
        result = await agent._csv_to_json(test_data, None, {})
        assert isinstance(result, str)
        
        import json
        parsed_result = json.loads(result)
        assert len(parsed_result) == 2


class TestAPIIntegrationAgent:
    """Test API integration agent functionality."""
    
    @pytest.mark.asyncio
    async def test_api_integration_agent_creation(self):
        """Test API integration agent creation."""
        from agents.api_integration import APIIntegrationAgent
        
        agent = APIIntegrationAgent()
        assert agent is not None
        assert agent.agent_name == "api_integration"
    
    def test_endpoint_registration(self):
        """Test API endpoint registration."""
        from agents.api_integration import APIIntegrationAgent
        
        agent = APIIntegrationAgent()
        
        # Test endpoint registration
        agent.api_endpoints = {}
        agent.api_endpoints["test_endpoint"] = {
            "url": "/api/test",
            "method": "POST"
        }
        
        assert "test_endpoint" in agent.api_endpoints





class TestMigrationWorkflow:
    """Test migration workflow functionality."""
    
    @pytest.mark.asyncio
    async def test_workflow_creation(self):
        """Test migration workflow creation."""
        from workflows.migration_graph import MigrationWorkflow
        
        from config.settings import get_llm_config, get_mcp_config
        llm_config = get_llm_config()
        mcp_config = get_mcp_config()
        workflow = MigrationWorkflow(llm_config, mcp_config)
        assert workflow is not None
        assert workflow.graph is not None
    
    def test_initial_state_creation(self):
        """Test initial state creation."""
        from workflows.migration_graph import MigrationState
        
        state: MigrationState = {
            "file_path": "test.csv",
            "mapping_config": None,
            "record_type": "disability",
            "target_system": {},
            "file_data": None,
            "validated_data": None,
            "mapped_data": None,
            "transformed_data": None,
            "api_results": None,
            "current_step": "initialization",
            "completed_steps": [],
            "errors": [],
            "warnings": [],
            "progress": 0.0,
            "llm_config": None,
            "mcp_config": None,
            "final_result": None,
            "success": False
        }
        
        assert state["file_path"] == "test.csv"
        assert state["record_type"] == "disability"
        assert state["current_step"] == "initialization"


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test end-to-end workflow with sample data."""
        from workflows.migration_graph import MigrationWorkflow
        from config.mappings import load_mapping_config
        
        # Check if sample data exists
        sample_file = "data/input/sample_disability_data.csv"
        mapping_file = "config/mappings/disability_mapping.yaml"
        
        if not os.path.exists(sample_file) or not os.path.exists(mapping_file):
            pytest.skip("Sample data or mapping file not found")
        
        # Load mapping configuration
        mapping_config = load_mapping_config(mapping_file)
        
        # Create workflow
        from config.settings import get_llm_config, get_mcp_config
        llm_config = get_llm_config()
        mcp_config = get_mcp_config()
        workflow = MigrationWorkflow(llm_config, mcp_config)
        
        # Run workflow (dry run)
        result = await workflow.run(
            file_path=sample_file,
            mapping_config=mapping_config,
            record_type="disability",
            target_system={"dry_run": True}
        )
        
        # Verify result
        assert result is not None
        assert "migration_summary" in result
        
        # Close workflow
        await workflow.close()


if __name__ == "__main__":
    pytest.main([__file__])
