#!/usr/bin/env python3
"""
Example usage of the Migration-Accelerators platform

This script demonstrates how to use the platform to migrate data
from legacy systems to modern platforms using Agentic AI.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings, LLMConfig, MCPConfig
from llm.providers import LLMProviderFactory
from mcp.client import MCPToolManager
from workflows.migration_graph import MigrationWorkflow
from config.mappings import load_mapping_config, create_default_mapping_config


async def example_migration():
    """Example of a complete migration workflow."""
    
    print("🚀 Starting Migration-Accelerators Platform Example")
    print("=" * 60)
    
    # Step 1: Initialize components
    print("\n1. Initializing components...")
    
    # Initialize LLM provider
    try:
        llm_config = settings.get_llm_config()
        llm_provider = LLMProviderFactory.create(llm_config)
        print("   ✓ LLM provider initialized")
    except Exception as e:
        print(f"   ⚠ LLM provider not available: {e}")
        print("   Running without LLM enhancements")
        llm_provider = None
    
    # Initialize MCP manager
    try:
        mcp_config = settings.get_mcp_config()
        mcp_manager = MCPToolManager(mcp_config)
        await mcp_manager.initialize()
        print("   ✓ MCP manager initialized")
    except Exception as e:
        print(f"   ⚠ MCP manager not available: {e}")
        print("   Running without MCP tools")
        mcp_manager = None
    
    # Step 2: Load or create mapping configuration
    print("\n2. Loading mapping configuration...")
    
    mapping_file = "config/mappings/sample_mapping.yaml"
    try:
        mapping_config = load_mapping_config(mapping_file)
        print(f"   ✓ Mapping configuration loaded from {mapping_file}")
    except FileNotFoundError:
        print(f"   ⚠ Mapping file not found, creating default...")
        mapping_config = create_default_mapping_config("customer_data")
        print("   ✓ Default mapping configuration created")
    
    # Step 3: Check input file
    print("\n3. Checking input file...")
    
    input_file = "data/input/sample_data.csv"
    if not os.path.exists(input_file):
        print(f"   ❌ Input file not found: {input_file}")
        print("   Please ensure the sample data file exists")
        return
    else:
        print(f"   ✓ Input file found: {input_file}")
    
    # Step 4: Initialize workflow
    print("\n4. Initializing migration workflow...")
    
    workflow = MigrationWorkflow(llm_config, mcp_config)
    print("   ✓ Migration workflow initialized")
    
    # Step 5: Execute migration
    print("\n5. Executing migration workflow...")
    
    try:
        result = await workflow.run(
            file_path=input_file,
            mapping_config=mapping_config,
            record_type="customer_data"
        )
        
        # Step 6: Display results
        print("\n6. Migration Results:")
        print("=" * 40)
        
        if result["success"]:
            print("   ✅ Migration completed successfully!")
        else:
            print("   ❌ Migration failed")
            if "error" in result:
                print(f"   Error: {result['error']}")
        
        summary = result.get("summary", {})
        
        print(f"   Total Records: {summary.get('total_records', 0)}")
        print(f"   Successful: {summary.get('successful_records', 0)}")
        print(f"   Failed: {summary.get('failed_records', 0)}")
        
        success_rate = summary.get("success_rate", 0)
        print(f"   Success Rate: {success_rate:.1%}")
        
        duration = summary.get("duration_seconds")
        if duration:
            print(f"   Duration: {duration:.2f} seconds")
        
        # Display errors and warnings
        errors = summary.get("errors", [])
        warnings = summary.get("warnings", [])
        
        if errors:
            print(f"\n   Errors ({len(errors)}):")
            for error in errors[:5]:  # Show first 5 errors
                print(f"     • {error}")
            if len(errors) > 5:
                print(f"     ... and {len(errors) - 5} more")
        
        if warnings:
            print(f"\n   Warnings ({len(warnings)}):")
            for warning in warnings[:5]:  # Show first 5 warnings
                print(f"     • {warning}")
            if len(warnings) > 5:
                print(f"     ... and {len(warnings) - 5} more")
        
        # Display agent status
        print(f"\n7. Agent Status:")
        print("=" * 40)
        
        agent_status = await workflow.get_agent_status()
        for agent_name, status in agent_status.items():
            print(f"   • {agent_name}: {status['status']}")
            if 'duration_seconds' in status['metrics']:
                print(f"     Duration: {status['metrics']['duration_seconds']:.2f}s")
            if 'records_processed' in status['metrics']:
                print(f"     Records: {status['metrics']['records_processed']}")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await workflow.cleanup()
        if mcp_manager:
            await mcp_manager.close()


async def example_individual_agents():
    """Example of using individual agents."""
    
    print("\n🔧 Individual Agent Examples")
    print("=" * 40)
    
    # Example 1: File Reader Agent
    print("\n1. File Reader Agent Example:")
    
    from agents.file_reader import FileReaderAgent
    
    file_reader = FileReaderAgent()
    input_file = "data/input/sample_disability_data.csv"
    
    if os.path.exists(input_file):
        result = await file_reader.execute({
            "file_path": input_file,
            "file_format": "csv",
            "record_type": "disability"
        })
        
        print(f"   • Records read: {result['total_records']}")
        print(f"   • File format: {result['file_format']}")
        print(f"   • Record type: {result['record_type']}")
        
        # Show first record
        if result['records']:
            first_record = result['records'][0]
            print(f"   • First record policy: {first_record.get('policy_number', 'N/A')}")
            print(f"   • First record employee: {first_record.get('employee_id', 'N/A')}")
    
    # Example 2: LLM Provider
    print("\n2. LLM Provider Example:")
    
    try:
        llm_config = settings.get_llm_config()
        llm_provider = LLMProviderFactory.create(llm_config)
        
        # Test LLM generation
        prompt = "What is the capital of France?"
        response = await llm_provider.generate(prompt)
        print(f"   • LLM Response: {response[:100]}...")
        
    except Exception as e:
        print(f"   • LLM not available: {e}")
    
    # Example 3: MCP Tools
    print("\n3. MCP Tools Example:")
    
    try:
        mcp_config = settings.get_mcp_config()
        mcp_manager = MCPToolManager(mcp_config)
        await mcp_manager.initialize()
        
        # List available tools
        tools = await mcp_manager.list_available_tools()
        print(f"   • Available tools: {len(tools)}")
        for tool in tools:
            print(f"     - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
        
        await mcp_manager.close()
        
    except Exception as e:
        print(f"   • MCP not available: {e}")


def example_configuration():
    """Example of configuration management."""
    
    print("\n⚙️ Configuration Examples")
    print("=" * 40)
    
    # Show current settings
    print("\n1. Current Platform Settings:")
    
    print(f"   • LLM Provider: {settings.llm_provider.value}")
    print(f"   • LLM Model: {settings.llm_model}")
    print(f"   • MCP Server: {settings.mcp_server_url}")
    print(f"   • LangSmith Project: {settings.langchain_project}")
    print(f"   • Input Directory: {settings.input_dir}")
    print(f"   • Output Directory: {settings.output_dir}")
    
    # Show mapping configuration
    print("\n2. Mapping Configuration Example:")
    
    mapping_file = "config/mappings/disability_mapping.yaml"
    if os.path.exists(mapping_file):
        try:
            mapping_config = load_mapping_config(mapping_file)
            print(f"   • Source Format: {mapping_config.source_format}")
            print(f"   • Target Format: {mapping_config.target_format}")
            print(f"   • Record Type: {mapping_config.record_type}")
            print(f"   • Number of Rules: {len(mapping_config.rules)}")
            
            # Show first few rules
            for i, rule in enumerate(mapping_config.rules[:3]):
                print(f"     Rule {i+1}: {rule.source_field} → {rule.target_field} ({rule.transformation_type.value})")
            
        except Exception as e:
            print(f"   • Error loading mapping: {e}")
    else:
        print("   • No mapping file found")


def main():
    """Main function to run all examples."""
    
    print("🎯 Agentic Insurance Data Migration Platform - Examples")
    print("=" * 70)
    
    # Run configuration examples
    example_configuration()
    
    # Run individual agent examples
    asyncio.run(example_individual_agents())
    
    # Run complete migration example
    print("\n" + "=" * 70)
    asyncio.run(example_migration())
    
    print("\n🎉 All examples completed!")
    print("\nTo run the platform with your own data:")
    print("  python main.py migrate your_data.csv --mapping your_mapping.yaml")
    print("\nTo validate your setup:")
    print("  python main.py validate your_data.csv")
    print("\nTo check platform status:")
    print("  python main.py status")


if __name__ == "__main__":
    main()
