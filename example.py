#!/usr/bin/env python3
"""
Example usage of the Migration-Accelerators platform

This script demonstrates how to use the platform to migrate insurance data
from legacy mainframe systems to modern platforms.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings, LLMConfig, MCPConfig
from llm.providers import LLMProviderFactory
from mcp_tools.client import MCPToolManager
from workflows.migration_graph import MigrationWorkflow



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
    
    # Step 2: Mapping configuration (handled automatically by LLM agent)
    print("\n2. Mapping configuration...")
    print("   ✓ Using LLM-powered intelligent mapping")
    
    # Step 3: Check input file
    print("\n3. Checking input file...")
    
    input_file = "data/input/sample_disability_data.csv"
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
            record_type="disability"
        )
        
        # Step 6: Display results
        print("\n6. Migration Results:")
        print("=" * 40)
        
        if result["migration_summary"]["success"]:
            print("   ✅ Migration completed successfully!")
        else:
            print("   ❌ Migration failed")
        
        summary = result["migration_summary"]
        print(f"   📊 Records processed: {summary.get('total_records_processed', 0)}")
        print(f"   📈 Progress: {summary.get('progress', 0):.1f}%")
        print(f"   ✅ Completed steps: {len(summary.get('completed_steps', []))}")
        
        # Show completed steps
        completed_steps = summary.get("completed_steps", [])
        if completed_steps:
            print("\n   Completed Steps:")
            for step in completed_steps:
                print(f"     ✓ {step.replace('_', ' ').title()}")
        
        # Show errors if any
        errors = result.get("errors", [])
        if errors:
            print(f"\n   ⚠ Errors ({len(errors)}):")
            for error in errors[:3]:  # Show first 3 errors
                print(f"     • {error}")
            if len(errors) > 3:
                print(f"     ... and {len(errors) - 3} more errors")
        
        # Show warnings if any
        warnings = result.get("warnings", [])
        if warnings:
            print(f"\n   ⚠ Warnings ({len(warnings)}):")
            for warning in warnings[:3]:  # Show first 3 warnings
                print(f"     • {warning}")
            if len(warnings) > 3:
                print(f"     ... and {len(warnings) - 3} more warnings")
        
        # Step 7: Show data pipeline
        print("\n7. Data Pipeline:")
        print("=" * 40)
        
        pipeline = result.get("data_pipeline", {})
        
        if pipeline.get("file_data"):
            print(f"   📁 File Data: {len(pipeline['file_data'])} records")
        

        
        if pipeline.get("mapped_data"):
            print(f"   🔄 Mapped Data: {len(pipeline['mapped_data'])} records")
        
        if pipeline.get("transformed_data"):
            transformed_data = pipeline["transformed_data"]
            if isinstance(transformed_data, str):
                print(f"   🔧 Transformed Data: JSON string ({len(transformed_data)} characters)")
            else:
                print(f"   🔧 Transformed Data: {len(transformed_data)} records")
        
        if pipeline.get("api_results"):
            api_results = pipeline["api_results"]
            if isinstance(api_results, dict) and "api_results" in api_results:
                print(f"   🌐 API Results: {len(api_results['api_results'])} API calls")
            else:
                print(f"   🌐 API Results: Available")
        
    except Exception as e:
        print(f"   ❌ Migration failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Step 8: Cleanup
        print("\n8. Cleaning up...")
        await workflow.close()
        if mcp_manager:
            await mcp_manager.close()
        print("   ✓ Cleanup completed")
    
    print("\n🎉 Example completed!")
    print("=" * 60)


async def example_absence_migration():
    """Example of absence data migration."""
    
    print("\n🚀 Starting Absence Data Migration Example")
    print("=" * 60)
    
    # Check if absence data exists
    input_file = "data/input/sample_absence_data.csv"
    mapping_file = "config/mappings/absence_mapping.yaml"
    
    if not os.path.exists(input_file):
        print(f"   ❌ Absence data file not found: {input_file}")
        return
    
    if not os.path.exists(mapping_file):
        print(f"   ❌ Absence mapping file not found: {mapping_file}")
        return
    
    try:
        # Load mapping configuration
        mapping_config = load_mapping_config(mapping_file)
        print(f"   ✓ Absence mapping configuration loaded")
        
        # Initialize workflow
        llm_config = settings.get_llm_config()
        mcp_config = settings.get_mcp_config()
        workflow = MigrationWorkflow(llm_config, mcp_config)
        
        # Run migration
        result = await workflow.run(
            file_path=input_file,
            mapping_config=mapping_config,
            record_type="absence"
        )
        
        # Display results
        summary = result["migration_summary"]
        print(f"   📊 Absence records processed: {summary.get('total_records_processed', 0)}")
        print(f"   ✅ Success: {summary.get('success', False)}")
        
        await workflow.close()
        
    except Exception as e:
        print(f"   ❌ Absence migration failed: {e}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(example_migration())
    
    # Run absence migration example
    asyncio.run(example_absence_migration())
