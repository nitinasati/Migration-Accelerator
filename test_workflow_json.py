#!/usr/bin/env python3
"""
Test the complete workflow JSON response with MCP tools simulation.
"""

import asyncio
import os
from config.settings import LLMConfig, LLMProvider, settings
from llm.providers import LLMProviderFactory
from llm.prompts import PromptTemplates

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


async def test_workflow_json():
    """Test the complete workflow JSON generation."""
    
    print("🔍 Testing Workflow JSON Generation")
    print("=" * 35)
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ No OPENROUTER_API_KEY found")
        return
    
    print(f"✅ API key found: {api_key[:15]}...")
    
    # Configure for workflow (more conservative settings)
    config = LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model="deepseek/deepseek-chat-v3.1:free",
        temperature=0.1,  # Very low for consistency
        max_tokens=4000,  # Higher limit for complex responses
        api_key=api_key,
        region=None
    )
    
    print(f"📋 Config: {config.provider.value} - {config.model}")
    
    try:
        # Create provider
        provider = LLMProviderFactory.create(config, "workflow_test_agent")
        await provider.initialize()
        print("✅ Provider initialized")
        
        # Test the actual workflow prompt
        print("\n🧪 Testing Full Workflow Prompt")
        
        # Use the actual prompt from the system
        file_path = "test_file.csv"  # Mock file path
        workflow_prompt = PromptTemplates.FILE_READER_LANGGRAPH_MCP_WORKFLOW.format(file_path=file_path)
        
        # Add system message for more control
        system_message = """You are a precise file processing agent. You must:
1. Follow the sequential steps exactly
2. Always return valid JSON
3. If any step fails, return {"success": false, "error": "description"}
4. Never return plain text responses
5. Always end with ONLY the JSON object"""
        
        print("📤 Sending workflow prompt...")
        
        try:
            response = await provider.generate(
                workflow_prompt,
                system_message=system_message,
                temperature=0.1,
                max_tokens=4000
            )
            print(f"📥 Response length: {len(response)} characters")
            print(f"📥 First 200 chars: {response[:200]}...")
            print(f"📥 Last 200 chars: ...{response[-200:]}")
            
            # Try to parse as JSON
            import json
            try:
                parsed = json.loads(response)
                print(f"✅ Valid JSON Response!")
                print(f"🔑 Keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'Not a dict'}")
                
                # Check expected structure
                if isinstance(parsed, dict):
                    if "success" in parsed:
                        print(f"✅ Has 'success' field: {parsed['success']}")
                    if "error" in parsed:
                        print(f"⚠️ Has 'error' field: {parsed['error']}")
                    if "file_data" in parsed:
                        print(f"✅ Has 'file_data' field")
                    if "metadata" in parsed:
                        print(f"✅ Has 'metadata' field")
                        
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parse Error: {e}")
                print(f"📝 Full response: {repr(response)}")
                
                # Try to find JSON in response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        extracted_json = json_match.group()
                        parsed = json.loads(extracted_json)
                        print(f"✅ Extracted valid JSON: {list(parsed.keys())}")
                    except:
                        print(f"❌ Extracted JSON also invalid")
                else:
                    print(f"❌ No JSON pattern found in response")
        
        except Exception as e:
            print(f"❌ Generation Error: {e}")
            print(f"📝 Error type: {type(e).__name__}")
            
            # Check if it's a server error
            if "server had an error" in str(e).lower():
                print("🔄 Server error detected - this is the root cause!")
                print("💡 Suggestions:")
                print("   1. Reduce prompt complexity")
                print("   2. Add retry logic")
                print("   3. Use simpler model")
                print("   4. Check rate limits")
            
    except Exception as e:
        print(f"❌ Provider Error: {e}")
        import traceback
        traceback.print_exc()


async def test_simple_prompt():
    """Test with a much simpler prompt."""
    print("\n🧪 Testing Simplified Prompt")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    config = LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model="deepseek/deepseek-chat-v3.1:free",
        temperature=0.1,
        max_tokens=1000,  # Much smaller
        api_key=api_key,
        region=None
    )
    
    provider = LLMProviderFactory.create(config, "simple_test_agent")
    await provider.initialize()
    
    simple_prompt = """Process this sample CSV data and return JSON:
name,age,city
John,30,NYC
Jane,25,LA

Return this exact format:
{
    "success": true,
    "file_data": [
        {"name": "John", "age": 30, "city": "NYC"},
        {"name": "Jane", "age": 25, "city": "LA"}
    ],
    "metadata": {
        "total_records": 2,
        "file_type": "csv"
    }
}"""
    
    try:
        response = await provider.generate(simple_prompt)
        print(f"📥 Simple response: {response}")
        
        import json
        parsed = json.loads(response)
        print(f"✅ Simple JSON valid: {list(parsed.keys())}")
        
    except Exception as e:
        print(f"❌ Simple test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_workflow_json())
    asyncio.run(test_simple_prompt())
