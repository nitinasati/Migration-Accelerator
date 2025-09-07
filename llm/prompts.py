"""
Prompt templates for the Migration-Accelerators platform.
Contains only the prompts that are actively used in the codebase.
"""

from typing import Dict, Any, List


class PromptTemplates:
    """Collection of prompt templates for different agents."""
    
    # File Reader Agent Prompts
    FILE_READER_LANGGRAPH_MCP_WORKFLOW = """
    You are a file processing agent in a LangGraph migration workflow. Process the file '{file_path}' and provide structured output for the next workflow step.

    WORKFLOW CONTEXT:
    - This is part of a data migration pipeline
    - Your output will be used by downstream mapping and transformation agents
    - Return data in the exact format expected by the workflow state

    PROCESSING STEPS:
    1. Use analyze_file_metadata_tool to get file metadata (type, size, timestamps, MIME type)
    2. Use validate_file_tool with appropriate extension (.csv, .json, .xml) and max_size_mb=100
    3. Use read_file_tool to read and parse the complete file contents
    4. Create a structured JSON response with your analysis
    5. Use parse_json_output_tool to validate your JSON response format

    CRITICAL OUTPUT REQUIREMENTS:
    1. After using all tools, create a structured JSON response
    2. Use parse_json_output_tool to validate your JSON structure 
    3. Return ONLY the validated JSON object (no explanations or markdown)

    Expected JSON structure:
    {{
        "success": true,
        "file_data": [
            {{"field1": "value1", "field2": "value2"}},
            {{"field1": "value3", "field2": "value4"}}
        ],
        "metadata": {{
            "total_records": <number>,
            "file_type": "<csv|json|xml>",
            "file_size_mb": <number>,
            "columns": ["field1", "field2", ...],
            "processing_timestamp": "<ISO timestamp>"
        }},
        "validation_results": {{
            "file_valid": true,
            "extension_valid": true,
            "size_valid": true
        }}
    }}

    FINAL INSTRUCTION: Your final response must be ONLY the validated JSON object. The 'file_data' array must contain ALL records from the file.
    """
    
    # Mapping Agent Prompts
    MAPPING_SYSTEM = """
    You are a Mapping Agent specialized in intelligent field transformation and data mapping.
    Your role is to analyze data attributes and select appropriate mapping configurations,
    then transform data using LLM intelligence.
    
    Key responsibilities:
    1. Analyze input data attributes to determine record type
    2. Select appropriate mapping configuration files
    3. Apply intelligent field transformations
    4. Handle data type conversions and business logic
    5. Generate properly structured output data
    
    Use your intelligence to understand data patterns and apply appropriate transformations.
    """
    
    MAPPING_SELECT_CONFIG = """
    Analyze the following input data and determine the most appropriate mapping configuration:
    
    Input Data Sample: {data_sample}
    Available Mapping Files: {available_mappings}
    
    Please:
    1. Analyze the data attributes and structure
    2. Identify the record type (disability, absence, group_policy, etc.)
    3. Match the data fields to the most appropriate mapping configuration
    4. Consider field names, data patterns, and business context
    
    Return your analysis in this format:
    {{
        "selected_mapping": "mapping_file_name",
        "record_type": "detected_record_type",
        "confidence": 0.95,
        "reasoning": "explanation of why this mapping was selected",
        "matched_fields": ["field1", "field2", ...]
    }}
    """
    
    MAPPING_TRANSFORM_WITH_LLM = """
    Transform the following data using the selected mapping configuration:
    
    Source Data: {source_data}
    Mapping Configuration: {mapping_config}
    Record Type: {record_type}
    Target Format: JSON
    
    Please:
    1. Apply all field mapping rules from the configuration
    2. Transform data types according to business logic
    3. Apply lookup tables and calculated fields
    4. Handle conditional transformations
    5. Ensure all required fields are present
    6. Maintain data integrity and relationships
    
    Return the transformed data in this format:
    {{
        "success": true,
        "transformed_data": [
            {{"field1": "value1", "field2": "value2", ...}},
            ...
        ],
        "metadata": {{
            "record_type": "detected_type",
            "records_count": number_of_records,
            "mapping_applied": "mapping_file_name",
            "transformation_timestamp": "ISO_timestamp"
        }}
    }}
    """


def get_prompt(template_name: str, **kwargs) -> str:
    """
    Get a formatted prompt template.
    
    Args:
        template_name: Name of the prompt template
        **kwargs: Template variables
        
    Returns:
        str: Formatted prompt
    """
    template = getattr(PromptTemplates, template_name.upper(), None)
    if template is None:
        raise ValueError(f"Prompt template '{template_name}' not found")
    
    return template.format(**kwargs)


def get_system_prompt(agent_type: str) -> str:
    """
    Get system prompt for an agent type.
    
    Args:
        agent_type: Type of agent (file_reader, mapping, transformation, etc.)
        
    Returns:
        str: System prompt
    """
    system_prompt_name = f"{agent_type.upper()}_SYSTEM"
    return getattr(PromptTemplates, system_prompt_name, "")