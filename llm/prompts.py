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

    SEQUENTIAL PROCESSING STEPS (Execute in exact order):
    1. ANALYZE METADATA: Use analyze_file_metadata_tool to get file metadata (type, size, timestamps, MIME type)
       - Use this information to understand the file format and choose appropriate processing parameters
    
    2. VALIDATE FILE: Use validate_file_tool with the file_path parameter
       - Based on metadata from step 1, determine if file type and size are acceptable
       - Use the file extension from metadata to set appropriate validation rules
    
    3. READ CONTENT: Use read_file_tool to read and parse the complete file contents
       - Use metadata from step 1 to determine optimal reading parameters (max_lines, etc.)
       - Adjust processing approach based on file type (CSV vs JSON vs XML)
    
    4. ANALYZE DATA: Create a structured JSON response with your complete analysis
       - Include metadata from step 1, validation results from step 2, and content from step 3
       - Determine data types, field names, record count, and structure patterns
    
    5. VALIDATE OUTPUT: Use parse_json_output_tool to validate your JSON response format
       - Pass your complete JSON object to parse_json_output_tool
       - After validation succeeds, return EXACTLY the same JSON object
       - Do NOT add any text, explanations, or formatting around the JSON

    CRITICAL PROCESSING REQUIREMENTS:
    1. EXECUTE STEPS SEQUENTIALLY - Do not skip steps or execute them out of order
    2. USE PREVIOUS OUTPUTS - Each step should inform the parameters and approach for subsequent steps
    3. ADAPTIVE PROCESSING - Adjust your approach based on what you learn from metadata and validation
    4. COMPLETE ANALYSIS - Include all relevant information for downstream workflow agents
    5. VALIDATED OUTPUT - Final response must pass parse_json_output_tool validation
    6. JSON ONLY RESPONSE - After parse_json_output_tool validation, return ONLY the JSON object

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

    CRITICAL FINAL INSTRUCTIONS:
    1. After using parse_json_output_tool successfully, your FINAL message must contain ONLY the JSON object
    2. Do NOT include any explanations, markdown formatting, or additional text
    3. Do NOT wrap the JSON in code blocks or backticks
    4. Return the exact JSON that was validated by parse_json_output_tool
    5. If any step fails, return a valid JSON error response with this format:
       {{"success": false, "error": "description of error", "step_failed": "step_name"}}
    6. ALWAYS return valid JSON - never return plain text or error messages without JSON structure
    5. The 'file_data' array must contain ALL records from the file
    6. Ensure the JSON is properly formatted and complete
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
    
    # File Reading Prompts
    FILE_CONVERSION_BASE = """
    Convert the following file content into structured JSON.

    CRITICAL INSTRUCTIONS:
    1. Return ONLY valid JSON - no explanations, no markdown, no code blocks
    2. Do NOT wrap the JSON in ```json or ``` 
    3. Do NOT include any text before or after the JSON
    4. Start your response directly with {{ and end with }}
    5. Analyze the content structure and create appropriate JSON representation
    6. Include metadata about the conversion

    FILE CONTENT:
    {file_content}

    Convert this to a JSON structure with:
    - Metadata (source_file, file_type, total_records)
    - Structured data based on content analysis
    - For CSV files: convert rows to array of objects with column headers as keys
    - For text files: structure data logically based on content patterns
    - For mainframe files: use layout information to parse fixed-width fields
    """
    
    FILE_CONVERSION_WITH_LAYOUT = """
    You are a mainframe data conversion expert. Convert the following fixed-width mainframe data file to structured JSON using the provided layout file.

    CRITICAL INSTRUCTIONS:
    1. Return ONLY valid JSON - no explanations, no markdown, no code blocks
    2. Do NOT wrap the JSON in ```json or ``` 
    3. Do NOT include any text before or after the JSON
    4. Start your response directly with {{ and end with }}
    5. Use the layout file to understand field positions and meanings
    6. Convert each record to a JSON object with meaningful field names
    7. Include metadata about the conversion (source files, record count, etc.)

    LAYOUT FILE CONTENT:
    {layout_content}

    MAINFRAME DATA FILE CONTENT:
    {file_content}

    Convert this to a JSON structure with:
    - Metadata (source_file, layout_file, total_records)
    - Array of records with properly named fields based on the layout
    - Each record should have all fields from the layout with appropriate data types
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
    
    # Mapping Agent Prompts
    MAPPING_SELECT_CONFIG = """
    You are a data mapping expert. Analyze the provided data sample and select the most appropriate mapping configuration.

    DATA SAMPLE:
    {data_sample}

    AVAILABLE MAPPING CONFIGURATIONS:
    {available_mappings}

    INSTRUCTIONS:
    1. Analyze the data structure, field names, and content patterns
    2. Match the data to the most appropriate mapping configuration
    3. Consider field names, data types, and business context
    4. Return your selection in valid JSON format

    REQUIRED JSON RESPONSE FORMAT:
    {{
        "selected_mapping": "mapping_config_name",
        "record_type": "detected_record_type",
        "confidence": 0.95,
        "reasoning": "Brief explanation of why this mapping was selected",
        "field_analysis": {{
            "primary_fields": ["field1", "field2"],
            "data_types": {{"field1": "string", "field2": "number"}},
            "patterns_detected": ["pattern1", "pattern2"]
        }}
    }}

    IMPORTANT: Return ONLY valid JSON. No explanations or markdown formatting.
    """
    
    MAPPING_SYSTEM = """
    You are an expert data mapping specialist with deep knowledge of:
    - Data transformation patterns and best practices
    - Field mapping strategies for different data types
    - Business logic and data validation rules
    - Record type detection and classification
    
    Your role is to analyze data structures and select appropriate mapping configurations
    that will ensure accurate and reliable data transformations.
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


# Module-level constants for direct import
FILE_CONVERSION_BASE = PromptTemplates.FILE_CONVERSION_BASE
FILE_CONVERSION_WITH_LAYOUT = PromptTemplates.FILE_CONVERSION_WITH_LAYOUT