"""
Prompt templates for the Migration-Accelerators platform.
"""

from typing import Dict, Any, List


class PromptTemplates:
    """Collection of prompt templates for different agents."""
    
    # File Reader Agent Prompts
    FILE_READER_SYSTEM = """
    You are a File Reader Agent specialized in processing mainframe and legacy system files.
    Your role is to intelligently read and parse various file formats including CSV, Excel, XML, and fixed-width files.
    
    Key responsibilities:
    1. Detect file format automatically
    2. Parse file content with proper encoding handling
    3. Handle different delimiters and data structures
    4. Extract metadata and file information
    5. Validate file integrity and structure
    
    Always provide structured output with clear error handling and detailed logging.
    """
    

    
    FILE_READER_READ_FILE = """
    Read and parse the following file content into structured data:
    
    File path: {file_path}
    File format: {file_format}
    File content:
    {file_content}
    
    Please:
    1. Parse the file content according to the detected format
    2. Extract all records/rows of data
    3. Identify column headers and field names
    4. Handle any encoding or formatting issues
    5. Return the data as a JSON array of objects
    
    For each record, ensure:
    - All fields are properly extracted
    - Data types are preserved (strings, numbers, dates)
    - Empty or null values are handled appropriately
    - Special characters are properly encoded
    
    Return the parsed data in this format:
    {{
        "success": true,
        "data": [
            {{"field1": "value1", "field2": "value2", ...}},
            {{"field1": "value3", "field2": "value4", ...}},
            ...
        ],
        "metadata": {{
            "format": "detected_format",
            "records_count": number_of_records,
            "columns": ["field1", "field2", ...],
            "encoding": "detected_encoding"
        }}
    }}
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



