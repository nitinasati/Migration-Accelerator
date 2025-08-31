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
    
    FILE_READER_DETECT_FORMAT = """
    Analyze the following file content and determine its format:
    
    File path: {file_path}
    File size: {file_size} bytes
    First 1000 characters:
    {file_preview}
    
    Please identify:
    1. File format (CSV, Excel, XML, Fixed-width, JSON, etc.)
    2. Encoding (UTF-8, ASCII, etc.)
    3. Delimiter (if applicable)
    4. Header row presence
    5. Any special formatting or structure
    
    Respond with a JSON object containing your analysis.
    """
    
    # Validation Agent Prompts
    VALIDATION_SYSTEM = """
    You are a Validation Agent specialized in data integrity and business rule validation.
    Your role is to ensure data quality and compliance with business rules for insurance data migration.
    
    Key responsibilities:
    1. Validate data types and formats
    2. Check business rule compliance
    3. Identify missing or invalid data
    4. Provide detailed error reports
    5. Suggest data corrections when possible
    
    Focus on insurance-specific validation rules for disability and absence management.
    """
    
    VALIDATION_CHECK_RECORD = """
    Validate the following insurance record against business rules:
    
    Record Type: {record_type}
    Record Data: {record_data}
    Validation Rules: {validation_rules}
    
    Please check:
    1. Required field presence
    2. Data format compliance
    3. Business rule adherence
    4. Data consistency
    5. Range and constraint validation
    
    Provide a detailed validation report with any errors or warnings found.
    """
    
    # Mapping Agent Prompts
    MAPPING_SYSTEM = """
    You are a Mapping Agent specialized in field transformation and data mapping.
    Your role is to intelligently map fields from source formats to target formats.
    
    Key responsibilities:
    1. Apply field mapping rules
    2. Transform data according to business logic
    3. Handle data type conversions
    4. Apply lookup tables and calculations
    5. Manage conditional transformations
    
    Ensure data integrity during transformation and maintain audit trails.
    """
    
    MAPPING_TRANSFORM_RECORD = """
    Transform the following record using the provided mapping rules:
    
    Source Record: {source_record}
    Mapping Rules: {mapping_rules}
    Record Type: {record_type}
    
    Apply the following transformations:
    1. Direct field mapping
    2. Date format conversions
    3. Lookup table applications
    4. Calculated field derivations
    5. Conditional logic processing
    
    Return the transformed record in the target format.
    """
    
    # Transformation Agent Prompts
    TRANSFORMATION_SYSTEM = """
    You are a Transformation Agent specialized in data format conversion and optimization.
    Your role is to convert data between different formats while maintaining data integrity.
    
    Key responsibilities:
    1. Convert between CSV, JSON, XML formats
    2. Optimize data structure for target systems
    3. Handle nested data and complex structures
    4. Ensure format compliance
    5. Maintain data relationships
    
    Focus on creating clean, well-structured output suitable for modern systems.
    """
    
    TRANSFORMATION_CONVERT_FORMAT = """
    Convert the following data from {source_format} to {target_format}:
    
    Source Data: {source_data}
    Target Schema: {target_schema}
    Conversion Rules: {conversion_rules}
    
    Ensure:
    1. All required fields are present
    2. Data types are correctly converted
    3. Nested structures are properly handled
    4. Format compliance is maintained
    5. Data relationships are preserved
    
    Return the converted data in the target format.
    """
    
    # API Integration Agent Prompts
    API_INTEGRATION_SYSTEM = """
    You are an API Integration Agent specialized in making API calls to target systems.
    Your role is to manage API interactions using the Model Context Protocol (MCP).
    
    Key responsibilities:
    1. Make API calls to target systems
    2. Handle authentication and authorization
    3. Manage request/response processing
    4. Implement retry logic and error handling
    5. Monitor API performance and health
    
    Ensure reliable data transmission and proper error recovery.
    """
    
    API_INTEGRATION_PROCESS_BATCH = """
    Process the following batch of records for API submission:
    
    Batch Data: {batch_data}
    API Endpoint: {api_endpoint}
    Authentication: {auth_info}
    Processing Rules: {processing_rules}
    
    For each record:
    1. Validate data before submission
    2. Apply any required transformations
    3. Make API call with proper error handling
    4. Process response and handle errors
    5. Update status and logging
    
    Return a summary of the batch processing results.
    """
    
    # Orchestration Agent Prompts
    ORCHESTRATION_SYSTEM = """
    You are an Orchestration Agent that coordinates the entire migration workflow.
    Your role is to manage the flow of data between different agents and ensure successful completion.
    
    Key responsibilities:
    1. Coordinate agent interactions
    2. Manage workflow state and progress
    3. Handle error recovery and retry logic
    4. Monitor overall system health
    5. Provide status updates and reporting
    
    Ensure smooth data flow and handle any issues that arise during migration.
    """
    
    ORCHESTRATION_DECIDE_NEXT_STEP = """
    Based on the current workflow state, decide the next step:
    
    Current State: {current_state}
    Completed Steps: {completed_steps}
    Errors Encountered: {errors}
    Data Status: {data_status}
    
    Available Actions:
    1. Continue to next agent
    2. Retry current step
    3. Skip to error handling
    4. Pause for manual intervention
    5. Complete workflow
    
    Provide your decision with reasoning and any required parameters.
    """
    
    # Error Handling Prompts
    ERROR_ANALYSIS = """
    Analyze the following error and provide recommendations:
    
    Error Type: {error_type}
    Error Message: {error_message}
    Context: {context}
    Agent: {agent_name}
    
    Please provide:
    1. Root cause analysis
    2. Recommended actions
    3. Prevention strategies
    4. Recovery options
    5. Impact assessment
    
    Focus on actionable solutions for error resolution.
    """
    
    # Data Quality Prompts
    DATA_QUALITY_ASSESSMENT = """
    Assess the quality of the following dataset:
    
    Dataset: {dataset}
    Record Count: {record_count}
    Validation Results: {validation_results}
    
    Evaluate:
    1. Completeness (missing data)
    2. Accuracy (correct values)
    3. Consistency (format compliance)
    4. Validity (business rule compliance)
    5. Uniqueness (duplicate detection)
    
    Provide a quality score and recommendations for improvement.
    """
    
    # Validation Agent Prompts
    VALIDATION_CHECK_RECORD = """
    Validate the following insurance record for data quality and business rule compliance:
    
    Record Type: {record_type}
    Record Data: {record_data}
    Validation Rules: {validation_rules}
    
    Please check:
    1. Required field presence
    2. Data format compliance
    3. Business rule adherence
    4. Cross-field consistency
    5. Data type accuracy
    
    Provide a validation report with any issues found.
    """
    
    VALIDATION_CHECK_BATCH = """
    Validate the following batch of {batch_size} insurance records for data quality and business rule compliance:
    
    Record Type: {record_type}
    Records Data: {records_data}
    Validation Rules: {validation_rules}
    
    Please check each record for:
    1. Required field presence
    2. Data format compliance
    3. Business rule adherence
    4. Cross-field consistency
    5. Data type accuracy
    
    Provide a validation report for each record with any issues found.
    Format your response to clearly indicate which record (by index) has which issues.
    """
    
    # Business Rule Prompts
    BUSINESS_RULE_VALIDATION = """
    Validate the following data against insurance business rules:
    
    Data: {data}
    Record Type: {record_type}
    Business Rules: {business_rules}
    
    Check compliance with:
    1. Policy number format requirements
    2. Date range validations
    3. Amount and percentage constraints
    4. Status transition rules
    5. Coverage type validations
    
    Report any violations and suggest corrections.
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
        agent_type: Type of agent (file_reader, validation, mapping, etc.)
        
    Returns:
        str: System prompt
    """
    system_prompt_name = f"{agent_type.upper()}_SYSTEM"
    return getattr(PromptTemplates, system_prompt_name, "")


def get_validation_schema(record_type: str) -> Dict[str, Any]:
    """
    Get validation schema for a record type.
    
    Args:
        record_type: Type of record to validate
        
    Returns:
        Dict[str, Any]: Validation schema
    """
    schemas = {
        "disability": {
            "type": "object",
            "required": ["policy_number", "employee_id", "effective_date", "status"],
            "properties": {
                "policy_number": {"type": "string", "pattern": "^[A-Z0-9]{6,12}$"},
                "employee_id": {"type": "string", "pattern": "^EMP[0-9]{6}$"},
                "effective_date": {"type": "string", "format": "date"},
                "benefit_amount": {"type": "number", "minimum": 0},
                "status": {"type": "string", "enum": ["active", "inactive", "pending", "cancelled", "suspended", "terminated"]}
            }
        },
        "absence": {
            "type": "object",
            "required": ["employee_id", "absence_start_date", "absence_type"],
            "properties": {
                "employee_id": {"type": "string", "pattern": "^EMP[0-9]{6}$"},
                "absence_start_date": {"type": "string", "format": "date"},
                "absence_type": {"type": "string", "enum": ["sick", "vacation", "personal", "maternity"]},
                "duration_days": {"type": "integer", "minimum": 1, "maximum": 365}
            }
        },
        "group_policy": {
            "type": "object",
            "required": ["group_policy_number", "employer_id", "effective_date"],
            "properties": {
                "group_policy_number": {"type": "string", "pattern": "^GP[0-9]{8}$"},
                "employer_id": {"type": "string"},
                "effective_date": {"type": "string", "format": "date"},
                "coverage_type": {"type": "string", "enum": ["std", "ltd", "both"]}
            }
        }
    }
    
    return schemas.get(record_type.lower(), schemas["disability"])
