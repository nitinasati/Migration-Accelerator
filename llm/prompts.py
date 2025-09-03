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
    
    MAPPING_ANALYSIS_BUSINESS_RULES = """
    Analyze business rules and logic for data transformation:
    
    SOURCE BUSINESS RULES: {source_business_rules}
    TARGET BUSINESS RULES: {target_business_rules}
    
    DATA CONTEXT: {data_context}
    BUSINESS PROCESS: {business_process}
    
    Please analyze the business rule transformations required:
    
    1. BUSINESS RULE MAPPING:
       - Source rules → Target rules mapping
       - Rule conflicts and resolutions
       - New business logic requirements
       - Deprecated rule handling
    
    2. TRANSFORMATION LOGIC:
       - Conditional logic requirements
       - Calculation formulas
       - Decision tree structures
       - Exception handling
    
    3. IMPLEMENTATION APPROACH:
       - Rule engine requirements
       - Code implementation strategy
       - Testing and validation
       - Rule maintenance
    
    Return business rule analysis in this format:
    {{
        "success": true,
        "business_rules_analysis": {{
            "rule_mappings": [
                {{
                    "source_rule": "source_rule_description",
                    "target_rule": "target_rule_description",
                    "transformation_logic": "transformation_description",
                    "implementation": "implementation_approach",
                    "testing": "testing_requirements"
                }}
            ],
            "new_rules": [
                {{
                    "rule_description": "new_rule_description",
                    "business_justification": "why_this_rule_is_needed",
                    "implementation": "implementation_approach",
                    "testing": "testing_requirements"
                }}
            ],
            "deprecated_rules": [
                {{
                    "rule_description": "deprecated_rule",
                    "replacement": "replacement_rule",
                    "migration_strategy": "how_to_handle_existing_data"
                }}
            ],
            "implementation_strategy": {{
                "approach": "rule_engine|code|hybrid",
                "complexity": "low|medium|high",
                "estimated_effort": "effort_estimate",
                "maintenance_requirements": "maintenance_needs"
            }}
        }}
    }}
    """
    
    # Data Mapping Analysis Prompts
    MAPPING_ANALYSIS_SYSTEM = """
    You are a Data Mapping Analysis Agent specialized in understanding relationships between source and target data structures.
    Your role is to analyze legacy system data and identify how it maps to modern platform requirements.
    
    Key responsibilities:
    1. Analyze source data structure and field definitions
    2. Identify target system requirements and field mappings
    3. Create comprehensive field-to-field mapping documentation
    4. Identify data transformation requirements and business rules
    5. Document data quality issues and validation requirements
    6. Provide recommendations for optimal data flow
    
    Use your intelligence to understand business context and create accurate mapping specifications.
    """
    
    MAPPING_ANALYSIS_COMPREHENSIVE = """
    Analyze the mapping relationship between source data and target data for data migration:
    
    SOURCE DATA STRUCTURE:
    {source_data_structure}
    
    TARGET DATA STRUCTURE:
    {target_data_structure}
    
    BUSINESS CONTEXT:
    {business_context}
    
    Please provide a comprehensive mapping analysis including:
    
    1. FIELD MAPPING MATRIX:
       - Source field → Target field mapping
       - Data type transformations required
       - Business logic and calculations
       - Default values and fallbacks
    
    2. DATA TRANSFORMATION REQUIREMENTS:
       - Format conversions (dates, numbers, text)
       - Lookup table mappings
       - Conditional logic and business rules
       - Data validation requirements
    
    3. DATA QUALITY ASSESSMENT:
       - Missing data handling
       - Data consistency issues
       - Required vs optional fields
       - Data integrity constraints
    
    4. MIGRATION STRATEGY:
       - Recommended transformation approach
       - Batch vs real-time processing
       - Error handling and rollback procedures
       - Testing and validation approach
    
    Return your analysis in this structured format:
    {{
        "success": true,
        "mapping_analysis": {{
            "field_mappings": [
                {{
                    "source_field": "field_name",
                    "target_field": "target_field_name",
                    "transformation_type": "direct|calculated|lookup|conditional",
                    "transformation_rule": "detailed_rule_description",
                    "data_type": "source_type → target_type",
                    "required": true|false,
                    "default_value": "default_if_missing",
                    "validation_rules": ["rule1", "rule2"]
                }}
            ],
            "transformation_requirements": {{
                "format_conversions": ["conversion1", "conversion2"],
                "lookup_mappings": ["lookup1", "lookup2"],
                "business_rules": ["rule1", "rule2"],
                "calculated_fields": ["calculation1", "calculation2"]
            }},
            "data_quality": {{
                "missing_data_fields": ["field1", "field2"],
                "consistency_issues": ["issue1", "issue2"],
                "validation_requirements": ["validation1", "validation2"]
            }},
            "migration_strategy": {{
                "approach": "batch|real_time|hybrid",
                "error_handling": "error_handling_strategy",
                "testing_approach": "testing_strategy",
                "rollback_procedure": "rollback_steps"
            }},
            "metadata": {{
                "analysis_timestamp": "ISO_timestamp",
                "confidence_score": 0.95,
                "complexity_level": "low|medium|high",
                "estimated_effort": "effort_estimate"
            }}
        }}
    }}
    """
    
    MAPPING_ANALYSIS_FIELD_DETAILED = """
    Provide detailed field-level mapping analysis for data migration:
    
    SOURCE FIELD: {source_field_name}
    Source Field Details: {source_field_details}
    
    TARGET FIELD: {target_field_name}
    Target Field Requirements: {target_field_requirements}
    
    BUSINESS CONTEXT: {business_context}
    
    Please analyze this specific field mapping and provide:
    
    1. FIELD MAPPING ANALYSIS:
       - Direct mapping possibility
       - Required transformations
       - Business logic implications
       - Data quality considerations
    
    2. TRANSFORMATION SPECIFICATION:
       - Exact transformation steps
       - Input validation rules
       - Output validation rules
       - Error handling approach
    
    3. IMPLEMENTATION DETAILS:
       - Code/script requirements
       - Testing scenarios
       - Performance considerations
       - Maintenance requirements
    
    Return detailed analysis in this format:
    {{
        "success": true,
        "field_mapping": {{
            "source_field": "field_name",
            "target_field": "target_field_name",
            "mapping_type": "direct|calculated|lookup|conditional|complex",
            "transformation_steps": [
                "step1_description",
                "step2_description",
                "step3_description"
            ],
            "business_logic": "detailed_business_logic_explanation",
            "validation_rules": {{
                "input_validation": ["rule1", "rule2"],
                "output_validation": ["rule1", "rule2"],
                "business_validation": ["rule1", "rule2"]
            }},
            "error_handling": {{
                "error_scenarios": ["scenario1", "scenario2"],
                "handling_strategies": ["strategy1", "strategy2"],
                "fallback_values": "fallback_value"
            }},
            "implementation": {{
                "code_requirements": "code_description",
                "testing_scenarios": ["test1", "test2"],
                "performance_notes": "performance_considerations",
                "maintenance_notes": "maintenance_requirements"
            }},
            "metadata": {{
                "complexity": "low|medium|high",
                "risk_level": "low|medium|high",
                "estimated_effort": "effort_estimate"
            }}
        }}
    }}
    """
    
    MAPPING_ANALYSIS_BUSINESS_RULES = """
    Analyze business rules and logic for data transformation:
    
    SOURCE BUSINESS RULES: {source_business_rules}
    TARGET BUSINESS RULES: {target_business_rules}
    
    DATA CONTEXT: {data_context}
    BUSINESS PROCESS: {business_process}
    
    Please analyze the business rule transformations required:
    
    1. BUSINESS RULE MAPPING:
       - Source rules → Target rules mapping
       - Rule conflicts and resolutions
       - New business logic requirements
       - Deprecated rule handling
    
    2. TRANSFORMATION LOGIC:
       - Conditional logic requirements
       - Calculation formulas
       - Decision tree structures
       - Exception handling
    
    3. IMPLEMENTATION APPROACH:
       - Rule engine requirements
       - Code implementation strategy
       - Testing and validation
       - Rule maintenance
    
    Return business rule analysis in this format:
    {{
        "success": true,
        "business_rules_analysis": {{
            "rule_mappings": [
                {{
                    "source_rule": "source_rule_description",
                    "target_rule": "target_rule_description",
                    "transformation_logic": "transformation_description",
                    "implementation": "implementation_approach",
                    "testing": "testing_requirements"
                }}
            ],
            "new_rules": [
                {{
                    "rule_description": "new_rule_description",
                    "business_justification": "why_this_rule_is_needed",
                    "implementation": "implementation_approach",
                    "testing": "testing_requirements"
                }}
            ],
            "deprecated_rules": [
                {{
                    "rule_description": "deprecated_rule",
                    "replacement": "replacement_rule",
                    "migration_strategy": "how_to_handle_existing_data"
                }}
            ],
            "implementation_strategy": {{
                "approach": "rule_engine|code|hybrid",
                "complexity": "low|medium|high",
                "estimated_effort": "effort_estimate",
                "maintenance_requirements": "maintenance_needs"
            }}
        }}
    }}
    """
    
    MAPPING_ANALYSIS_DATA_SAMPLE = """
    Analyze data sample mapping for migration from legacy system to modern platform:
    
    SOURCE DATA SAMPLE:
    {source_data_sample}
    
    TARGET SCHEMA:
    {target_schema}
    
    BUSINESS CONTEXT: {business_context}
    
    Please analyze this data sample and provide:
    
    1. DATA STRUCTURE ANALYSIS:
       - Field-by-field mapping analysis
       - Data type compatibility assessment
       - Business logic identification
       - Data quality observations
    
    2. MAPPING RECOMMENDATIONS:
       - Direct field mappings
       - Calculated field requirements
       - Lookup table needs
       - Conditional logic requirements
    
    3. TRANSFORMATION SPECIFICATIONS:
       - Exact transformation steps for each field
       - Data validation requirements
       - Error handling strategies
       - Performance considerations
    
    4. IMPLEMENTATION GUIDANCE:
       - Code/script templates
       - Testing scenarios
       - Data validation rules
       - Migration checklist
    
    Return detailed analysis in this format:
    {{
        "success": true,
        "data_sample_analysis": {{
            "sample_summary": {{
                "record_count": number_of_records,
                "field_count": number_of_fields,
                "data_types": ["type1", "type2"],
                "quality_score": "high|medium|low"
            }},
            "field_mappings": [
                {{
                    "source_field": "field_name",
                    "source_value_example": "example_value",
                    "target_field": "target_field_name",
                    "mapping_type": "direct|calculated|lookup|conditional",
                    "transformation_rule": "detailed_transformation_description",
                    "validation_required": true|false,
                    "business_logic": "business_logic_explanation"
                }}
            ],
            "data_quality_issues": [
                {{
                    "issue_type": "missing_data|format_inconsistency|business_rule_violation",
                    "field_affected": "field_name",
                    "description": "issue_description",
                    "severity": "high|medium|low",
                    "recommended_action": "action_to_take"
                }}
            ],
            "transformation_requirements": {{
                "format_conversions": ["conversion1", "conversion2"],
                "lookup_mappings": ["lookup1", "lookup2"],
                "business_rules": ["rule1", "rule2"],
                "calculated_fields": ["calculation1", "calculation2"]
            }},
            "implementation_guide": {{
                "code_template": "code_template_or_description",
                "testing_scenarios": ["test1", "test2"],
                "validation_rules": ["validation1", "validation2"],
                "migration_checklist": ["check1", "check2"]
            }},
            "metadata": {{
                "analysis_timestamp": "ISO_timestamp",
                "confidence_score": 0.95,
                "complexity_level": "low|medium|high",
                "estimated_effort": "effort_estimate"
            }}
        }}
    }}
    """
    
    MAPPING_VALIDATION_REALTIME = """
    Validate real-time data mapping during migration execution:
    
    SOURCE RECORD: {source_record}
    TARGET RECORD: {target_record}
    MAPPING RULES: {mapping_rules}
    VALIDATION CONTEXT: {validation_context}
    
    Please validate this mapping and provide:
    
    1. MAPPING VALIDATION:
       - Field-by-field validation results
       - Data type compatibility check
       - Business rule compliance
       - Data quality assessment
    
    2. ISSUE IDENTIFICATION:
       - Mapping errors or inconsistencies
       - Data quality problems
       - Business rule violations
       - Performance or efficiency issues
    
    3. CORRECTIVE ACTIONS:
       - Immediate fixes required
       - Mapping rule adjustments
       - Data quality improvements
       - Process optimizations
    
    4. MONITORING RECOMMENDATIONS:
       - Key metrics to track
       - Alert thresholds
       - Performance baselines
       - Continuous improvement areas
    
    Return validation results in this format:
    {{
        "success": true,
        "validation_results": {{
            "overall_status": "pass|fail|warning",
            "validation_score": 0.95,
            "field_validations": [
                {{
                    "source_field": "field_name",
                    "target_field": "target_field_name",
                    "status": "pass|fail|warning",
                    "validation_details": "detailed_validation_result",
                    "issues_found": ["issue1", "issue2"],
                    "recommendations": ["recommendation1", "recommendation2"]
                }}
            ],
            "business_rule_compliance": {{
                "rules_checked": ["rule1", "rule2"],
                "compliance_status": "compliant|non_compliant|partial",
                "violations": ["violation1", "violation2"],
                "severity": "high|medium|low"
            }},
            "data_quality_assessment": {{
                "quality_score": 0.90,
                "issues": ["issue1", "issue2"],
                "improvements": ["improvement1", "improvement2"]
            }},
            "corrective_actions": {{
                "immediate_actions": ["action1", "action2"],
                "long_term_improvements": ["improvement1", "improvement2"],
                "priority": "high|medium|low"
            }},
            "monitoring_recommendations": {{
                "key_metrics": ["metric1", "metric2"],
                "alert_thresholds": ["threshold1", "threshold2"],
                "performance_baselines": ["baseline1", "baseline2"]
            }},
            "metadata": {{
                "validation_timestamp": "ISO_timestamp",
                "validation_duration_ms": 150,
                "confidence_score": 0.95
            }}
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



