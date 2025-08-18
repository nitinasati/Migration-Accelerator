"""
Prompt templates for the migration platform.
"""

from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate


class MigrationPrompts:
    """Prompt templates for migration tasks."""
    
    # File Reading Prompts
    FILE_ANALYSIS_PROMPT = PromptTemplate(
        input_variables=["file_content", "file_format"],
        template="""
You are an expert in analyzing insurance data files. Analyze the following {file_format} file content and provide insights:

File Content:
{file_content}

Please provide analysis in JSON format:
{{
    "file_type": "csv|excel|json|xml|fixed_width",
    "record_count": 123,
    "field_names": ["field1", "field2", "field3"],
    "data_quality_score": 0.85,
    "issues": ["issue1", "issue2"],
    "recommendations": ["recommendation1", "recommendation2"]
}}
"""
    )
    
    # Validation Prompts
    DATA_VALIDATION_PROMPT = PromptTemplate(
        input_variables=["record_data", "validation_rules", "record_type"],
        template="""
You are an expert insurance data validator. Validate the following {record_type} record against the specified rules:

Record Data:
{record_data}

Validation Rules:
{validation_rules}

Please provide validation results in JSON format:
{{
    "is_valid": true/false,
    "errors": [
        {{
            "field": "field_name",
            "error": "error_description",
            "severity": "error|warning|info"
        }}
    ],
    "warnings": [
        {{
            "field": "field_name",
            "warning": "warning_description",
            "severity": "warning|info"
        }}
    ],
    "validation_score": 0.85,
    "recommendations": ["recommendation1", "recommendation2"]
}}
"""
    )
    
    BUSINESS_RULE_VALIDATION_PROMPT = PromptTemplate(
        input_variables=["record_data", "business_rules", "record_type"],
        template="""
You are an expert in insurance business rules. Validate the following {record_type} record against business rules:

Record Data:
{record_data}

Business Rules:
{business_rules}

Please provide business rule validation results in JSON format:
{{
    "compliance_score": 0.85,
    "business_rule_violations": [
        {{
            "rule": "rule_name",
            "violation": "violation_description",
            "severity": "critical|high|medium|low"
        }}
    ],
    "risk_assessment": "low|medium|high",
    "recommendations": ["recommendation1", "recommendation2"]
}}
"""
    )
    
    # Mapping Prompts
    FIELD_MAPPING_PROMPT = PromptTemplate(
        input_variables=["source_fields", "target_fields", "source_format", "target_format"],
        template="""
You are an expert in data field mapping. Suggest mappings between source and target fields:

Source Fields ({source_format}):
{source_fields}

Target Fields ({target_format}):
{target_fields}

Please suggest field mappings in JSON format:
{{
    "mappings": [
        {{
            "source_field": "source_field_name",
            "target_field": "target_field_name",
            "transformation_type": "direct|date_format|lookup|calculated|conditional",
            "confidence": 0.95,
            "reasoning": "explanation for this mapping"
        }}
    ],
    "unmapped_source_fields": ["field1", "field2"],
    "unmapped_target_fields": ["field1", "field2"],
    "recommendations": ["recommendation1", "recommendation2"]
}}
"""
    )
    
    # Transformation Prompts
    DATA_TRANSFORMATION_PROMPT = PromptTemplate(
        input_variables=["source_data", "mapping_rules", "record_type"],
        template="""
You are an expert in data transformation. Transform the following {record_type} record according to the mapping rules:

Source Data:
{source_data}

Mapping Rules:
{mapping_rules}

Please provide the transformed data in JSON format. Apply all transformations and validations as specified in the mapping rules.
"""
    )
    
    # API Integration Prompts
    API_PAYLOAD_GENERATION_PROMPT = PromptTemplate(
        input_variables=["transformed_data", "api_spec", "operation"],
        template="""
You are an expert in API integration. Generate the appropriate payload for the {operation} operation:

Transformed Data:
{transformed_data}

API Specification:
{api_spec}

Please generate the API payload in JSON format that matches the API specification.
"""
    )
    
    # Error Handling Prompts
    ERROR_ANALYSIS_PROMPT = PromptTemplate(
        input_variables=["error_message", "context", "record_data"],
        template="""
You are an expert in error analysis. Analyze the following error and provide insights:

Error Message:
{error_message}

Context:
{context}

Record Data:
{record_data}

Please provide error analysis in JSON format:
{{
    "error_type": "validation|transformation|api|system",
    "severity": "critical|high|medium|low",
    "root_cause": "explanation of the root cause",
    "suggested_fixes": ["fix1", "fix2"],
    "prevention_measures": ["measure1", "measure2"]
}}
"""
    )
    
    # Quality Assessment Prompts
    DATA_QUALITY_ASSESSMENT_PROMPT = PromptTemplate(
        input_variables=["data_sample", "record_type", "quality_criteria"],
        template="""
You are an expert in data quality assessment. Assess the quality of the following {record_type} data:

Data Sample:
{data_sample}

Quality Criteria:
{quality_criteria}

Please provide quality assessment in JSON format:
{{
    "overall_quality_score": 0.85,
    "completeness_score": 0.90,
    "accuracy_score": 0.80,
    "consistency_score": 0.85,
    "timeliness_score": 0.90,
    "quality_issues": [
        {{
            "issue_type": "completeness|accuracy|consistency|timeliness",
            "description": "issue description",
            "impact": "high|medium|low"
        }}
    ],
    "recommendations": ["recommendation1", "recommendation2"]
}}
"""
    )
    
    # Business Logic Prompts
    BUSINESS_LOGIC_APPLICATION_PROMPT = PromptTemplate(
        input_variables=["record_data", "business_rules", "record_type"],
        template="""
You are an expert in insurance business logic. Apply business rules to the following {record_type} record:

Record Data:
{record_data}

Business Rules:
{business_rules}

Please apply the business rules and provide the enhanced record in JSON format. Include any calculated fields, derived values, or business logic transformations.
"""
    )
    
    # Compliance Checking Prompts
    COMPLIANCE_CHECK_PROMPT = PromptTemplate(
        input_variables=["record_data", "compliance_rules", "record_type"],
        template="""
You are an expert in insurance compliance. Check the following {record_type} record for compliance:

Record Data:
{record_data}

Compliance Rules:
{compliance_rules}

Please provide compliance check results in JSON format:
{{
    "compliance_score": 0.85,
    "compliant": true/false,
    "compliance_issues": [
        {{
            "rule": "compliance_rule_name",
            "issue": "compliance_issue_description",
            "severity": "critical|high|medium|low"
        }}
    ],
    "required_actions": ["action1", "action2"],
    "certification": "certified|needs_review|non_compliant"
}}
"""
    )


class A2APrompts:
    """Prompt templates for A2A (Agent-to-Agent) communication."""
    
    AGENT_COORDINATION_PROMPT = PromptTemplate(
        input_variables=["agent_role", "task_description", "context"],
        template="""
You are an {agent_role} agent in a multi-agent migration system. Your task is:

{task_description}

Context:
{context}

Please coordinate with other agents and provide your response in JSON format:
{{
    "status": "ready|processing|completed|failed",
    "result": "your_result_here",
    "next_agent": "next_agent_name",
    "dependencies": ["dependency1", "dependency2"],
    "estimated_duration": "estimated_time_in_seconds"
}}
"""
    )
    
    AGENT_HANDOFF_PROMPT = PromptTemplate(
        input_variables=["current_agent", "next_agent", "handoff_data", "task_context"],
        template="""
Agent handoff from {current_agent} to {next_agent}:

Task Context:
{task_context}

Handoff Data:
{handoff_data}

Please acknowledge the handoff and provide your plan in JSON format:
{{
    "acknowledged": true/false,
    "plan": "your_execution_plan",
    "estimated_duration": "estimated_time_in_seconds",
    "dependencies": ["dependency1", "dependency2"],
    "risks": ["risk1", "risk2"]
}}
"""
    )


class MCPPrompts:
    """Prompt templates for MCP (Model Context Protocol) interactions."""
    
    MCP_TOOL_SELECTION_PROMPT = PromptTemplate(
        input_variables=["available_tools", "task_description"],
        template="""
You need to select the appropriate MCP tool for the following task:

Task Description:
{task_description}

Available Tools:
{available_tools}

Please select the best tool and provide your reasoning in JSON format:
{{
    "selected_tool": "tool_name",
    "reasoning": "explanation for tool selection",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "alternatives": ["alternative_tool1", "alternative_tool2"]
}}
"""
    )
    
    MCP_RESPONSE_PARSING_PROMPT = PromptTemplate(
        input_variables=["mcp_response", "expected_format"],
        template="""
Parse the following MCP response and convert it to the expected format:

MCP Response:
{mcp_response}

Expected Format:
{expected_format}

Please provide the parsed response in the expected format.
"""
    )


def get_prompt_by_name(prompt_name: str) -> PromptTemplate:
    """Get prompt template by name."""
    prompts = {
        # Migration prompts
        "file_analysis": MigrationPrompts.FILE_ANALYSIS_PROMPT,
        "data_validation": MigrationPrompts.DATA_VALIDATION_PROMPT,
        "business_rule_validation": MigrationPrompts.BUSINESS_RULE_VALIDATION_PROMPT,
        "field_mapping": MigrationPrompts.FIELD_MAPPING_PROMPT,
        "data_transformation": MigrationPrompts.DATA_TRANSFORMATION_PROMPT,
        "api_payload_generation": MigrationPrompts.API_PAYLOAD_GENERATION_PROMPT,
        "error_analysis": MigrationPrompts.ERROR_ANALYSIS_PROMPT,
        "data_quality_assessment": MigrationPrompts.DATA_QUALITY_ASSESSMENT_PROMPT,
        "business_logic_application": MigrationPrompts.BUSINESS_LOGIC_APPLICATION_PROMPT,
        "compliance_check": MigrationPrompts.COMPLIANCE_CHECK_PROMPT,
        
        # A2A prompts
        "agent_coordination": A2APrompts.AGENT_COORDINATION_PROMPT,
        "agent_handoff": A2APrompts.AGENT_HANDOFF_PROMPT,
        
        # MCP prompts
        "mcp_tool_selection": MCPPrompts.MCP_TOOL_SELECTION_PROMPT,
        "mcp_response_parsing": MCPPrompts.MCP_RESPONSE_PARSING_PROMPT,
    }
    
    if prompt_name not in prompts:
        raise ValueError(f"Unknown prompt name: {prompt_name}")
    
    return prompts[prompt_name]
