"""
Mapping configuration management for the Migration-Accelerators platform.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import structlog

from .settings import FieldMappingConfig, FieldMappingRule, ValidationRule, TransformationType, FileFormat, RecordType

logger = structlog.get_logger()


def load_mapping_config(file_path: str) -> FieldMappingConfig:
    """
    Load mapping configuration from YAML file.
    
    Args:
        file_path: Path to the mapping configuration file
        
    Returns:
        FieldMappingConfig: Loaded mapping configuration
        
    Raises:
        FileNotFoundError: If the mapping file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
        ValueError: If the configuration is invalid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        logger.info("Loading mapping configuration", file_path=file_path)
        
        # Convert string enums to proper enum values
        if 'source_format' in data:
            data['source_format'] = FileFormat(data['source_format'])
        if 'target_format' in data:
            data['target_format'] = FileFormat(data['target_format'])
        if 'record_type' in data:
            data['record_type'] = RecordType(data['record_type'])
        
        # Convert rules
        if 'rules' in data:
            converted_rules = []
            for rule_data in data['rules']:
                # Convert transformation type
                if 'transformation_type' in rule_data:
                    rule_data['transformation_type'] = TransformationType(rule_data['transformation_type'])
                
                # Convert validation rule if present
                if 'validation' in rule_data and rule_data['validation']:
                    rule_data['validation'] = ValidationRule(**rule_data['validation'])
                
                converted_rules.append(FieldMappingRule(**rule_data))
            data['rules'] = converted_rules
        
        config = FieldMappingConfig(**data)
        
        logger.info("Mapping configuration loaded successfully", 
                   record_type=config.record_type.value,
                   rule_count=len(config.rules))
        
        return config
        
    except FileNotFoundError:
        logger.error("Mapping file not found", file_path=file_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Invalid YAML in mapping file", file_path=file_path, error=str(e))
        raise
    except Exception as e:
        logger.error("Error loading mapping configuration", file_path=file_path, error=str(e))
        raise ValueError(f"Invalid mapping configuration: {e}")


def save_mapping_config(config: FieldMappingConfig, file_path: str) -> None:
    """
    Save mapping configuration to YAML file.
    
    Args:
        config: Mapping configuration to save
        file_path: Path where to save the configuration
    """
    try:
        # Convert to dictionary for YAML serialization
        data = config.model_dump()
        
        # Convert enums to strings
        data['source_format'] = data['source_format'].value
        data['target_format'] = data['target_format'].value
        data['record_type'] = data['record_type'].value
        
        # Convert rules
        for rule in data['rules']:
            rule['transformation_type'] = rule['transformation_type'].value
            if rule.get('validation'):
                rule['validation'] = rule['validation']
        
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        
        logger.info("Mapping configuration saved", file_path=file_path)
        
    except Exception as e:
        logger.error("Error saving mapping configuration", file_path=file_path, error=str(e))
        raise


def create_default_mapping_config(record_type: str) -> FieldMappingConfig:
    """
    Create a default mapping configuration for a record type.
    
    Args:
        record_type: Type of record to create mapping for
        
    Returns:
        FieldMappingConfig: Default mapping configuration
    """
    logger.info("Creating default mapping configuration", record_type=record_type)
    
    if record_type.lower() == "disability":
        return _create_disability_mapping()
    elif record_type.lower() == "absence":
        return _create_absence_mapping()
    elif record_type.lower() == "group_policy":
        return _create_group_policy_mapping()
    else:
        return _create_generic_mapping(record_type)


def _create_disability_mapping() -> FieldMappingConfig:
    """Create default disability mapping configuration."""
    rules = [
        FieldMappingRule(
            source_field="policy_number",
            target_field="policyId",
            transformation_type=TransformationType.DIRECT,
            validation=ValidationRule(required=True, pattern="^[A-Z0-9]{6,12}$")
        ),
        FieldMappingRule(
            source_field="employee_id",
            target_field="employeeId",
            transformation_type=TransformationType.DIRECT,
            validation=ValidationRule(required=True, pattern="^EMP[0-9]{6}$")
        ),
        FieldMappingRule(
            source_field="effective_date",
            target_field="effectiveDate",
            transformation_type=TransformationType.DATE_FORMAT,
            source_format="%Y-%m-%d",
            target_format="ISO8601",
            validation=ValidationRule(required=True, future_date=False)
        ),
        FieldMappingRule(
            source_field="benefit_amount",
            target_field="benefitAmount",
            transformation_type=TransformationType.DIRECT,
            validation=ValidationRule(required=False, min_value=0.0, max_value=1000000.0)
        ),
        FieldMappingRule(
            source_field="status",
            target_field="policyStatus",
            transformation_type=TransformationType.LOOKUP,
            lookup_table={
                "active": "ACTIVE",
                "inactive": "INACTIVE",
                "pending": "PENDING",
                "cancelled": "CANCELLED"
            },
            validation=ValidationRule(required=True)
        )
    ]
    
    return FieldMappingConfig(
        source_format=FileFormat.CSV,
        target_format=FileFormat.JSON,
        record_type=RecordType.DISABILITY,
        version="1.0",
        description="Default disability insurance mapping",
        rules=rules
    )


def _create_absence_mapping() -> FieldMappingConfig:
    """Create default absence mapping configuration."""
    rules = [
        FieldMappingRule(
            source_field="employee_id",
            target_field="employeeId",
            transformation_type=TransformationType.DIRECT,
            validation=ValidationRule(required=True, pattern="^EMP[0-9]{6}$")
        ),
        FieldMappingRule(
            source_field="absence_start_date",
            target_field="startDate",
            transformation_type=TransformationType.DATE_FORMAT,
            source_format="%Y-%m-%d",
            target_format="ISO8601",
            validation=ValidationRule(required=True)
        ),
        FieldMappingRule(
            source_field="absence_type",
            target_field="absenceType",
            transformation_type=TransformationType.LOOKUP,
            lookup_table={
                "sick": "SICK_LEAVE",
                "vacation": "VACATION",
                "personal": "PERSONAL_LEAVE",
                "maternity": "MATERNITY_LEAVE"
            },
            validation=ValidationRule(required=True)
        ),
        FieldMappingRule(
            source_field="duration_days",
            target_field="durationDays",
            transformation_type=TransformationType.DIRECT,
            validation=ValidationRule(required=True, min_value=1, max_value=365)
        )
    ]
    
    return FieldMappingConfig(
        source_format=FileFormat.CSV,
        target_format=FileFormat.JSON,
        record_type=RecordType.ABSENCE,
        version="1.0",
        description="Default absence management mapping",
        rules=rules
    )


def _create_group_policy_mapping() -> FieldMappingConfig:
    """Create default group policy mapping configuration."""
    rules = [
        FieldMappingRule(
            source_field="group_policy_number",
            target_field="groupPolicyId",
            transformation_type=TransformationType.DIRECT,
            validation=ValidationRule(required=True, pattern="^GP[0-9]{8}$")
        ),
        FieldMappingRule(
            source_field="employer_id",
            target_field="employerId",
            transformation_type=TransformationType.DIRECT,
            validation=ValidationRule(required=True)
        ),
        FieldMappingRule(
            source_field="effective_date",
            target_field="effectiveDate",
            transformation_type=TransformationType.DATE_FORMAT,
            source_format="%Y-%m-%d",
            target_format="ISO8601",
            validation=ValidationRule(required=True)
        ),
        FieldMappingRule(
            source_field="coverage_type",
            target_field="coverageType",
            transformation_type=TransformationType.LOOKUP,
            lookup_table={
                "std": "SHORT_TERM_DISABILITY",
                "ltd": "LONG_TERM_DISABILITY",
                "both": "BOTH"
            },
            validation=ValidationRule(required=True)
        )
    ]
    
    return FieldMappingConfig(
        source_format=FileFormat.CSV,
        target_format=FileFormat.JSON,
        record_type=RecordType.GROUP_POLICY,
        version="1.0",
        description="Default group policy mapping",
        rules=rules
    )


def _create_generic_mapping(record_type: str) -> FieldMappingConfig:
    """Create generic mapping configuration."""
    rules = [
        FieldMappingRule(
            source_field="id",
            target_field="id",
            transformation_type=TransformationType.DIRECT,
            validation=ValidationRule(required=True)
        ),
        FieldMappingRule(
            source_field="name",
            target_field="name",
            transformation_type=TransformationType.DIRECT,
            validation=ValidationRule(required=True, min_length=1)
        ),
        FieldMappingRule(
            source_field="created_date",
            target_field="createdDate",
            transformation_type=TransformationType.DATE_FORMAT,
            source_format="%Y-%m-%d",
            target_format="ISO8601",
            validation=ValidationRule(required=False)
        )
    ]
    
    return FieldMappingConfig(
        source_format=FileFormat.CSV,
        target_format=FileFormat.JSON,
        record_type=RecordType(record_type.lower()),
        version="1.0",
        description=f"Generic mapping for {record_type}",
        rules=rules
    )


def validate_mapping_config(config: FieldMappingConfig) -> List[str]:
    """
    Validate mapping configuration and return list of errors.
    
    Args:
        config: Mapping configuration to validate
        
    Returns:
        List[str]: List of validation errors (empty if valid)
    """
    errors = []
    
    # Check for duplicate source fields
    source_fields = [rule.source_field for rule in config.rules]
    if len(source_fields) != len(set(source_fields)):
        errors.append("Duplicate source fields found in mapping rules")
    
    # Check for duplicate target fields
    target_fields = [rule.target_field for rule in config.rules]
    if len(target_fields) != len(set(target_fields)):
        errors.append("Duplicate target fields found in mapping rules")
    
    # Validate individual rules
    for i, rule in enumerate(config.rules):
        if not rule.source_field.strip():
            errors.append(f"Rule {i}: source_field is required")
        
        if not rule.target_field.strip():
            errors.append(f"Rule {i}: target_field is required")
        
        # Validate transformation-specific requirements
        if rule.transformation_type == TransformationType.DATE_FORMAT:
            if not rule.source_format:
                errors.append(f"Rule {i}: source_format required for date_format transformation")
            if not rule.target_format:
                errors.append(f"Rule {i}: target_format required for date_format transformation")
        
        elif rule.transformation_type == TransformationType.LOOKUP:
            if not rule.lookup_table:
                errors.append(f"Rule {i}: lookup_table required for lookup transformation")
        
        elif rule.transformation_type == TransformationType.CALCULATED:
            if not rule.calculation_formula:
                errors.append(f"Rule {i}: calculation_formula required for calculated transformation")
        
        elif rule.transformation_type == TransformationType.CONDITIONAL:
            if not rule.condition:
                errors.append(f"Rule {i}: condition required for conditional transformation")
    
    return errors


def get_mapping_by_source_field(config: FieldMappingConfig, source_field: str) -> Optional[FieldMappingRule]:
    """
    Get mapping rule by source field name.
    
    Args:
        config: Mapping configuration
        source_field: Source field name to find
        
    Returns:
        FieldMappingRule or None if not found
    """
    for rule in config.rules:
        if rule.source_field == source_field:
            return rule
    return None


def get_mapping_by_target_field(config: FieldMappingConfig, target_field: str) -> Optional[FieldMappingRule]:
    """
    Get mapping rule by target field name.
    
    Args:
        config: Mapping configuration
        target_field: Target field name to find
        
    Returns:
        FieldMappingRule or None if not found
    """
    for rule in config.rules:
        if rule.target_field == target_field:
            return rule
    return None
