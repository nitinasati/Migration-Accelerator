"""
Mapping configuration loader and utilities.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

from config.settings import FieldMappingConfig, FieldMappingRule, ValidationRule, TransformationType


def load_mapping_config(file_path: str) -> FieldMappingConfig:
    """Load mapping configuration from YAML or JSON file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            data = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    return _parse_mapping_config(data)


def save_mapping_config(config: FieldMappingConfig, file_path: str):
    """Save mapping configuration to file."""
    path = Path(file_path)
    
    data = _serialize_mapping_config(config)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            yaml.dump(data, f, default_flow_style=False, indent=2)
        elif path.suffix.lower() == '.json':
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


def _parse_mapping_config(data: Dict[str, Any]) -> FieldMappingConfig:
    """Parse mapping configuration from dictionary."""
    rules = []
    
    for rule_data in data.get("rules", []):
        # Parse validation rules
        validation = None
        if "validation" in rule_data:
            validation_data = rule_data["validation"]
            validation = ValidationRule(
                required=validation_data.get("required", False),
                pattern=validation_data.get("pattern"),
                min_length=validation_data.get("min_length"),
                max_length=validation_data.get("max_length"),
                min_value=validation_data.get("min_value"),
                max_value=validation_data.get("max_value"),
                future_date=validation_data.get("future_date"),
                custom_rule=validation_data.get("custom_rule")
            )
        
        # Create field mapping rule
        rule = FieldMappingRule(
            source_field=rule_data["source_field"],
            target_field=rule_data["target_field"],
            transformation_type=TransformationType(rule_data["transformation_type"]),
            validation=validation,
            source_format=rule_data.get("source_format"),
            target_format=rule_data.get("target_format"),
            lookup_table=rule_data.get("lookup_table"),
            default_value=rule_data.get("default_value"),
            calculation_formula=rule_data.get("calculation_formula"),
            condition=rule_data.get("condition")
        )
        rules.append(rule)
    
    return FieldMappingConfig(
        source_format=data["source_format"],
        target_format=data["target_format"],
        record_type=data["record_type"],
        rules=rules,
        version=data.get("version", "1.0"),
        description=data.get("description")
    )


def _serialize_mapping_config(config: FieldMappingConfig) -> Dict[str, Any]:
    """Serialize mapping configuration to dictionary."""
    data = {
        "source_format": config.source_format,
        "target_format": config.target_format,
        "record_type": config.record_type,
        "rules": [],
        "version": config.version
    }
    
    if config.description:
        data["description"] = config.description
    
    for rule in config.rules:
        rule_data = {
            "source_field": rule.source_field,
            "target_field": rule.target_field,
            "transformation_type": rule.transformation_type.value
        }
        
        if rule.validation:
            rule_data["validation"] = {
                "required": rule.validation.required,
                "pattern": rule.validation.pattern,
                "min_length": rule.validation.min_length,
                "max_length": rule.validation.max_length,
                "min_value": rule.validation.min_value,
                "max_value": rule.validation.max_value,
                "future_date": rule.validation.future_date,
                "custom_rule": rule.validation.custom_rule
            }
        
        if rule.source_format:
            rule_data["source_format"] = rule.source_format
        
        if rule.target_format:
            rule_data["target_format"] = rule.target_format
        
        if rule.lookup_table:
            rule_data["lookup_table"] = rule.lookup_table
        
        if rule.default_value is not None:
            rule_data["default_value"] = rule.default_value
        
        if rule.calculation_formula:
            rule_data["calculation_formula"] = rule.calculation_formula
        
        if rule.condition:
            rule_data["condition"] = rule.condition
        
        data["rules"].append(rule_data)
    
    return data


def create_default_mapping_config(record_type: str, source_format: str = "csv", target_format: str = "json") -> FieldMappingConfig:
    """Create a default mapping configuration."""
    rules = [
        FieldMappingRule(
            source_field="id",
            target_field="recordId",
            transformation_type=TransformationType.DIRECT,
            validation=ValidationRule(required=True)
        ),
        FieldMappingRule(
            source_field="created_date",
            target_field="createdAt",
            transformation_type=TransformationType.DATE_FORMAT,
            source_format="%Y-%m-%d",
            target_format="ISO8601",
            validation=ValidationRule(required=False, future_date=False)
        )
    ]
    
    return FieldMappingConfig(
        source_format=source_format,
        target_format=target_format,
        record_type=record_type,
        rules=rules,
        version="1.0",
        description=f"Default mapping for {record_type} records"
    )


def validate_mapping_config(config: FieldMappingConfig) -> list:
    """Validate mapping configuration and return list of errors."""
    errors = []
    
    # Check required fields
    if not config.source_format:
        errors.append("source_format is required")
    
    if not config.target_format:
        errors.append("target_format is required")
    
    if not config.record_type:
        errors.append("record_type is required")
    
    if not config.rules:
        errors.append("At least one mapping rule is required")
    
    # Validate rules
    for i, rule in enumerate(config.rules):
        if not rule.source_field:
            errors.append(f"Rule {i+1}: source_field is required")
        
        if not rule.target_field:
            errors.append(f"Rule {i+1}: target_field is required")
        
        # Validate transformation-specific fields
        if rule.transformation_type == TransformationType.DATE_FORMAT:
            if not rule.source_format:
                errors.append(f"Rule {i+1}: source_format is required for date_format transformation")
            if not rule.target_format:
                errors.append(f"Rule {i+1}: target_format is required for date_format transformation")
        
        elif rule.transformation_type == TransformationType.LOOKUP:
            if not rule.lookup_table:
                errors.append(f"Rule {i+1}: lookup_table is required for lookup transformation")
        
        elif rule.transformation_type == TransformationType.CALCULATED:
            if not rule.calculation_formula:
                errors.append(f"Rule {i+1}: calculation_formula is required for calculated transformation")
        
        elif rule.transformation_type == TransformationType.CONDITIONAL:
            if not rule.condition:
                errors.append(f"Rule {i+1}: condition is required for conditional transformation")
    
    return errors


def list_mapping_files(directory: str = "config/mappings") -> list:
    """List all mapping configuration files in a directory."""
    path = Path(directory)
    
    if not path.exists():
        return []
    
    mapping_files = []
    for file_path in path.glob("*.yaml"):
        mapping_files.append(str(file_path))
    for file_path in path.glob("*.yml"):
        mapping_files.append(str(file_path))
    for file_path in path.glob("*.json"):
        mapping_files.append(str(file_path))
    
    return sorted(mapping_files)
