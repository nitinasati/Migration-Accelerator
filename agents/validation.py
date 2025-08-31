"""
Validation Agent for the Migration-Accelerators platform.
"""

import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union
import structlog

from .base_agent import BaseAgent, AgentResult
from config.settings import LLMConfig, MCPConfig, ValidationRule, ValidationSeverity
from llm.providers import BaseLLMProvider, LLMProviderFactory
from llm.prompts import get_prompt, get_system_prompt, get_validation_schema


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(
        self,
        field_name: str,
        value: Any,
        is_valid: bool,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        message: str = "",
        suggested_value: Optional[Any] = None
    ):
        self.field_name = field_name
        self.value = value
        self.is_valid = is_valid
        self.severity = severity
        self.message = message
        self.suggested_value = suggested_value


class ValidationAgent(BaseAgent):
    """
    Validation Agent for data integrity and business rule validation.
    
    This agent handles:
    - Field-level validation
    - Business rule validation
    - Data type validation
    - Format validation
    - Cross-field validation
    - LLM-powered intelligent validation
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        mcp_config: Optional[MCPConfig] = None
    ):
        super().__init__("validation", llm_config, mcp_config)
        self.llm_provider: Optional[BaseLLMProvider] = None
        self.validation_cache: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize the validation agent."""
        await super().start()
        
        if self.llm_config:
            try:
                self.llm_provider = LLMProviderFactory.create(self.llm_config, self.agent_name)
                await self.llm_provider.initialize()
                self.logger.info("LLM provider initialized for validation agent")
            except ImportError as e:
                self.logger.warning(f"LLM provider not available: {e}")
                self.logger.info("Running without LLM enhancements")
                self.llm_provider = None
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM provider: {e}")
                self.logger.info("Running without LLM enhancements")
                self.llm_provider = None
        
        self.logger.info("Validation agent initialized")
    
    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Process validation request.
        
        Args:
            data: Data to validate (record or list of records)
            context: Validation context (rules, schema, record_type, etc.)
            
        Returns:
            AgentResult: Validation result
        """
        try:
            self.logger.info("Starting validation process", data_type=type(data).__name__)
            
            # Validate input
            if not await self.validate_input(data):
                return AgentResult(
                    success=False,
                    errors=["Invalid input data for validation"]
                )
            
            # Get validation context
            validation_rules = context.get("validation_rules", []) if context else []
            record_type = context.get("record_type", "unknown") if context else "unknown"
            schema = context.get("schema") if context else None
            
            # If no schema provided, get default schema for record type
            if not schema:
                schema = get_validation_schema(record_type)
            
            # Validate data
            if isinstance(data, list):
                validation_results = await self._validate_batch(data, validation_rules, schema, record_type)
            else:
                validation_results = await self._validate_record(data, validation_rules, schema, record_type)
            
            # Process validation results
            errors = []
            warnings = []
            valid_records = []
            invalid_records = []
            
            for result in validation_results:
                if result.severity == ValidationSeverity.ERROR:
                    errors.append(result.message)
                    invalid_records.append(result)
                elif result.severity == ValidationSeverity.WARNING:
                    warnings.append(result.message)
                else:
                    valid_records.append(result)
            
            success = len(errors) == 0
            
            self.logger.info(
                "Validation completed",
                success=success,
                total_records=len(data) if isinstance(data, list) else 1,
                errors_count=len(errors),
                warnings_count=len(warnings)
            )
            
            return AgentResult(
                success=success,
                data={
                    "validation_results": validation_results,
                    "valid_records": valid_records,
                    "invalid_records": invalid_records,
                    "summary": {
                        "total_records": len(data) if isinstance(data, list) else 1,
                        "valid_count": len(valid_records),
                        "invalid_count": len(invalid_records),
                        "error_count": len(errors),
                        "warning_count": len(warnings)
                    }
                },
                errors=errors,
                warnings=warnings,
                metadata={
                    "record_type": record_type,
                    "validation_rules_count": len(validation_rules),
                    "agent": self.agent_name
                }
            )
            
        except Exception as e:
            return await self.handle_error(e, {"data_type": type(data).__name__, "context": context})
    
    async def _validate_batch(
        self,
        records: List[Dict[str, Any]],
        validation_rules: List[ValidationRule],
        schema: Dict[str, Any],
        record_type: str
    ) -> List[ValidationResult]:
        """Validate a batch of records with optimized LLM calls."""
        validation_results = []
        
        # First, do non-LLM validation for all records
        for i, record in enumerate(records):
            # Schema validation
            schema_results = await self._validate_against_schema(record, schema)
            validation_results.extend(schema_results)
            
            # Business rule validation
            for rule in validation_rules:
                rule_results = await self._validate_rule(record, rule)
                validation_results.extend(rule_results)
            
            # Cross-field validation
            cross_field_results = await self._validate_cross_fields(record, record_type)
            validation_results.extend(cross_field_results)
            
            # Add record index to results
            for result in schema_results + rule_results + cross_field_results:
                result.message = f"Record {i + 1}: {result.message}"
        
        # Then, do batch LLM validation for all records at once
        if self.llm_provider and records:
            llm_results = await self._validate_batch_with_llm(records, record_type)
            validation_results.extend(llm_results)
        
        return validation_results
    
    async def _validate_record(
        self,
        record: Dict[str, Any],
        validation_rules: List[ValidationRule],
        schema: Dict[str, Any],
        record_type: str
    ) -> List[ValidationResult]:
        """Validate a single record."""
        validation_results = []
        
        # Validate against schema
        schema_results = await self._validate_against_schema(record, schema)
        validation_results.extend(schema_results)
        
        # Validate against business rules
        for rule in validation_rules:
            rule_results = await self._validate_rule(record, rule)
            validation_results.extend(rule_results)
        
        # Cross-field validation
        cross_field_results = await self._validate_cross_fields(record, record_type)
        validation_results.extend(cross_field_results)
        
        # LLM-powered intelligent validation
        if self.llm_provider:
            llm_results = await self._validate_with_llm(record, record_type)
            validation_results.extend(llm_results)
        
        return validation_results
    
    async def _validate_against_schema(
        self,
        record: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate record against JSON schema."""
        results = []
        
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in record or record[field] is None or record[field] == "":
                results.append(ValidationResult(
                    field_name=field,
                    value=record.get(field),
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required field '{field}' is missing or empty"
                ))
        
        # Validate field properties
        properties = schema.get("properties", {})
        for field, value in record.items():
            if field in properties:
                field_schema = properties[field]
                field_results = await self._validate_field_against_schema(field, value, field_schema)
                results.extend(field_results)
        
        return results
    
    async def _validate_field_against_schema(
        self,
        field_name: str,
        value: Any,
        field_schema: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate a field against its schema definition."""
        results = []
        
        # Type validation
        expected_type = field_schema.get("type")
        if expected_type:
            type_valid = await self._validate_type(value, expected_type)
            if not type_valid:
                results.append(ValidationResult(
                    field_name=field_name,
                    value=value,
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field '{field_name}' should be {expected_type}, got {type(value).__name__}"
                ))
        
        # Pattern validation
        pattern = field_schema.get("pattern")
        if pattern and isinstance(value, str):
            if not re.match(pattern, value):
                results.append(ValidationResult(
                    field_name=field_name,
                    value=value,
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field '{field_name}' does not match pattern '{pattern}'"
                ))
        
        # Enum validation
        enum_values = field_schema.get("enum")
        if enum_values and value not in enum_values:
            results.append(ValidationResult(
                field_name=field_name,
                value=value,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Field '{field_name}' value '{value}' is not in allowed values: {enum_values}"
            ))
        
        # Range validation
        if isinstance(value, (int, float)):
            min_val = field_schema.get("minimum")
            max_val = field_schema.get("maximum")
            if min_val is not None and value < min_val:
                results.append(ValidationResult(
                    field_name=field_name,
                    value=value,
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field '{field_name}' value {value} is below minimum {min_val}"
                ))
            if max_val is not None and value > max_val:
                results.append(ValidationResult(
                    field_name=field_name,
                    value=value,
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field '{field_name}' value {value} is above maximum {max_val}"
                ))
        
        # Length validation
        if isinstance(value, str):
            min_length = field_schema.get("minLength")
            max_length = field_schema.get("maxLength")
            if min_length is not None and len(value) < min_length:
                results.append(ValidationResult(
                    field_name=field_name,
                    value=value,
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field '{field_name}' length {len(value)} is below minimum {min_length}"
                ))
            if max_length is not None and len(value) > max_length:
                results.append(ValidationResult(
                    field_name=field_name,
                    value=value,
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field '{field_name}' length {len(value)} is above maximum {max_length}"
                ))
        
        return results
    
    async def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate if value matches expected type."""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            # Allow both actual numbers and string representations of numbers
            if isinstance(value, (int, float)):
                return True
            elif isinstance(value, str):
                try:
                    float(value)
                    return True
                except (ValueError, TypeError):
                    return False
            return False
        elif expected_type == "integer":
            # Allow both actual integers and string representations of integers
            if isinstance(value, int):
                return True
            elif isinstance(value, str):
                try:
                    int(value)
                    return True
                except (ValueError, TypeError):
                    return False
            return False
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        elif expected_type == "null":
            return value is None
        else:
            return True  # Unknown type, assume valid
    
    async def _validate_rule(
        self,
        record: Dict[str, Any],
        rule: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate against a specific validation rule."""
        results = []
        
        # Extract field name and validation rule
        field_name = rule.get("field_name")
        validation_rule = rule.get("validation_rule")
        
        if not field_name or not validation_rule:
            return results
        
        # This is a simplified implementation
        # In a real system, you would have more sophisticated rule evaluation
        
        if validation_rule.required:
            # Check if field is present and not empty
            field_value = record.get(field_name)
            if field_value is None or field_value == "":
                results.append(ValidationResult(
                    field_name=field_name,
                    value=field_value,
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required field '{field_name}' is missing"
                ))
        
        return results
    
    async def _validate_cross_fields(
        self,
        record: Dict[str, Any],
        record_type: str
    ) -> List[ValidationResult]:
        """Validate cross-field relationships."""
        results = []
        
        # Insurance-specific cross-field validations
        if record_type.lower() == "disability":
            results.extend(await self._validate_disability_cross_fields(record))
        elif record_type.lower() == "absence":
            results.extend(await self._validate_absence_cross_fields(record))
        elif record_type.lower() == "group_policy":
            results.extend(await self._validate_group_policy_cross_fields(record))
        
        return results
    
    async def _validate_disability_cross_fields(self, record: Dict[str, Any]) -> List[ValidationResult]:
        """Validate disability-specific cross-field rules."""
        results = []
        
        # Check if effective date is before termination date
        effective_date = record.get("effective_date")
        termination_date = record.get("termination_date")
        
        if effective_date and termination_date:
            try:
                eff_date = datetime.strptime(effective_date, "%Y-%m-%d").date()
                term_date = datetime.strptime(termination_date, "%Y-%m-%d").date()
                
                if eff_date >= term_date:
                    results.append(ValidationResult(
                        field_name="effective_date",
                        value=effective_date,
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message="Effective date must be before termination date"
                    ))
            except ValueError:
                # Date parsing errors are handled by individual field validation
                pass
        
        # Check benefit amount vs premium amount
        benefit_amount = record.get("benefit_amount")
        premium_amount = record.get("premium_amount")
        
        if benefit_amount and premium_amount:
            try:
                benefit = float(benefit_amount)
                premium = float(premium_amount)
                
                if benefit <= premium:
                    results.append(ValidationResult(
                        field_name="benefit_amount",
                        value=benefit_amount,
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        message="Benefit amount should be greater than premium amount"
                    ))
            except (ValueError, TypeError):
                # Type conversion errors are handled by individual field validation
                pass
        
        return results
    
    async def _validate_absence_cross_fields(self, record: Dict[str, Any]) -> List[ValidationResult]:
        """Validate absence-specific cross-field rules."""
        results = []
        
        # Check if absence start date is not in the future
        start_date = record.get("absence_start_date")
        if start_date:
            try:
                abs_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                today = date.today()
                
                if abs_date > today:
                    results.append(ValidationResult(
                        field_name="absence_start_date",
                        value=start_date,
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        message="Absence start date is in the future"
                    ))
            except ValueError:
                pass
        
        return results
    
    async def _validate_group_policy_cross_fields(self, record: Dict[str, Any]) -> List[ValidationResult]:
        """Validate group policy-specific cross-field rules."""
        results = []
        
        # Check if group policy number format matches employer ID
        group_policy = record.get("group_policy_number", "")
        employer_id = record.get("employer_id", "")
        
        if group_policy and employer_id:
            # Simple validation: group policy should contain employer ID
            if employer_id not in group_policy:
                results.append(ValidationResult(
                    field_name="group_policy_number",
                    value=group_policy,
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message="Group policy number should contain employer ID"
                ))
        
        return results
    
    async def _validate_with_llm(
        self,
        record: Dict[str, Any],
        record_type: str
    ) -> List[ValidationResult]:
        """Use LLM for intelligent validation of a single record."""
        # This method is kept for backward compatibility but should not be used
        # for batch processing. Use _validate_batch_with_llm instead.
        return await self._validate_batch_with_llm([record], record_type)
    
    async def _validate_batch_with_llm(
        self,
        records: List[Dict[str, Any]],
        record_type: str
    ) -> List[ValidationResult]:
        """Use LLM for intelligent validation of multiple records in a single call."""
        try:
            if not self.llm_provider or not records:
                return []
            
            # Limit batch size to avoid token limits (adjust based on your model)
            batch_size = 5  # Process 5 records at a time
            all_results = []
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                batch_results = await self._process_llm_batch(batch, record_type)
                all_results.extend(batch_results)
            
            return all_results
            
        except Exception as e:
            self.logger.error("Batch LLM validation failed", error=str(e))
            return []
    
    async def _process_llm_batch(
        self,
        batch: List[Dict[str, Any]],
        record_type: str
    ) -> List[ValidationResult]:
        """Process a batch of records with a single LLM call."""
        try:
            # Create batch validation prompt
            prompt = get_prompt(
                "validation_check_batch",
                record_type=record_type,
                records_data=batch,
                validation_rules="Business rules for insurance data",
                batch_size=len(batch)
            )
            
            system_prompt = get_system_prompt("validation")
            
            # Get LLM response for the entire batch
            response = await self.llm_provider.generate(prompt, system_prompt)
            
            # Parse LLM response and distribute results to individual records
            results = []
            for i, record in enumerate(batch):
                # Extract validation result for this specific record from the batch response
                record_result = self._parse_batch_validation_response(response, record, i)
                if record_result:
                    results.append(record_result)
            
            return results
            
        except Exception as e:
            self.logger.error("LLM batch processing failed", error=str(e))
            return []
    
    def _parse_batch_validation_response(
        self,
        response: str,
        record: Dict[str, Any],
        record_index: int
    ) -> Optional[ValidationResult]:
        """Parse LLM response for a specific record from batch validation."""
        try:
            # Simple parsing - look for record-specific issues
            # In a more sophisticated implementation, you'd parse structured JSON response
            
            # Check if response contains warnings or errors
            if "error" in response.lower() or "invalid" in response.lower() or "warning" in response.lower():
                return ValidationResult(
                    field_name="record",
                    value=record,
                    is_valid=True,  # Not blocking, just a warning
                    severity=ValidationSeverity.WARNING,
                    message=f"LLM validation warning for record {record_index + 1}: {response[:200]}"
                )
            
            return None
            
        except Exception as e:
            self.logger.error("Failed to parse batch validation response", error=str(e))
            return None
    
    async def validate_field(
        self,
        field_name: str,
        value: Any,
        validation_rule: ValidationRule
    ) -> ValidationResult:
        """
        Validate a single field against a validation rule.
        
        Args:
            field_name: Name of the field
            value: Field value
            validation_rule: Validation rule to apply
            
        Returns:
            ValidationResult: Validation result
        """
        try:
            # Required field validation
            if validation_rule.required and (value is None or value == ""):
                return ValidationResult(
                    field_name=field_name,
                    value=value,
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required field '{field_name}' is missing or empty"
                )
            
            # Pattern validation
            if validation_rule.pattern and isinstance(value, str):
                if not re.match(validation_rule.pattern, value):
                    return ValidationResult(
                        field_name=field_name,
                        value=value,
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field_name}' does not match required pattern"
                    )
            
            # Length validation
            if isinstance(value, str):
                if validation_rule.min_length and len(value) < validation_rule.min_length:
                    return ValidationResult(
                        field_name=field_name,
                        value=value,
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field_name}' is too short (minimum {validation_rule.min_length})"
                    )
                
                if validation_rule.max_length and len(value) > validation_rule.max_length:
                    return ValidationResult(
                        field_name=field_name,
                        value=value,
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field_name}' is too long (maximum {validation_rule.max_length})"
                    )
            
            # Numeric range validation
            if isinstance(value, (int, float)):
                if validation_rule.min_value is not None and value < validation_rule.min_value:
                    return ValidationResult(
                        field_name=field_name,
                        value=value,
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field_name}' value {value} is below minimum {validation_rule.min_value}"
                    )
                
                if validation_rule.max_value is not None and value > validation_rule.max_value:
                    return ValidationResult(
                        field_name=field_name,
                        value=value,
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field_name}' value {value} is above maximum {validation_rule.max_value}"
                    )
            
            # Date validation
            if validation_rule.future_date is not None and isinstance(value, str):
                try:
                    date_value = datetime.strptime(value, "%Y-%m-%d").date()
                    today = date.today()
                    
                    if not validation_rule.future_date and date_value > today:
                        return ValidationResult(
                            field_name=field_name,
                            value=value,
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            message=f"Field '{field_name}' date cannot be in the future"
                        )
                except ValueError:
                    return ValidationResult(
                        field_name=field_name,
                        value=value,
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field_name}' is not a valid date"
                    )
            
            # All validations passed
            return ValidationResult(
                field_name=field_name,
                value=value,
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message=f"Field '{field_name}' is valid"
            )
            
        except Exception as e:
            return ValidationResult(
                field_name=field_name,
                value=value,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Validation error for field '{field_name}': {str(e)}"
            )
