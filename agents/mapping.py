"""
Mapping Agent for the Migration-Accelerators platform.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import structlog

from .base_agent import BaseAgent, AgentResult
from config.settings import LLMConfig, MCPConfig, FieldMappingRule, TransformationType
from llm.providers import BaseLLMProvider, LLMProviderFactory
from llm.prompts import get_prompt, get_system_prompt


class MappingAgent(BaseAgent):
    """
    Mapping Agent for field transformation and data mapping.
    
    This agent handles:
    - Field mapping according to configuration rules
    - Data type transformations
    - Lookup table applications
    - Calculated field derivations
    - Conditional logic processing
    - LLM-powered intelligent mapping
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        mcp_config: Optional[MCPConfig] = None
    ):
        super().__init__("mapping", llm_config, mcp_config)
        self.llm_provider: Optional[BaseLLMProvider] = None
        self.mapping_cache: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize the mapping agent."""
        await super().start()
        
        if self.llm_config:
            try:
                self.llm_provider = LLMProviderFactory.create(self.llm_config, self.agent_name)
                await self.llm_provider.initialize()
                self.logger.info("LLM provider initialized for mapping agent")
            except ImportError as e:
                self.logger.warning(f"LLM provider not available: {e}")
                self.logger.info("Running without LLM enhancements")
                self.llm_provider = None
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM provider: {e}")
                self.logger.info("Running without LLM enhancements")
                self.llm_provider = None
        
        self.logger.info("Mapping agent initialized")
    
    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Process mapping request.
        
        Args:
            data: Source data to transform (record or list of records)
            context: Mapping context (mapping_rules, record_type, etc.)
            
        Returns:
            AgentResult: Mapping result with transformed data
        """
        try:
            self.logger.info("Starting mapping process", data_type=type(data).__name__)
            
            # Validate input
            if not await self.validate_input(data):
                return AgentResult(
                    success=False,
                    errors=["Invalid input data for mapping"]
                )
            
            # Get mapping context
            mapping_rules = context.get("mapping_rules", []) if context else []
            record_type = context.get("record_type", "unknown") if context else "unknown"
            
            if not mapping_rules:
                return AgentResult(
                    success=False,
                    errors=["No mapping rules provided"]
                )
            
            # Transform data
            if isinstance(data, list):
                transformed_data = await self._transform_batch(data, mapping_rules, record_type)
            else:
                transformed_data = await self._transform_record(data, mapping_rules, record_type)
            
            # Validate transformation results
            validation_result = await self._validate_transformation(transformed_data, mapping_rules)
            
            success = validation_result["success"]
            errors = validation_result.get("errors", [])
            warnings = validation_result.get("warnings", [])
            
            self.logger.info(
                "Mapping completed",
                success=success,
                source_records=len(data) if isinstance(data, list) else 1,
                transformed_records=len(transformed_data) if isinstance(transformed_data, list) else 1,
                errors_count=len(errors)
            )
            
            return AgentResult(
                success=success,
                data=transformed_data,
                errors=errors,
                warnings=warnings,
                metadata={
                    "record_type": record_type,
                    "mapping_rules_count": len(mapping_rules),
                    "transformation_summary": validation_result.get("summary", {}),
                    "agent": self.agent_name
                }
            )
            
        except Exception as e:
            return await self.handle_error(e, {"data_type": type(data).__name__, "context": context})
    
    async def _transform_batch(
        self,
        records: List[Dict[str, Any]],
        mapping_rules: List[FieldMappingRule],
        record_type: str
    ) -> List[Dict[str, Any]]:
        """Transform a batch of records."""
        transformed_records = []
        
        for i, record in enumerate(records):
            try:
                transformed_record = await self._transform_record(record, mapping_rules, record_type)
                transformed_records.append(transformed_record)
                
                self.logger.debug("Record transformed", record_index=i)
                
            except Exception as e:
                self.logger.error("Error transforming record", record_index=i, error=str(e))
                # Add error record with original data and error info
                error_record = {
                    **record,
                    "_transformation_error": str(e),
                    "_record_index": i
                }
                transformed_records.append(error_record)
        
        return transformed_records
    
    async def _transform_record(
        self,
        record: Dict[str, Any],
        mapping_rules: List[FieldMappingRule],
        record_type: str
    ) -> Dict[str, Any]:
        """Transform a single record according to mapping rules."""
        transformed_record = {}
        
        for rule in mapping_rules:
            try:
                # Get source value
                source_value = record.get(rule.source_field)
                
                # Apply transformation
                transformed_value = await self._apply_transformation(
                    source_value, rule, record, record_type
                )
                
                # Set target field
                transformed_record[rule.target_field] = transformed_value
                
            except Exception as e:
                self.logger.error(
                    "Error applying mapping rule",
                    source_field=rule.source_field,
                    target_field=rule.target_field,
                    error=str(e)
                )
                # Set error value
                transformed_record[rule.target_field] = None
        
        # Add metadata
        transformed_record["_mapping_metadata"] = {
            "record_type": record_type,
            "transformation_timestamp": datetime.utcnow().isoformat(),
            "source_fields_count": len(record),
            "target_fields_count": len(transformed_record)
        }
        
        return transformed_record
    
    async def _apply_transformation(
        self,
        source_value: Any,
        rule: FieldMappingRule,
        full_record: Dict[str, Any],
        record_type: str
    ) -> Any:
        """Apply a specific transformation rule."""
        
        if rule.transformation_type == TransformationType.DIRECT:
            return await self._direct_transformation(source_value, rule)
        
        elif rule.transformation_type == TransformationType.DATE_FORMAT:
            return await self._date_format_transformation(source_value, rule)
        
        elif rule.transformation_type == TransformationType.LOOKUP:
            return await self._lookup_transformation(source_value, rule)
        
        elif rule.transformation_type == TransformationType.CALCULATED:
            return await self._calculated_transformation(source_value, rule, full_record)
        
        elif rule.transformation_type == TransformationType.CONDITIONAL:
            return await self._conditional_transformation(source_value, rule, full_record)
        
        else:
            self.logger.warning("Unknown transformation type", type=rule.transformation_type)
            return source_value
    
    async def _direct_transformation(self, source_value: Any, rule: FieldMappingRule) -> Any:
        """Apply direct transformation (no change)."""
        return source_value
    
    async def _date_format_transformation(self, source_value: Any, rule: FieldMappingRule) -> Any:
        """Apply date format transformation."""
        if not source_value:
            return None
        
        try:
            # Parse source date
            source_format = rule.source_format or "%Y-%m-%d"
            parsed_date = datetime.strptime(str(source_value), source_format)
            
            # Format target date
            target_format = rule.target_format or "ISO8601"
            
            if target_format.upper() == "ISO8601":
                return parsed_date.isoformat()
            else:
                return parsed_date.strftime(target_format)
                
        except ValueError as e:
            self.logger.error("Date format transformation failed", error=str(e))
            return None
    
    async def _lookup_transformation(self, source_value: Any, rule: FieldMappingRule) -> Any:
        """Apply lookup table transformation."""
        if not source_value or not rule.lookup_table:
            return source_value
        
        # Convert source value to string for lookup
        source_str = str(source_value).lower()
        
        # Look for exact match
        for key, value in rule.lookup_table.items():
            if key.lower() == source_str:
                return value
        
        # Look for partial match
        for key, value in rule.lookup_table.items():
            if source_str in key.lower() or key.lower() in source_str:
                return value
        
        # Return original value if no match found
        self.logger.warning("Lookup transformation: no match found", source_value=source_value)
        return source_value
    
    async def _calculated_transformation(
        self,
        source_value: Any,
        rule: FieldMappingRule,
        full_record: Dict[str, Any]
    ) -> Any:
        """Apply calculated field transformation."""
        if not rule.calculation_formula:
            return source_value
        
        try:
            # Handle None values
            if source_value is None:
                self.logger.warning("Source value is None for calculated transformation", 
                                  field=rule.source_field, formula=rule.calculation_formula)
                return None
            
            # Create safe evaluation context
            context = {
                "source_value": source_value,
                "record": full_record,
                "int": int,
                "float": float,
                "str": str,
                "len": len,
                "abs": abs,
                "round": round
            }
            
            # Evaluate formula safely
            result = eval(rule.calculation_formula, {"__builtins__": {}}, context)
            return result
            
        except Exception as e:
            self.logger.error("Calculated transformation failed", error=str(e))
            return source_value
    
    async def _conditional_transformation(
        self,
        source_value: Any,
        rule: FieldMappingRule,
        full_record: Dict[str, Any]
    ) -> Any:
        """Apply conditional transformation."""
        if not rule.condition:
            return source_value
        
        try:
            # Handle None values
            if source_value is None:
                self.logger.warning("Source value is None for conditional transformation", 
                                  field=rule.source_field, condition=rule.condition)
                return None
            
            # Create safe evaluation context
            context = {
                "source_value": source_value,
                "record": full_record,
                "int": int,
                "float": float,
                "str": str,
                "len": len,
                "abs": abs,
                "round": round
            }
            
            # Evaluate condition safely
            result = eval(rule.condition, {"__builtins__": {}}, context)
            return result
            
        except Exception as e:
            self.logger.error("Conditional transformation failed", error=str(e))
            return source_value
    
    async def _validate_transformation(
        self,
        transformed_data: Union[Dict[str, Any], List[Dict[str, Any]]],
        mapping_rules: List[FieldMappingRule]
    ) -> Dict[str, Any]:
        """Validate transformation results."""
        errors = []
        warnings = []
        summary = {}
        
        if isinstance(transformed_data, list):
            records = transformed_data
        else:
            records = [transformed_data]
        
        # Check if all target fields are present
        target_fields = {rule.target_field for rule in mapping_rules}
        
        for i, record in enumerate(records):
            record_errors = []
            record_warnings = []
            
            # Check for missing target fields
            missing_fields = target_fields - set(record.keys())
            if missing_fields:
                record_errors.append(f"Missing target fields: {missing_fields}")
            
            # Check for null values in required fields
            for rule in mapping_rules:
                if rule.validation and rule.validation.required:
                    if record.get(rule.target_field) is None:
                        record_errors.append(f"Required field '{rule.target_field}' is null")
            
            # Check for transformation errors
            if "_transformation_error" in record:
                record_errors.append(f"Transformation error: {record['_transformation_error']}")
            
            if record_errors:
                errors.extend([f"Record {i}: {error}" for error in record_errors])
            if record_warnings:
                warnings.extend([f"Record {i}: {warning}" for warning in record_warnings])
        
        # Create summary
        summary = {
            "total_records": len(records),
            "records_with_errors": len([r for r in records if "_transformation_error" in r]),
            "target_fields_count": len(target_fields),
            "mapping_rules_count": len(mapping_rules)
        }
        
        return {
            "success": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "summary": summary
        }
    
    async def discover_mapping_rules(
        self,
        source_data: List[Dict[str, Any]],
        target_schema: Dict[str, Any],
        record_type: str
    ) -> List[FieldMappingRule]:
        """
        Discover mapping rules using LLM.
        
        Args:
            source_data: Sample source data
            target_schema: Target schema definition
            record_type: Type of records
            
        Returns:
            List[FieldMappingRule]: Discovered mapping rules
        """
        if not self.llm_provider:
            self.logger.warning("LLM provider not available for rule discovery")
            return []
        
        try:
            # Create discovery prompt
            prompt = get_prompt(
                "mapping_transform_record",
                source_record=source_data[0] if source_data else {},
                mapping_rules="Auto-discover mapping rules",
                record_type=record_type
            )
            
            system_prompt = get_system_prompt("mapping")
            
            # Get LLM response
            response = await self.llm_provider.generate(prompt, system_prompt)
            
            # Parse response to extract mapping rules
            # This is a simplified implementation
            # In a real system, you would have more sophisticated parsing
            
            discovered_rules = []
            
            # Simple rule discovery based on field name similarity
            source_fields = set(source_data[0].keys()) if source_data else set()
            target_fields = set(target_schema.get("properties", {}).keys())
            
            for source_field in source_fields:
                for target_field in target_fields:
                    # Simple similarity check
                    if (source_field.lower() in target_field.lower() or 
                        target_field.lower() in source_field.lower()):
                        
                        rule = FieldMappingRule(
                            source_field=source_field,
                            target_field=target_field,
                            transformation_type=TransformationType.DIRECT
                        )
                        discovered_rules.append(rule)
            
            self.logger.info("Mapping rules discovered", rules_count=len(discovered_rules))
            return discovered_rules
            
        except Exception as e:
            self.logger.error("Error discovering mapping rules", error=str(e))
            return []
    
    async def optimize_mapping_rules(
        self,
        mapping_rules: List[FieldMappingRule],
        performance_data: Dict[str, Any]
    ) -> List[FieldMappingRule]:
        """
        Optimize mapping rules based on performance data.
        
        Args:
            mapping_rules: Current mapping rules
            performance_data: Performance metrics
            
        Returns:
            List[FieldMappingRule]: Optimized mapping rules
        """
        # This is a placeholder for optimization logic
        # In a real system, you would analyze performance data and optimize rules
        
        optimized_rules = []
        
        for rule in mapping_rules:
            # Simple optimization: cache lookup tables
            if rule.transformation_type == TransformationType.LOOKUP:
                cache_key = f"lookup_{rule.source_field}_{rule.target_field}"
                if cache_key not in self.mapping_cache:
                    self.mapping_cache[cache_key] = rule.lookup_table
            
            optimized_rules.append(rule)
        
        self.logger.info("Mapping rules optimized", rules_count=len(optimized_rules))
        return optimized_rules
    
    async def validate_mapping_rule(self, rule: FieldMappingRule) -> Dict[str, Any]:
        """
        Validate a single mapping rule.
        
        Args:
            rule: Mapping rule to validate
            
        Returns:
            Dict[str, Any]: Validation result
        """
        errors = []
        warnings = []
        
        # Check required fields
        if not rule.source_field:
            errors.append("Source field is required")
        
        if not rule.target_field:
            errors.append("Target field is required")
        
        # Check transformation-specific requirements
        if rule.transformation_type == TransformationType.DATE_FORMAT:
            if not rule.source_format:
                warnings.append("Source format not specified for date transformation")
            if not rule.target_format:
                warnings.append("Target format not specified for date transformation")
        
        elif rule.transformation_type == TransformationType.LOOKUP:
            if not rule.lookup_table:
                errors.append("Lookup table is required for lookup transformation")
        
        elif rule.transformation_type == TransformationType.CALCULATED:
            if not rule.calculation_formula:
                errors.append("Calculation formula is required for calculated transformation")
        
        elif rule.transformation_type == TransformationType.CONDITIONAL:
            if not rule.condition:
                errors.append("Condition is required for conditional transformation")
        
        return {
            "success": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
