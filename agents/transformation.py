"""
Transformation Agent for the Migration-Accelerators platform.
"""

import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union
import structlog

from .base_agent import BaseAgent, AgentResult
from config.settings import LLMConfig, MCPConfig, FileFormat
from llm.providers import BaseLLMProvider, LLMProviderFactory
from llm.prompts import get_prompt, get_system_prompt


class TransformationAgent(BaseAgent):
    """
    Transformation Agent for data format conversion and optimization.
    
    This agent handles:
    - Converting between CSV, JSON, XML formats
    - Optimizing data structure for target systems
    - Handling nested data and complex structures
    - Ensuring format compliance
    - Maintaining data relationships
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        mcp_config: Optional[MCPConfig] = None
    ):
        super().__init__("transformation", llm_config, mcp_config)
        self.llm_provider: Optional[BaseLLMProvider] = None
        self.format_converters = {
            (FileFormat.CSV, FileFormat.JSON): self._csv_to_json,
            (FileFormat.CSV, FileFormat.XML): self._csv_to_xml,
            (FileFormat.JSON, FileFormat.CSV): self._json_to_csv,
            (FileFormat.JSON, FileFormat.XML): self._json_to_xml,
            (FileFormat.XML, FileFormat.CSV): self._xml_to_csv,
            (FileFormat.XML, FileFormat.JSON): self._xml_to_json,
        }
    
    async def initialize(self) -> None:
        """Initialize the transformation agent."""
        await super().start()
        
        if self.llm_config:
            self.llm_provider = LLMProviderFactory.create(self.llm_config, self.agent_name)
            await self.llm_provider.initialize()
            self.logger.info("LLM provider initialized for transformation agent")
        
        self.logger.info("Transformation agent initialized")
    
    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Process transformation request.
        
        Args:
            data: Data to transform
            context: Transformation context (source_format, target_format, schema, etc.)
            
        Returns:
            AgentResult: Transformation result
        """
        try:
            self.logger.info("Starting transformation process", data_type=type(data).__name__)
            
            # Validate input
            if not await self.validate_input(data):
                return AgentResult(
                    success=False,
                    errors=["Invalid input data for transformation"]
                )
            
            # Get transformation context
            source_format = context.get("source_format", FileFormat.JSON) if context else FileFormat.JSON
            target_format = context.get("target_format", FileFormat.JSON) if context else FileFormat.JSON
            target_schema = context.get("target_schema") if context else None
            conversion_rules = context.get("conversion_rules", {}) if context else {}
            
            # Check if conversion is needed
            if source_format == target_format:
                self.logger.info("Source and target formats are the same, no conversion needed")
                return AgentResult(
                    success=True,
                    data=data,
                    metadata={
                        "source_format": source_format.value,
                        "target_format": target_format.value,
                        "conversion_performed": False,
                        "agent": self.agent_name
                    }
                )
            
            # Perform transformation
            transformed_data = await self._convert_format(
                data, source_format, target_format, target_schema, conversion_rules
            )
            
            # Validate transformation result
            validation_result = await self._validate_transformation(
                transformed_data, target_format, target_schema
            )
            
            success = validation_result["success"]
            errors = validation_result.get("errors", [])
            warnings = validation_result.get("warnings", [])
            
            self.logger.info(
                "Transformation completed",
                success=success,
                source_format=source_format.value,
                target_format=target_format.value,
                errors_count=len(errors)
            )
            
            return AgentResult(
                success=success,
                data=transformed_data,
                errors=errors,
                warnings=warnings,
                metadata={
                    "source_format": source_format.value,
                    "target_format": target_format.value,
                    "conversion_performed": True,
                    "validation_result": validation_result,
                    "agent": self.agent_name
                }
            )
            
        except Exception as e:
            return await self.handle_error(e, {"data_type": type(data).__name__, "context": context})
    
    async def _convert_format(
        self,
        data: Any,
        source_format: FileFormat,
        target_format: FileFormat,
        target_schema: Optional[Dict[str, Any]],
        conversion_rules: Dict[str, Any]
    ) -> Any:
        """Convert data from source format to target format."""
        
        # Get converter function
        converter_key = (source_format, target_format)
        if converter_key in self.format_converters:
            converter_func = self.format_converters[converter_key]
            return await converter_func(data, target_schema, conversion_rules)
        else:
            # Use LLM for complex transformations
            if self.llm_provider:
                return await self._llm_convert_format(
                    data, source_format, target_format, target_schema, conversion_rules
                )
            else:
                raise ValueError(f"Conversion from {source_format.value} to {target_format.value} not supported")
    
    async def _csv_to_json(
        self,
        data: List[Dict[str, Any]],
        target_schema: Optional[Dict[str, Any]],
        conversion_rules: Dict[str, Any]
    ) -> str:
        """Convert CSV data to JSON format."""
        try:
            # Optimize data structure if schema provided
            if target_schema:
                optimized_data = await self._optimize_for_schema(data, target_schema)
            else:
                optimized_data = data
            
            # Convert to JSON
            json_data = json.dumps(optimized_data, indent=2, default=str)
            
            self.logger.info("CSV to JSON conversion completed", records_count=len(data))
            return json_data
            
        except Exception as e:
            self.logger.error("CSV to JSON conversion failed", error=str(e))
            raise
    
    async def _csv_to_xml(
        self,
        data: List[Dict[str, Any]],
        target_schema: Optional[Dict[str, Any]],
        conversion_rules: Dict[str, Any]
    ) -> str:
        """Convert CSV data to XML format."""
        try:
            root_tag = conversion_rules.get("root_tag", "records")
            record_tag = conversion_rules.get("record_tag", "record")
            
            root = ET.Element(root_tag)
            
            for record in data:
                record_elem = ET.SubElement(root, record_tag)
                for key, value in record.items():
                    field_elem = ET.SubElement(record_elem, key)
                    field_elem.text = str(value) if value is not None else ""
            
            xml_data = ET.tostring(root, encoding='unicode', method='xml')
            
            self.logger.info("CSV to XML conversion completed", records_count=len(data))
            return xml_data
            
        except Exception as e:
            self.logger.error("CSV to XML conversion failed", error=str(e))
            raise
    
    async def _json_to_csv(
        self,
        data: Union[str, List[Dict[str, Any]]],
        target_schema: Optional[Dict[str, Any]],
        conversion_rules: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert JSON data to CSV format."""
        try:
            # Parse JSON if string
            if isinstance(data, str):
                parsed_data = json.loads(data)
            else:
                parsed_data = data
            
            # Ensure data is a list
            if isinstance(parsed_data, dict):
                # Check if it's a single record or container
                if any(key in parsed_data for key in ['records', 'data', 'items']):
                    records = parsed_data.get('records', parsed_data.get('data', parsed_data.get('items', [parsed_data])))
                else:
                    records = [parsed_data]
            else:
                records = parsed_data
            
            # Flatten nested structures if needed
            flattened_records = []
            for record in records:
                flattened_record = await self._flatten_record(record)
                flattened_records.append(flattened_record)
            
            self.logger.info("JSON to CSV conversion completed", records_count=len(flattened_records))
            return flattened_records
            
        except Exception as e:
            self.logger.error("JSON to CSV conversion failed", error=str(e))
            raise
    
    async def _json_to_xml(
        self,
        data: Union[str, List[Dict[str, Any]]],
        target_schema: Optional[Dict[str, Any]],
        conversion_rules: Dict[str, Any]
    ) -> str:
        """Convert JSON data to XML format."""
        try:
            # Parse JSON if string
            if isinstance(data, str):
                parsed_data = json.loads(data)
            else:
                parsed_data = data
            
            # Ensure data is a list
            if isinstance(parsed_data, dict):
                if any(key in parsed_data for key in ['records', 'data', 'items']):
                    records = parsed_data.get('records', parsed_data.get('data', parsed_data.get('items', [parsed_data])))
                else:
                    records = [parsed_data]
            else:
                records = parsed_data
            
            root_tag = conversion_rules.get("root_tag", "records")
            record_tag = conversion_rules.get("record_tag", "record")
            
            root = ET.Element(root_tag)
            
            for record in records:
                record_elem = ET.SubElement(root, record_tag)
                await self._dict_to_xml(record, record_elem)
            
            xml_data = ET.tostring(root, encoding='unicode', method='xml')
            
            self.logger.info("JSON to XML conversion completed", records_count=len(records))
            return xml_data
            
        except Exception as e:
            self.logger.error("JSON to XML conversion failed", error=str(e))
            raise
    
    async def _xml_to_csv(
        self,
        data: str,
        target_schema: Optional[Dict[str, Any]],
        conversion_rules: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert XML data to CSV format."""
        try:
            root = ET.fromstring(data)
            record_tag = conversion_rules.get("record_tag", "record")
            
            records = []
            for element in root.findall(f".//{record_tag}"):
                record = {}
                for child in element:
                    record[child.tag] = child.text
                records.append(record)
            
            # If no records found with specified tag, try to extract from root
            if not records and root:
                record = {}
                for child in root:
                    record[child.tag] = child.text
                if record:
                    records = [record]
            
            self.logger.info("XML to CSV conversion completed", records_count=len(records))
            return records
            
        except Exception as e:
            self.logger.error("XML to CSV conversion failed", error=str(e))
            raise
    
    async def _xml_to_json(
        self,
        data: str,
        target_schema: Optional[Dict[str, Any]],
        conversion_rules: Dict[str, Any]
    ) -> str:
        """Convert XML data to JSON format."""
        try:
            root = ET.fromstring(data)
            record_tag = conversion_rules.get("record_tag", "record")
            
            records = []
            for element in root.findall(f".//{record_tag}"):
                record = {}
                for child in element:
                    record[child.tag] = child.text
                records.append(record)
            
            # If no records found with specified tag, try to extract from root
            if not records and root:
                record = {}
                for child in root:
                    record[child.tag] = child.text
                if record:
                    records = [record]
            
            # Convert to JSON
            json_data = json.dumps(records, indent=2, default=str)
            
            self.logger.info("XML to JSON conversion completed", records_count=len(records))
            return json_data
            
        except Exception as e:
            self.logger.error("XML to JSON conversion failed", error=str(e))
            raise
    
    async def _llm_convert_format(
        self,
        data: Any,
        source_format: FileFormat,
        target_format: FileFormat,
        target_schema: Optional[Dict[str, Any]],
        conversion_rules: Dict[str, Any]
    ) -> Any:
        """Use LLM for complex format conversions."""
        try:
            # Create conversion prompt
            prompt = get_prompt(
                "transformation_convert_format",
                source_format=source_format.value,
                target_format=target_format.value,
                source_data=str(data)[:1000],  # Limit data size
                target_schema=target_schema or {},
                conversion_rules=conversion_rules
            )
            
            system_prompt = get_system_prompt("transformation")
            
            # Get LLM response
            response = await self.llm_provider.generate(prompt, system_prompt)
            
            # Try to parse response as target format
            if target_format == FileFormat.JSON:
                return json.loads(response)
            else:
                return response
                
        except Exception as e:
            self.logger.error("LLM format conversion failed", error=str(e))
            raise
    
    async def _optimize_for_schema(
        self,
        data: List[Dict[str, Any]],
        schema: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Optimize data structure for target schema."""
        optimized_data = []
        
        for record in data:
            optimized_record = {}
            
            # Map fields according to schema
            properties = schema.get("properties", {})
            for field_name, field_schema in properties.items():
                # Try to find matching field in source data
                source_value = None
                for source_field, source_value in record.items():
                    if (source_field.lower() == field_name.lower() or
                        field_name.lower() in source_field.lower()):
                        source_value = source_value
                        break
                
                if source_value is not None:
                    # Apply type conversion if needed
                    expected_type = field_schema.get("type")
                    if expected_type == "string":
                        optimized_record[field_name] = str(source_value)
                    elif expected_type == "number":
                        try:
                            optimized_record[field_name] = float(source_value)
                        except (ValueError, TypeError):
                            optimized_record[field_name] = 0.0
                    elif expected_type == "integer":
                        try:
                            optimized_record[field_name] = int(source_value)
                        except (ValueError, TypeError):
                            optimized_record[field_name] = 0
                    elif expected_type == "boolean":
                        optimized_record[field_name] = bool(source_value)
                    else:
                        optimized_record[field_name] = source_value
                else:
                    # Set default value
                    default_value = field_schema.get("default")
                    if default_value is not None:
                        optimized_record[field_name] = default_value
            
            optimized_data.append(optimized_record)
        
        return optimized_data
    
    async def _flatten_record(self, record: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested record structure."""
        flattened = {}
        
        for key, value in record.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                nested_flattened = await self._flatten_record(value, new_key)
                flattened.update(nested_flattened)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        nested_flattened = await self._flatten_record(item, f"{new_key}[{i}]")
                        flattened.update(nested_flattened)
                    else:
                        flattened[f"{new_key}[{i}]"] = item
            else:
                flattened[new_key] = value
        
        return flattened
    
    async def _dict_to_xml(self, data: Dict[str, Any], parent: ET.Element) -> None:
        """Convert dictionary to XML elements."""
        for key, value in data.items():
            if isinstance(value, dict):
                child = ET.SubElement(parent, key)
                await self._dict_to_xml(value, child)
            elif isinstance(value, list):
                for item in value:
                    child = ET.SubElement(parent, key)
                    if isinstance(item, dict):
                        await self._dict_to_xml(item, child)
                    else:
                        child.text = str(item) if item is not None else ""
            else:
                child = ET.SubElement(parent, key)
                child.text = str(value) if value is not None else ""
    
    async def _validate_transformation(
        self,
        transformed_data: Any,
        target_format: FileFormat,
        target_schema: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate transformation result."""
        errors = []
        warnings = []
        
        try:
            # Format-specific validation
            if target_format == FileFormat.JSON:
                if isinstance(transformed_data, str):
                    json.loads(transformed_data)  # Validate JSON syntax
                elif not isinstance(transformed_data, (dict, list)):
                    errors.append("JSON format requires dict or list data")
            
            elif target_format == FileFormat.XML:
                if isinstance(transformed_data, str):
                    ET.fromstring(transformed_data)  # Validate XML syntax
                else:
                    errors.append("XML format requires string data")
            
            elif target_format == FileFormat.CSV:
                if not isinstance(transformed_data, list):
                    errors.append("CSV format requires list of records")
                elif transformed_data and not isinstance(transformed_data[0], dict):
                    errors.append("CSV format requires list of dictionaries")
            
            # Schema validation if provided
            if target_schema and isinstance(transformed_data, list):
                schema_errors = await self._validate_against_schema(transformed_data, target_schema)
                errors.extend(schema_errors)
            
            return {
                "success": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            }
            
        except Exception as e:
            return {
                "success": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": warnings
            }
    
    async def _validate_against_schema(
        self,
        data: List[Dict[str, Any]],
        schema: Dict[str, Any]
    ) -> List[str]:
        """Validate data against schema."""
        errors = []
        
        # Check required fields
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})
        
        for i, record in enumerate(data):
            for field in required_fields:
                if field not in record or record[field] is None:
                    errors.append(f"Record {i}: Required field '{field}' is missing")
            
            # Check field types
            for field, value in record.items():
                if field in properties:
                    field_schema = properties[field]
                    expected_type = field_schema.get("type")
                    
                    if expected_type == "string" and not isinstance(value, str):
                        errors.append(f"Record {i}: Field '{field}' should be string")
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        errors.append(f"Record {i}: Field '{field}' should be number")
                    elif expected_type == "integer" and not isinstance(value, int):
                        errors.append(f"Record {i}: Field '{field}' should be integer")
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        errors.append(f"Record {i}: Field '{field}' should be boolean")
        
        return errors
    
    async def create_target_file(
        self,
        data: Any,
        file_path: str,
        target_format: FileFormat,
        target_schema: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Create target file with transformed data.
        
        Args:
            data: Transformed data
            file_path: Path for the target file
            target_format: Target file format
            target_schema: Target schema for validation
            
        Returns:
            AgentResult: File creation result
        """
        try:
            import aiofiles
            
            # Prepare file content based on format
            if target_format == FileFormat.JSON:
                if isinstance(data, str):
                    content = data
                else:
                    content = json.dumps(data, indent=2, default=str)
                file_extension = ".json"
            
            elif target_format == FileFormat.XML:
                if isinstance(data, str):
                    content = data
                else:
                    # Convert to XML string
                    root = ET.Element("data")
                    await self._dict_to_xml(data, root)
                    content = ET.tostring(root, encoding='unicode', method='xml')
                file_extension = ".xml"
            
            elif target_format == FileFormat.CSV:
                if isinstance(data, list) and data:
                    import csv
                    import io
                    
                    # Convert to CSV string
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                    content = output.getvalue()
                else:
                    content = ""
                file_extension = ".csv"
            
            else:
                return AgentResult(
                    success=False,
                    errors=[f"Unsupported target format: {target_format.value}"]
                )
            
            # Ensure file has correct extension
            if not file_path.endswith(file_extension):
                file_path += file_extension
            
            # Write file
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as file:
                await file.write(content)
            
            self.logger.info("Target file created", file_path=file_path, format=target_format.value)
            
            return AgentResult(
                success=True,
                data={"file_path": file_path, "content_length": len(content)},
                metadata={
                    "target_format": target_format.value,
                    "file_size": len(content),
                    "agent": self.agent_name
                }
            )
            
        except Exception as e:
            return await self.handle_error(e, {"file_path": file_path, "target_format": target_format.value})
