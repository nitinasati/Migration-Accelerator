"""
File Reader Agent for the Migration-Accelerators platform.
"""

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import structlog

from .base_agent import BaseAgent, AgentResult
from config.settings import LLMConfig, MCPConfig, FileFormat
from llm.providers import BaseLLMProvider, LLMProviderFactory
from llm.prompts import get_prompt, get_system_prompt


class FileReaderAgent(BaseAgent):
    """
    File Reader Agent for processing mainframe and legacy system files.
    
    This agent handles:
    - Automatic file format detection
    - Parsing various file formats (CSV, Excel, XML, JSON, Fixed-width)
    - Encoding detection and handling
    - Data extraction and validation
    - Metadata collection
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        mcp_config: Optional[MCPConfig] = None
    ):
        super().__init__("file_reader", llm_config, mcp_config)
        self.llm_provider: Optional[BaseLLMProvider] = None
        self.supported_formats = {
            FileFormat.CSV: self._read_csv,
            FileFormat.EXCEL: self._read_excel,
            FileFormat.JSON: self._read_json,
            FileFormat.XML: self._read_xml,
            FileFormat.FIXED_WIDTH: self._read_fixed_width
        }
    
    async def initialize(self) -> None:
        """Initialize the file reader agent."""
        await super().start()
        
        if self.llm_config:
            try:
                self.llm_provider = LLMProviderFactory.create(self.llm_config, self.agent_name)
                await self.llm_provider.initialize()
                self.logger.info("LLM provider initialized for file reader")
            except ImportError as e:
                self.logger.warning(f"LLM provider not available: {e}")
                self.logger.info("Running without LLM enhancements")
                self.llm_provider = None
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM provider: {e}")
                self.logger.info("Running without LLM enhancements")
                self.llm_provider = None
        
        self.logger.info("File reader agent initialized")
    
    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Process file reading request.
        
        Args:
            data: File path or file data
            context: Additional context (file format, encoding, etc.)
            
        Returns:
            AgentResult: Processing result with file data
        """
        try:
            self.logger.info("Starting file reading process", data_type=type(data).__name__)
            
            # Validate input
            if not await self.validate_input(data):
                return AgentResult(
                    success=False,
                    errors=["Invalid input data for file reading"]
                )
            
            # Determine if data is file path or file content
            if isinstance(data, str) and Path(data).exists():
                file_path = data
                file_data = None
            else:
                file_path = context.get("file_path") if context else None
                file_data = data
            
            # Detect file format if not provided
            file_format = context.get("file_format") if context else None
            if not file_format:
                file_format = await self._detect_file_format(file_path, file_data)
            
            # Read file based on format
            if file_format in self.supported_formats:
                reader_func = self.supported_formats[file_format]
                result = await reader_func(file_path, file_data, context)
            else:
                return AgentResult(
                    success=False,
                    errors=[f"Unsupported file format: {file_format}"]
                )
            
            # Add metadata
            result.metadata.update({
                "file_format": file_format.value if hasattr(file_format, 'value') else str(file_format),
                "file_path": file_path,
                "agent": self.agent_name
            })
            
            self.logger.info(
                "File reading completed",
                format=file_format,
                records_count=len(result.data) if isinstance(result.data, list) else 1
            )
            
            return result
            
        except Exception as e:
            return await self.handle_error(e, {"data_type": type(data).__name__, "context": context})
    
    async def _detect_file_format(self, file_path: Optional[str], file_data: Any) -> FileFormat:
        """
        Detect file format using LLM or file extension.
        
        Args:
            file_path: Path to the file
            file_data: File data content
            
        Returns:
            FileFormat: Detected file format
        """
        try:
            # First try file extension
            if file_path:
                extension = Path(file_path).suffix.lower()
                format_mapping = {
                    '.csv': FileFormat.CSV,
                    '.xlsx': FileFormat.EXCEL,
                    '.xls': FileFormat.EXCEL,
                    '.json': FileFormat.JSON,
                    '.xml': FileFormat.XML,
                    '.txt': FileFormat.FIXED_WIDTH
                }
                
                if extension in format_mapping:
                    detected_format = format_mapping[extension]
                    self.logger.info("File format detected by extension", format=detected_format.value)
                    return detected_format
            
            # Use LLM for intelligent detection if available
            if self.llm_provider and file_data:
                return await self._detect_format_with_llm(file_path, file_data)
            
            # Default to CSV if cannot determine
            self.logger.warning("Could not detect file format, defaulting to CSV")
            return FileFormat.CSV
            
        except Exception as e:
            self.logger.error("Error detecting file format", error=str(e))
            return FileFormat.CSV
    
    async def _detect_format_with_llm(self, file_path: Optional[str], file_data: Any) -> FileFormat:
        """
        Use LLM to detect file format.
        
        Args:
            file_path: Path to the file
            file_data: File data content
            
        Returns:
            FileFormat: Detected file format
        """
        try:
            # Prepare file preview
            if isinstance(file_data, str):
                preview = file_data[:1000]
            else:
                preview = str(file_data)[:1000]
            
            file_size = len(str(file_data))
            
            # Create detection prompt
            prompt = get_prompt(
                "file_reader_detect_format",
                file_path=file_path or "unknown",
                file_size=file_size,
                file_preview=preview
            )
            
            system_prompt = get_system_prompt("file_reader")
            
            # Get LLM response
            response = await self.llm_provider.generate(prompt, system_prompt)
            
            # Parse response to determine format
            response_lower = response.lower()
            if 'csv' in response_lower:
                return FileFormat.CSV
            elif 'excel' in response_lower or 'xlsx' in response_lower:
                return FileFormat.EXCEL
            elif 'json' in response_lower:
                return FileFormat.JSON
            elif 'xml' in response_lower:
                return FileFormat.XML
            elif 'fixed' in response_lower or 'width' in response_lower:
                return FileFormat.FIXED_WIDTH
            else:
                return FileFormat.CSV
                
        except Exception as e:
            self.logger.error("LLM format detection failed", error=str(e))
            return FileFormat.CSV
    
    async def _read_csv(
        self,
        file_path: Optional[str],
        file_data: Any,
        context: Optional[Dict[str, Any]]
    ) -> AgentResult:
        """Read CSV file."""
        try:
            if file_path:
                # Read from file
                encoding = context.get("encoding", "utf-8")
                delimiter = context.get("delimiter", ",")
                
                with open(file_path, 'r', encoding=encoding) as file:
                    reader = csv.DictReader(file, delimiter=delimiter)
                    records = list(reader)
            else:
                # Read from data
                import io
                encoding = context.get("encoding", "utf-8")
                delimiter = context.get("delimiter", ",")
                
                csv_data = io.StringIO(file_data)
                reader = csv.DictReader(csv_data, delimiter=delimiter)
                records = list(reader)
            
            self.logger.info("CSV file read successfully", records_count=len(records))
            
            return AgentResult(
                success=True,
                data=records,
                metadata={
                    "format": "csv",
                    "records_count": len(records),
                    "columns": list(records[0].keys()) if records else []
                }
            )
            
        except Exception as e:
            self.logger.error("Error reading CSV file", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"CSV reading error: {str(e)}"]
            )
    
    async def _read_excel(
        self,
        file_path: Optional[str],
        file_data: Any,
        context: Optional[Dict[str, Any]]
    ) -> AgentResult:
        """Read Excel file."""
        try:
            if file_path:
                # Read from file
                sheet_name = context.get("sheet_name", 0)
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                # Read from data (would need to save to temp file first)
                return AgentResult(
                    success=False,
                    errors=["Excel file reading from data not implemented"]
                )
            
            # Convert to list of dictionaries
            records = df.to_dict('records')
            
            self.logger.info("Excel file read successfully", records_count=len(records))
            
            return AgentResult(
                success=True,
                data=records,
                metadata={
                    "format": "excel",
                    "records_count": len(records),
                    "columns": list(df.columns)
                }
            )
            
        except Exception as e:
            self.logger.error("Error reading Excel file", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"Excel reading error: {str(e)}"]
            )
    
    async def _read_json(
        self,
        file_path: Optional[str],
        file_data: Any,
        context: Optional[Dict[str, Any]]
    ) -> AgentResult:
        """Read JSON file."""
        try:
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            else:
                data = json.loads(file_data)
            
            # Handle different JSON structures
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # Check if it's a single record or container
                if any(key in data for key in ['records', 'data', 'items']):
                    records = data.get('records', data.get('data', data.get('items', [data])))
                else:
                    records = [data]
            else:
                records = [data]
            
            self.logger.info("JSON file read successfully", records_count=len(records))
            
            return AgentResult(
                success=True,
                data=records,
                metadata={
                    "format": "json",
                    "records_count": len(records),
                    "structure": "array" if isinstance(data, list) else "object"
                }
            )
            
        except Exception as e:
            self.logger.error("Error reading JSON file", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"JSON reading error: {str(e)}"]
            )
    
    async def _read_xml(
        self,
        file_path: Optional[str],
        file_data: Any,
        context: Optional[Dict[str, Any]]
    ) -> AgentResult:
        """Read XML file."""
        try:
            if file_path:
                tree = ET.parse(file_path)
                root = tree.getroot()
            else:
                root = ET.fromstring(file_data)
            
            # Convert XML to list of dictionaries
            records = []
            record_tag = context.get("record_tag", "record")
            
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
            
            self.logger.info("XML file read successfully", records_count=len(records))
            
            return AgentResult(
                success=True,
                data=records,
                metadata={
                    "format": "xml",
                    "records_count": len(records),
                    "root_tag": root.tag
                }
            )
            
        except Exception as e:
            self.logger.error("Error reading XML file", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"XML reading error: {str(e)}"]
            )
    
    async def _read_fixed_width(
        self,
        file_path: Optional[str],
        file_data: Any,
        context: Optional[Dict[str, Any]]
    ) -> AgentResult:
        """Read fixed-width file."""
        try:
            # Get field specifications
            field_specs = context.get("field_specs", [])
            if not field_specs:
                return AgentResult(
                    success=False,
                    errors=["Field specifications required for fixed-width files"]
                )
            
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
            else:
                lines = file_data.split('\n')
            
            records = []
            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                
                record = {}
                for field_spec in field_specs:
                    field_name = field_spec['name']
                    start_pos = field_spec['start'] - 1  # Convert to 0-based
                    end_pos = field_spec['end']
                    
                    if end_pos <= len(line):
                        value = line[start_pos:end_pos].strip()
                        record[field_name] = value
                    else:
                        record[field_name] = ""
                
                records.append(record)
            
            self.logger.info("Fixed-width file read successfully", records_count=len(records))
            
            return AgentResult(
                success=True,
                data=records,
                metadata={
                    "format": "fixed_width",
                    "records_count": len(records),
                    "field_count": len(field_specs)
                }
            )
            
        except Exception as e:
            self.logger.error("Error reading fixed-width file", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"Fixed-width reading error: {str(e)}"]
            )
    
    async def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get file metadata information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict[str, Any]: File metadata
        """
        try:
            path = Path(file_path)
            
            metadata = {
                "file_name": path.name,
                "file_size": path.stat().st_size,
                "file_extension": path.suffix,
                "created_time": path.stat().st_ctime,
                "modified_time": path.stat().st_mtime,
                "exists": path.exists()
            }
            
            self.logger.info("File metadata retrieved", file_path=file_path)
            return metadata
            
        except Exception as e:
            self.logger.error("Error getting file metadata", error=str(e))
            return {"error": str(e)}
    
    async def validate_file_structure(self, data: List[Dict[str, Any]]) -> AgentResult:
        """
        Validate file structure and data consistency.
        
        Args:
            data: File data to validate
            
        Returns:
            AgentResult: Validation result
        """
        try:
            if not data:
                return AgentResult(
                    success=False,
                    errors=["No data to validate"]
                )
            
            # Check if all records have the same structure
            first_record_keys = set(data[0].keys())
            errors = []
            warnings = []
            
            for i, record in enumerate(data[1:], 1):
                record_keys = set(record.keys())
                
                # Check for missing keys
                missing_keys = first_record_keys - record_keys
                if missing_keys:
                    errors.append(f"Record {i}: Missing keys {missing_keys}")
                
                # Check for extra keys
                extra_keys = record_keys - first_record_keys
                if extra_keys:
                    warnings.append(f"Record {i}: Extra keys {extra_keys}")
            
            # Check for empty records
            empty_records = [i for i, record in enumerate(data) if not any(record.values())]
            if empty_records:
                warnings.append(f"Empty records found at positions: {empty_records}")
            
            success = len(errors) == 0
            
            self.logger.info(
                "File structure validation completed",
                success=success,
                errors_count=len(errors),
                warnings_count=len(warnings)
            )
            
            return AgentResult(
                success=success,
                data=data,
                errors=errors,
                warnings=warnings,
                metadata={
                    "total_records": len(data),
                    "field_count": len(first_record_keys),
                    "fields": list(first_record_keys)
                }
            )
            
        except Exception as e:
            return await self.handle_error(e, {"data_length": len(data) if data else 0})
