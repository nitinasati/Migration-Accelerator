"""
File Reader Agent for reading mainframe files.
"""

import os
import pandas as pd
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from agents.base_agent import BaseAgent, AgentRole
from llm.providers import LLMProviderBase
from mcp.client import MCPToolManager
from config.settings import FileFormat


class FileReaderAgent(BaseAgent):
    """Agent responsible for reading and parsing mainframe files."""
    
    def __init__(
        self,
        name: str = "FileReaderAgent",
        llm_provider: Optional[LLMProviderBase] = None,
        mcp_manager: Optional[MCPToolManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name=name,
            role=AgentRole.FILE_READER,
            llm_provider=llm_provider,
            mcp_manager=mcp_manager,
            config=config
        )
        
        # Supported file formats
        self.supported_formats = {
            FileFormat.CSV: self._read_csv,
            FileFormat.EXCEL: self._read_excel,
            FileFormat.JSON: self._read_json,
            FileFormat.XML: self._read_xml,
            FileFormat.FIXED_WIDTH: self._read_fixed_width
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process file reading request."""
        file_path = input_data.get("file_path")
        file_format = input_data.get("file_format")
        record_type = input_data.get("record_type", "unknown")
        
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.logger.info("Starting file reading", file_path=file_path, format=file_format)
        
        # Detect file format if not provided
        if not file_format:
            file_format = self._detect_file_format(file_path)
        
        # Read the file
        records = await self._read_file(file_path, file_format)
        
        # Enhance with LLM if available
        if self.llm_provider:
            enhanced_records = await self._enhance_records(records, record_type)
        else:
            enhanced_records = records
        
        # Update metrics
        self.metrics.records_processed = len(enhanced_records)
        
        result = {
            "file_path": file_path,
            "file_format": file_format,
            "record_type": record_type,
            "total_records": len(enhanced_records),
            "records": enhanced_records,
            "file_analysis": await self._analyze_file(enhanced_records, file_format),
            "enhanced": self.llm_provider is not None
        }
        
        self.logger.info(
            "File reading completed",
            total_records=len(enhanced_records),
            file_format=file_format
        )
        
        return result
    
    def _detect_file_format(self, file_path: str) -> str:
        """Detect file format based on extension and content."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Extension-based detection
        if extension == '.csv':
            return FileFormat.CSV
        elif extension in ['.xlsx', '.xls']:
            return FileFormat.EXCEL
        elif extension == '.json':
            return FileFormat.JSON
        elif extension == '.xml':
            return FileFormat.XML
        elif extension in ['.txt', '.dat']:
            # Try to detect fixed-width format
            return FileFormat.FIXED_WIDTH
        
        # Content-based detection
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
                if first_line.startswith('{') or first_line.startswith('['):
                    return FileFormat.JSON
                elif first_line.startswith('<'):
                    return FileFormat.XML
                elif ',' in first_line:
                    return FileFormat.CSV
                else:
                    return FileFormat.FIXED_WIDTH
        except Exception:
            return FileFormat.CSV  # Default fallback
    
    async def _read_file(self, file_path: str, file_format: str) -> List[Dict[str, Any]]:
        """Read file based on format."""
        if file_format not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        reader_func = self.supported_formats[file_format]
        return await reader_func(file_path)
    
    async def _read_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Read CSV file."""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            return df.to_dict('records')
        except UnicodeDecodeError:
            # Try with different encoding
            df = pd.read_csv(file_path, encoding='latin-1')
            return df.to_dict('records')
    
    async def _read_excel(self, file_path: str) -> List[Dict[str, Any]]:
        """Read Excel file."""
        df = pd.read_excel(file_path)
        return df.to_dict('records')
    
    async def _read_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Read JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both array and object formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError("Invalid JSON format")
    
    async def _read_xml(self, file_path: str) -> List[Dict[str, Any]]:
        """Read XML file."""
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Convert XML to list of dictionaries
        records = []
        for child in root:
            record = {}
            for elem in child.iter():
                if elem != child:  # Skip the root element
                    record[elem.tag] = elem.text
            if record:
                records.append(record)
        
        return records
    
    async def _read_fixed_width(self, file_path: str) -> List[Dict[str, Any]]:
        """Read fixed-width file."""
        records = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            return records
        
        # Try to detect field widths from the first few lines
        field_widths = self._detect_field_widths(lines[:10])
        
        for line in lines:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            
            record = {}
            start_pos = 0
            
            for i, width in enumerate(field_widths):
                field_name = f"field_{i+1}"
                field_value = line[start_pos:start_pos + width].strip()
                record[field_name] = field_value
                start_pos += width
            
            records.append(record)
        
        return records
    
    def _detect_field_widths(self, lines: List[str]) -> List[int]:
        """Detect field widths in fixed-width files."""
        if not lines:
            return []
        
        # Simple heuristic: look for consistent spacing patterns
        max_length = max(len(line) for line in lines)
        
        # Try common field widths
        common_widths = [10, 15, 20, 25, 30]
        
        for width in common_widths:
            if max_length % width == 0:
                return [width] * (max_length // width)
        
        # Fallback: assume equal-width fields
        num_fields = 10  # Default assumption
        field_width = max_length // num_fields
        return [field_width] * num_fields
    
    async def _enhance_records(self, records: List[Dict[str, Any]], record_type: str) -> List[Dict[str, Any]]:
        """Enhance records using LLM."""
        if not self.llm_provider or not records:
            return records
        
        enhanced_records = []
        
        # Process records in batches
        batch_size = self.config.get("batch_size", 10)
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            # Create enhancement prompt
            prompt = f"""
You are an expert in analyzing {record_type} insurance data. Enhance the following records by:
1. Identifying missing fields and suggesting default values
2. Validating data consistency
3. Adding derived fields where appropriate
4. Flagging potential data quality issues

Records to enhance:
{json.dumps(batch, indent=2)}

Please provide enhanced records in JSON format, maintaining the original structure while adding:
- "_enhanced" field with enhancement details
- "_quality_score" field (0-1)
- "_issues" field with any data quality issues found
"""
            
            try:
                response = await self.llm_provider.generate(prompt)
                self.metrics.llm_calls += 1
                
                # Parse enhanced records
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    enhanced_batch = json.loads(json_match.group())
                    enhanced_records.extend(enhanced_batch)
                else:
                    # Fallback to original records
                    enhanced_records.extend(batch)
                    
            except Exception as e:
                self.logger.warning("LLM enhancement failed", error=str(e))
                enhanced_records.extend(batch)
        
        return enhanced_records
    
    async def _analyze_file(self, records: List[Dict[str, Any]], file_format: str) -> Dict[str, Any]:
        """Analyze file content and provide insights."""
        if not records:
            return {"error": "No records to analyze"}
        
        analysis = {
            "file_format": file_format,
            "record_count": len(records),
            "field_names": list(records[0].keys()) if records else [],
            "data_quality_score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        # Basic data quality analysis
        if records:
            # Check for missing values
            missing_values = {}
            for field in analysis["field_names"]:
                missing_count = sum(1 for record in records if not record.get(field))
                if missing_count > 0:
                    missing_values[field] = missing_count
            
            if missing_values:
                analysis["issues"].append(f"Missing values found: {missing_values}")
            
            # Calculate quality score
            total_fields = len(analysis["field_names"]) * len(records)
            filled_fields = sum(
                sum(1 for field in analysis["field_names"] if record.get(field))
                for record in records
            )
            
            analysis["data_quality_score"] = filled_fields / total_fields if total_fields > 0 else 0.0
        
        # Add recommendations
        if analysis["data_quality_score"] < 0.8:
            analysis["recommendations"].append("Consider data quality improvement")
        
        if len(analysis["field_names"]) < 5:
            analysis["recommendations"].append("File may be missing important fields")
        
        return analysis
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        required_fields = ["file_path"]
        
        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        file_path = input_data["file_path"]
        if not os.path.exists(file_path):
            self.logger.error(f"File does not exist: {file_path}")
            return False
        
        return True
