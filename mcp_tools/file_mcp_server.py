"""
MCP Server with fastmcp
Tools:
1. analyze_file_metadata_tool
2. validate_file_tool
3. read_file_tool

Run with: python your_mcp_server.py
"""

import os
import json
import datetime
import mimetypes
from typing import Dict, Any
from fastmcp import FastMCP

app = FastMCP("file_reader_server")


# ------------------------
# Utility function
# ------------------------

def _get_file_metadata(file_path: str) -> Dict[str, Any]:
    """Collect file metadata safely."""
    path = os.path.abspath(file_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    stats = os.stat(path)
    mime_type, _ = mimetypes.guess_type(path)

    return {
        "file_path": path,
        "exists": True,
        "is_directory": os.path.isdir(path),
        "file_extension": os.path.splitext(path)[1].lower(),
        "mime_type": mime_type or "unknown",
        "size_bytes": stats.st_size,
        "size_mb": round(stats.st_size / (1024 * 1024), 2),
        "created_time": datetime.datetime.fromtimestamp(stats.st_ctime).isoformat(),
        "modified_time": datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
    }


# ------------------------
# Tool Implementations
# ------------------------

@app.tool()
def analyze_file_metadata_tool(file_path: str) -> Dict[str, Any]:
    """Analyze file metadata including type, size, and timestamps."""
    return _get_file_metadata(file_path)


@app.tool()
def validate_file_tool(file_path: str, allowed_extensions: list, max_size_mb: float) -> Dict[str, Any]:
    """Validate file extension and size based on thresholds."""
    metadata = _get_file_metadata(file_path)
    extension_valid = metadata["file_extension"] in [ext.lower() for ext in allowed_extensions]
    size_valid = metadata["size_mb"] <= max_size_mb

    return {
        "file": metadata["file_path"],
        "extension_valid": extension_valid,
        "size_valid": size_valid,
        "is_valid": extension_valid and size_valid,
        "metadata": metadata
    }


@app.tool()
def read_file_tool(file_path: str, max_lines: int = 1000) -> Dict[str, Any]:
    """Read CSV, JSON, TXT, or DAT file and return contents as JSON."""
    metadata = _get_file_metadata(file_path)
    ext = metadata["file_extension"]
    result: Dict[str, Any] = {"file": metadata["file_path"], "data": None, "metadata": metadata}

    try:
        if ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                result["data"] = json.load(f)

        elif ext in [".csv", ".txt", ".dat"]:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                result["data"] = [line.strip() for line in lines[:max_lines]]

        else:
            result["data"] = f"Unsupported file type: {ext}"

    except Exception as e:
        result["error"] = str(e)

    return result


@app.tool()
def parse_json_output_tool(text_output: str = "", content: str = "", text: str = "", output: str = "") -> Dict[str, Any]:
    """
    Parse JSON from text output (useful for extracting structured data from LLM responses).
    
    Args:
        text_output: The main text to parse (primary parameter)
        content: Alternative parameter name for text content
        text: Alternative parameter name for text
        output: Alternative parameter name for output
    
    Returns:
        Dict with success status and parsed JSON or error message
    """
    # Try different parameter names to be flexible with LangGraph calls
    input_text = text_output or content or text or output
    
    if not input_text:
        return {"success": False, "error": "No text provided. Please provide text via text_output, content, text, or output parameter."}
        
    try:
        # Try direct JSON parsing first (for clean output)
        input_text = input_text.strip()
        if input_text.startswith('{') and input_text.endswith('}'):
            parsed_json = json.loads(input_text)
            return {"success": True, "parsed_json": parsed_json}
        
        # If there's extra text, find the JSON block
        start = input_text.find('{')
        end = input_text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = input_text[start:end]
            parsed_json = json.loads(json_str)
            return {"success": True, "parsed_json": parsed_json}
            
        return {"success": False, "error": "No valid JSON found in text output"}
        
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parsing error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


# ------------------------
# Entry point
# ------------------------

if __name__ == "__main__":
    app.run()
