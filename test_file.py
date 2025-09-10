import json
import os
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def mainframe_to_json(data_file, layout_file, output_json="mainframe_output.json"):
    """Convert mainframe fixed-width file to JSON using LLM with layout file."""
    # Read the mainframe data file
    with open(data_file, "r", encoding="utf-8") as f:
        data_content = f.read()
    
    # Read the layout file
    with open(layout_file, "r", encoding="utf-8") as f:
        layout_content = f.read()

    # Create prompt for GPT
    prompt = f"""
    You are a mainframe data conversion expert. Convert the following fixed-width mainframe data file to structured JSON using the provided layout file.

    CRITICAL INSTRUCTIONS:
    1. Return ONLY valid JSON - no explanations, no markdown, no code blocks
    2. Do NOT wrap the JSON in ```json or ``` 
    3. Do NOT include any text before or after the JSON
    4. Start your response directly with {{ and end with }}
    5. Use the layout file to understand field positions and meanings
    6. Convert each record to a JSON object with meaningful field names
    7. Include metadata about the conversion (source files, record count, etc.)

    LAYOUT FILE CONTENT:
    {layout_content}

    MAINFRAME DATA FILE CONTENT:
    {data_content}

    Convert this to a JSON structure with:
    - Metadata (source_file, layout_file, total_records)
    - Array of records with properly named fields based on the layout
    - Each record should have all fields from the layout with appropriate data types
    """

    # Call GPT model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a mainframe data conversion expert. You MUST respond with only valid JSON. No explanations, no markdown formatting, no code blocks. Start with { and end with }."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,  # deterministic output
    )

    # Extract JSON text
    json_text = response.choices[0].message.content.strip()

    # Try parsing JSON to validate
    try:
        json_data = json.loads(json_text)
    except json.JSONDecodeError:
        raise ValueError("The model did not return valid JSON:\n" + json_text)

    # Save to file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"✅ Mainframe data converted to JSON: {output_json}")
    print(f"📊 Processed data from {data_file} using layout {layout_file}")
    return json_data

def file_to_json(file_path, output_json="output.json"):
    prompt = f"""
    Convert the following text/csv content into structured JSON.
    
    CRITICAL INSTRUCTIONS:
    1. Return ONLY valid JSON - no explanations, no markdown, no code blocks
    2. Do NOT wrap the JSON in ```json or ``` 
    3. Do NOT include any text before or after the JSON
    4. Start your response directly with {{ and end with }}
    5. Ensure the JSON is properly formatted and valid

    Content:
    {file_content}
    """

    # Call GPT model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a JSON converter. You MUST respond with only valid JSON. No explanations, no markdown formatting, no code blocks. Start with { and end with }."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,  # deterministic output
    )

    # Extract JSON text
    json_text = response.choices[0].message.content.strip()

    # Try parsing JSON to validate
    try:
        json_data = json.loads(json_text)
    except json.JSONDecodeError:
        raise ValueError("The model did not return valid JSON:\n" + json_text)

    # Save to file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"✅ JSON saved to {output_json}")
    return json_data


if __name__ == "__main__":
    # Example usage - choose one:
    
    # Option 1: Convert regular text file to JSON
    # input_file = "data/input/sample_member_data.txt"
    # result = file_to_json(input_file, "member_data_output.json")
    # print(json.dumps(result, indent=2))
    
    # Option 2: Convert mainframe fixed-width file to JSON
    data_file = "data/input/mainframe_member_data.dat"
    layout_file = "data/input/mainframe_member_layout.txt"
    result = mainframe_to_json(data_file, layout_file, "mainframe_output.json")
    print(json.dumps(result, indent=2))
