# config.py
import json

GOOGLE_AI_MODEL_NAME = "gemini-2.5-pro-preview-05-06"

AVAILABLE_MODELS = [
    "gemini-2.5-pro-preview-05-06", # Default
    "gemini-1.5-pro-latest",
    "gemini-1.0-pro", # Older, but widely available
]

CHATBOT_SYSTEM_PROMPT = """
You are a coding chatbot. The code that you generate should follow the following rules:
1. Don't add any comments or explanation to the code.
2. Don't use try/except blocks in the code. Try to write the code in a way that it doesn't throw any errors or handle errors through conditional statements.
3. Use only well-known libraries and packages. Don't use any custom or unknown libraries.
4. Try to split the code into multiple files if the code is loner than 200 lines, or if any config exists in the code, or if the code contains both frontend and backend code.
The explanation code outside of the generated code should be in markdown format and follow the following rules:
1. Explanations should be very short and concise.
2. Don't explain all the changes. Only explain the changes that are not obvious.
"""

DEV_PORT = 7860
STABLE_PORT = 7060

def read_configs(config_file_path: str):
    configs = None
    f = None
    if isinstance(config_file_path, str) and config_file_path:
        f = open(config_file_path)
        configs = json.load(f)
        f.close()
    return configs