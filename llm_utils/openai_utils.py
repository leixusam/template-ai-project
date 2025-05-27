"""
OpenAI LLM Utilities

This module provides functions to interact with OpenAI's language models,
specifically for chat completions. It handles API key management, model selection
from a configuration file, and error handling.
"""

# Standard library imports
import os
import json
import logging # Added for logging

# Third-party imports
import openai # OpenAI Python SDK v1.x.x

# --- Configuration Loading ---
# Construct the absolute path to config.json relative to this file's location
# (llm_utils/openai_utils.py -> project_root/config.json)
CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')

# Default model configuration as a fallback
DEFAULT_MODEL_CONFIG = {
    "openai": {
        "default_model": "gpt-3.5-turbo", # A widely available and capable model
        "vision_model": "gpt-4-turbo" # Or gpt-4o if preferred and available
    }
}

# Attempt to load model configurations from config.json
try:
    with open(CONFIG_FILE_PATH, 'r') as f:
        loaded_config = json.load(f)
    if "openai" not in loaded_config or not isinstance(loaded_config["openai"], dict):
        logging.warning(f"OpenAI configuration missing or malformed in {CONFIG_FILE_PATH}. Using default OpenAI model names.")
        MODEL_CONFIG = DEFAULT_MODEL_CONFIG
    else:
        MODEL_CONFIG = loaded_config
except FileNotFoundError:
    logging.warning(f"Warning: Configuration file {CONFIG_FILE_PATH} not found. Using default OpenAI model names.")
    MODEL_CONFIG = DEFAULT_MODEL_CONFIG
except json.JSONDecodeError:
    logging.warning(f"Warning: Error decoding {CONFIG_FILE_PATH}. Using default OpenAI model names.")
    MODEL_CONFIG = DEFAULT_MODEL_CONFIG

# Set model names from loaded config or defaults
OPENAI_DEFAULT_MODEL = MODEL_CONFIG.get("openai", {}).get("default_model", DEFAULT_MODEL_CONFIG["openai"]["default_model"])
# The vision model is not directly used in the current get_openai_chat_response,
# as it primarily handles text and provides a placeholder for file content.
# For true vision capabilities (e.g., image analysis), the API usage would be different (e.g. gpt-4o with image inputs).
# OPENAI_VISION_MODEL = MODEL_CONFIG.get("openai", {}).get("vision_model", DEFAULT_MODEL_CONFIG["openai"]["vision_model"])

# --- Main Function ---

def get_openai_chat_response(prompt: str, file_content: bytes = None, filename: str = None) -> str:
    """
    Gets a chat response from an OpenAI model (e.g., gpt-3.5-turbo, gpt-4, gpt-4o).

    This function uses the OpenAI SDK (v1.x.x) which expects the API key to be set
    as an environment variable `OPENAI_API_KEY`.

    If `file_content` is provided, a system message is prepended to the chat,
    informing the model about the uploaded file and including a snippet of its content.
    This is a basic way to provide file context and may have limitations for very large
    files or specific multimodal tasks which might require different API endpoints or methods
    (e.g. Assistants API, or direct image input with models like GPT-4o).

    Args:
        prompt (str): The user's prompt for the AI.
        file_content (bytes, optional): The binary content of an uploaded file. Defaults to None.
        filename (str, optional): The name of the uploaded file. Defaults to None.

    Returns:
        str: The AI's response text, or an error message formatted as "Error: <details>".
    """
    # API Key Check: The OpenAI SDK (v1.x.x) automatically attempts to load the API key
    # from the OPENAI_API_KEY environment variable. If it's not set, openai.OpenAI()
    # or subsequent API calls will raise an AuthenticationError.
    # We add an explicit check for a more user-friendly initial error message.
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY not found in environment variables.")
        return "Error: OPENAI_API_KEY not found in environment variables. Please set it in your .env file or system environment."

    messages = []
    if file_content and filename:
        # System message to inform the model about the file.
        # This approach is suitable for providing textual context from a file.
        # For models like GPT-4o that can directly process images, the 'messages'
        # format would involve image URLs or base64 encoded images.
        # The current implementation keeps it simple by treating file content as text.
        try:
            file_text_snippet = file_content[:2000].decode(errors='ignore') # Increased snippet size
        except Exception as e:
            logging.error(f"Error decoding file content for OpenAI: {e}")
            return f"Error: Could not decode file content for OpenAI. Ensure it's a text-based file or handle binary data appropriately. Details: {e}"

        messages.append({
            "role": "system",
            "content": (
                f"The user has uploaded a file named '{filename}'. "
                f"Its content (first 2000 characters) is: {file_text_snippet}"
            )
        })
    elif file_content and not filename:
        logging.warning("File content provided to OpenAI without a filename.")
        # Decide if this is an error or if a default filename should be used.
        # For now, we'll still try to include it.
        try:
            file_text_snippet = file_content[:2000].decode(errors='ignore')
        except Exception as e:
            logging.error(f"Error decoding file content without filename for OpenAI: {e}")
            return f"Error: Could not decode file content for OpenAI. Details: {e}"
        messages.append({
            "role": "system",
            "content": f"The user has uploaded a file (name not provided). Its content (first 2000 characters) is: {file_text_snippet}"
        })

    messages.append({"role": "user", "content": prompt})

    try:
        # Initialize the OpenAI client.
        # The client automatically picks up the API key from the OPENAI_API_KEY environment variable.
        client = openai.OpenAI()

        logging.info(f"Sending request to OpenAI model: {OPENAI_DEFAULT_MODEL} with prompt length: {len(prompt)}")
        response = client.chat.completions.create(
            model=OPENAI_DEFAULT_MODEL, # Use the model name loaded from config.json or default
            messages=messages,
            max_tokens=1024 # Adjusted for potentially more comprehensive responses
        )
        
        if response.choices and response.choices[0].message:
            bot_response = response.choices[0].message.content
            if bot_response:
                return bot_response.strip()
            else:
                logging.warning("OpenAI response was empty.")
                return "Error: OpenAI returned an empty response."
        else:
            logging.error(f"Invalid response structure from OpenAI: {response}")
            return "Error: Received an invalid or incomplete response structure from OpenAI."

    except openai.APIConnectionError as e:
        logging.error(f"OpenAI API Connection Error: {e}")
        return f"Error: OpenAI API Connection Error - {e}"
    except openai.RateLimitError as e:
        logging.error(f"OpenAI API Rate Limit Exceeded: {e}")
        return f"Error: OpenAI API Rate Limit Exceeded - {e}"
    except openai.AuthenticationError as e:
        # This error is also caught by the initial os.getenv("OPENAI_API_KEY") check,
        # but it's good to have it here as a fallback if the SDK raises it directly.
        logging.error(f"OpenAI API Authentication Error: {e}. Check your API key.")
        return f"Error: OpenAI API Authentication Error - {e}. Ensure your OPENAI_API_KEY is correct and active."
    except openai.APIError as e: # General API error
        logging.error(f"OpenAI API Error: {e} (Status Code: {e.status_code}, Type: {e.type})")
        return f"Error: OpenAI API Error - {e.message} (Status Code: {e.status_code}, Type: {e.type})"
    except Exception as e:
        logging.error(f"An unexpected error occurred with OpenAI: {e}", exc_info=True) # Log stack trace
        return f"Error: An unexpected error occurred while interacting with OpenAI - {str(e)}"