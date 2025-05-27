"""
Google GenAI Utilities

This module provides functions to interact with Google's Generative AI models (Gemini)
using the `google-generativeai` SDK. It handles API key management, model selection
from a configuration file, multimodal input (text and images/files), and error handling.
"""

# Standard library imports
import os
import json
import logging # Added for logging

# Third-party imports
from google import genai as google_genai_sdk # Renamed for clarity
from google.generativeai import types as google_genai_types # For Part creation
from google.generativeai import errors as google_genai_errors # For specific GenAI errors

# --- Configuration Loading ---
CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')

DEFAULT_MODEL_CONFIG = {
    "google": {
        "default_model": "gemini-1.5-flash-latest", # Fallback text model
        "vision_model": "gemini-1.5-flash-latest" # Fallback vision model (Gemini 1.5 Flash can also handle vision)
    }
}

try:
    with open(CONFIG_FILE_PATH, 'r') as f:
        loaded_config = json.load(f)
    if "google" not in loaded_config or not isinstance(loaded_config["google"], dict):
        logging.warning(f"Google configuration missing or malformed in {CONFIG_FILE_PATH}. Using default Google model names.")
        MODEL_CONFIG = DEFAULT_MODEL_CONFIG
    else:
        MODEL_CONFIG = loaded_config
except FileNotFoundError:
    logging.warning(f"Warning: Configuration file {CONFIG_FILE_PATH} not found. Using default Google model names.")
    MODEL_CONFIG = DEFAULT_MODEL_CONFIG
except json.JSONDecodeError:
    logging.warning(f"Warning: Error decoding {CONFIG_FILE_PATH}. Using default Google model names.")
    MODEL_CONFIG = DEFAULT_MODEL_CONFIG

# Set model names from loaded config or defaults
# Ensure these keys exist in your config.json or they will use the fallback.
GOOGLE_DEFAULT_MODEL = MODEL_CONFIG.get("google", {}).get("default_model", DEFAULT_MODEL_CONFIG["google"]["default_model"])
GOOGLE_VISION_MODEL = MODEL_CONFIG.get("google", {}).get("vision_model", DEFAULT_MODEL_CONFIG["google"]["vision_model"])

# --- Main Function ---

def get_google_gemini_response(
    prompt: str,
    file_content: bytes = None,
    filename: str = None,
    mime_type: str = None
) -> str:
    """
    Gets a response from the Google Gemini API, supporting text and file inputs.

    Handles API key loading, model selection based on input type (text or vision-capable for images),
    and formats requests for the `google-generativeai` SDK.

    Args:
        prompt (str): The user's text prompt.
        file_content (bytes, optional): Binary content of the uploaded file.
        filename (str, optional): Name of the uploaded file. Required if `file_content` is provided.
        mime_type (str, optional): MIME type of the file (e.g., "image/jpeg", "text/plain").
                                   Crucial for determining if a vision model should be used
                                   and for the API to correctly interpret the file.

    Returns:
        str: The AI's response text, or an error message formatted as "Error: <details>".
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.error("GOOGLE_API_KEY not found in environment variables.")
        return "Error: GOOGLE_API_KEY not found in environment variables. Please set it in your .env file or system environment."

    try:
        # Configure the SDK with the API key.
        # The google.genai.configure() method is the standard way if you're not creating multiple clients.
        # However, client = genai.Client(api_key=api_key) is also valid if you prefer explicit client management.
        # For consistency with how the SDK examples often start, using configure():
        google_genai_sdk.configure(api_key=api_key)
    except Exception as e:
        logging.error(f"Error configuring Google GenAI SDK: {e}", exc_info=True)
        return f"Error: Failed to configure Google GenAI SDK - {str(e)}. Check API key and library installation."

    model_name_to_use = GOOGLE_DEFAULT_MODEL
    input_contents = [] # This will hold the parts for the GenAI API call

    if file_content:
        if not filename or not mime_type:
            logging.warning("File content provided to Google Gemini without filename or MIME type.")
            # It's often problematic to proceed without mime_type for Gemini.
            # Depending on strictness, could return an error here.
            # For now, we'll try to treat it as a generic blob if possible, but it's not ideal.
            # Gemini typically requires MIME type for blobs.
            return "Error: File content provided without filename or MIME type. Both are required for Google Gemini."

        # If the MIME type indicates an image, switch to the vision-capable model.
        # The `gemini-1.5-flash-latest` and `gemini-1.5-pro-latest` can handle both text and vision.
        # So, explicitly switching to GOOGLE_VISION_MODEL might only be necessary if it's a different
        # model specialized *only* for vision (like older "gemini-pro-vision" if that was distinct).
        # Given modern Gemini models, GOOGLE_DEFAULT_MODEL might often be the same as GOOGLE_VISION_MODEL
        # if they are set to "gemini-X.Y-flash/pro-latest".
        if 'image' in mime_type.lower():
            model_name_to_use = GOOGLE_VISION_MODEL # Use the configured vision model
            logging.info(f"Using Google Vision Model ({model_name_to_use}) due to image MIME type: {mime_type}")
            # For vision models, create a Part for the image data and one for the prompt.
            image_part = google_genai_types.Part.from_bytes(data=file_content, mime_type=mime_type)
            # The prompt should accompany the image.
            input_contents = [prompt, image_part] # Order can matter: often text first, then image.
        else:
            # For non-image files, attempt to decode as text and append to the prompt.
            # This makes the file content part of the textual input to the standard model.
            logging.info(f"Treating file '{filename}' as text for Google Gemini model {model_name_to_use}.")
            try:
                # Limit snippet size to avoid overly large prompts.
                file_text_snippet = file_content[:50000].decode(errors='replace') # 'replace' to handle decoding errors gracefully
                enhanced_prompt = (
                    f"{prompt}\n\n"
                    f"--- User Uploaded File: {filename} (MIME type: {mime_type}) ---\n"
                    f"{file_text_snippet}\n"
                    f"--- End of File Content ---"
                )
                input_contents = [enhanced_prompt]
            except Exception as e:
                logging.error(f"Error decoding file '{filename}' as text for Google Gemini: {e}", exc_info=True)
                return f"Error: Failed to decode file content for {filename} as text. Details: {str(e)}"
    else:
        # No file, just a text prompt
        input_contents = [prompt]

    try:
        # Initialize the generative model with the chosen model name.
        logging.info(f"Sending request to Google Gemini model: {model_name_to_use} with prompt length: {len(prompt)}")
        model = google_genai_sdk.GenerativeModel(model_name=model_name_to_use)
        
        # Generate content. The `contents` argument should be a list of Parts or strings.
        response = model.generate_content(contents=input_contents)

        # --- Response Handling ---
        # 1. Check for immediate blocking reasons in prompt_feedback
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason_name = response.prompt_feedback.block_reason.name
            safety_ratings_str = ", ".join([f"{sr.category.name}: {sr.probability.name}" for sr in response.prompt_feedback.safety_ratings])
            logging.warning(f"Google API blocked the prompt. Reason: {block_reason_name}. Safety: {safety_ratings_str}")
            return (f"Error: Prompt blocked by Google API. Reason: {block_reason_name}. "
                    f"Safety Ratings: [{safety_ratings_str}]")

        # 2. Check if the response text itself is empty or missing (can also indicate blocking or other issues)
        # Accessing response.text can raise an exception if the response was blocked.
        # google.generativeai.types.generation_types.GenerateContentResponse.text
        #  Raises: ValueError if the response does not contain text. (e.g. was blocked).
        #  See also: `prompt_feedback`.
        try:
            generated_text = response.text
            if not generated_text: # Check if text is empty string
                 logging.warning("Google Gemini returned an empty response text but was not explicitly blocked via prompt_feedback.")
                 # This could happen if the model genuinely has nothing to say or if there's a more subtle issue.
                 # Check safety ratings again, as sometimes blocking occurs without block_reason but shows in ratings.
                 if response.candidates and response.candidates[0].finish_reason.name != "STOP":
                     finish_reason = response.candidates[0].finish_reason.name
                     safety_ratings_str = ", ".join([f"{sr.category.name}: {sr.probability.name}" for sr in response.candidates[0].safety_ratings])
                     return f"Error: Google Gemini returned no content. Finish Reason: {finish_reason}. Safety: [{safety_ratings_str}]"
                 return "Error: Received an empty response from Google Gemini. The prompt may have been too vague or resulted in no actionable output."
            return generated_text.strip()
        except ValueError as ve: # Raised if response.text is accessed on a blocked prompt
            logging.warning(f"ValueError accessing response.text (likely blocked): {ve}. Prompt feedback: {response.prompt_feedback}")
            # Fallback to a generic message if specific block reason wasn't caught above
            # (though the first check should usually catch it).
            return "Error: Content generation failed. The prompt was likely blocked by Google API due to safety filters or other policy reasons. Please review prompt_feedback for details if available."


    except google_genai_errors.InvalidArgumentError as e: # Specific error for invalid arguments
        logging.error(f"Google API Invalid Argument Error: {e}", exc_info=True)
        return f"Error: Google API Invalid Argument - {str(e)}. This often means an issue with the request structure (e.g. malformed `contents` or invalid `mime_type`)."
    except google_genai_errors.GoogleAPIError as e: # More general Google API error
        logging.error(f"Google API Error: {e} (Type: {e.__class__.__name__})", exc_info=True)
        error_message = f"Error: Google API Error - {str(e)} (Type: {e.__class__.__name__})."
        # Attempt to provide more specific advice based on common error patterns
        if isinstance(e, google_genai_errors.PermissionDenied): # Inherits from GoogleAPIError
            error_message += " This indicates an issue with your GOOGLE_API_KEY (e.g., invalid, disabled, or missing necessary permissions for the Gemini API). Please verify your API key and ensure the 'Generative Language API' or 'Vertex AI API' (if using Vertex) is enabled in your Google Cloud Project."
        elif isinstance(e, google_genai_errors.ResourceExhausted): # Inherits from GoogleAPIError
             error_message += " You may have exceeded your API quota (Rate Limit). Please check your usage and limits in the Google Cloud Console."
        elif "model" in str(e).lower() and ("not found" in str(e).lower() or "could not find" in str(e).lower()):
             error_message += f" The requested model ('{model_name_to_use}') might not be found, not available for your project/region, or the name is misspelled."
        return error_message
    except Exception as e: # Catch-all for any other unexpected exceptions
        logging.error(f"An unexpected error occurred with Google Gemini: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while interacting with Google Generative AI - {str(e)} (Type: {e.__class__.__name__})"