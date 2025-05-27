# Standard library imports
import os
import sys
from typing import Optional

# Third-party imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Local application/library specific imports
# Add project root to sys.path to allow importing modules from the root directory (e.g., llm_utils)
# This is necessary because the backend is a sub-directory and Python's default import mechanism
# might not find modules in the parent directory without this modification.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from llm_utils import get_openai_chat_response, get_google_gemini_response

# --- Application Setup ---

# Load environment variables from .env file located at the project root
# This allows for secure management of API keys and other configurations.
DOTENV_PATH = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(DOTENV_PATH)

# Initialize FastAPI application
app = FastAPI(title="AI Text Generation API", version="0.1.0")

# Configure CORS (Cross-Origin Resource Sharing) middleware
# This allows the frontend (running on a different origin, e.g., http://localhost:3000)
# to make requests to this backend API.
# In a production environment, allow_origins should be restricted to the specific frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "null"],  # "null" for local file:// access (e.g. opening index.html directly)
    allow_credentials=True,  # Allows cookies to be included in requests
    allow_methods=["POST"],  # Restrict to POST method for the /api/generate endpoint
    allow_headers=["*"],  # Allows all headers; can be restricted for security
)

# --- Helper Functions ---

def _check_api_key(provider: str) -> None:
    """
    Checks if the API key for the specified provider is available in environment variables.
    Raises HTTPException if the key is not found.
    """
    key_name = ""
    if provider == "openai":
        key_name = "OPENAI_API_KEY"
    elif provider == "google":
        key_name = "GOOGLE_API_KEY"
    else:
        # This case should ideally not be reached if provider is validated before calling this
        return 

    if not os.getenv(key_name):
        error_msg = f"{key_name} not configured. Please set it in the .env file."
        print(f"API Key Error: {error_msg}") # Server-side logging
        raise HTTPException(status_code=500, detail=error_msg)

# --- API Endpoints ---

@app.post("/api/generate")
async def generate_text(
    provider: str = Form(...),  # AI provider ('openai' or 'google')
    text: str = Form(...),  # User's input text/prompt
    file: Optional[UploadFile] = File(None)  # Optional file upload for multimodal input
):
    """
    Endpoint to generate text using a specified AI provider.
    It can optionally accept a file for multimodal interactions.
    """
    print(f"Received request: Provider='{provider}', Text Length='{len(text)}', File='{file.filename if file else 'None'}'")

    # Validate provider first
    if provider not in ["openai", "google"]:
        print(f"Validation Error: Invalid provider '{provider}' specified.")
        raise HTTPException(status_code=400, detail="Invalid AI provider specified. Choose 'openai' or 'google'.")

    # Check for necessary API keys before proceeding
    _check_api_key(provider)

    # --- File Handling Logic ---
    file_content: Optional[bytes] = None
    filename: Optional[str] = None
    mime_type: Optional[str] = None

    if file:
        try:
            filename = file.filename
            mime_type = file.content_type
            file_content = await file.read() # Read file content
            print(f"File processed: Name='{filename}', Type='{mime_type}', Size='{len(file_content)} bytes'")
        except Exception as e:
            print(f"Error reading uploaded file '{filename}': {e}")
            raise HTTPException(status_code=400, detail=f"Error processing uploaded file: {str(e)}")
        finally:
            await file.close() # Ensure file is closed

    # --- AI Model Interaction ---
    response_text = ""
    try:
        if provider == "openai":
            response_text = get_openai_chat_response(prompt=text, file_content=file_content, filename=filename)
        elif provider == "google":
            # Google Gemini requires mime_type for file inputs
            response_text = get_google_gemini_response(prompt=text, file_content=file_content, filename=filename, mime_type=mime_type)
        
        # Check if the utility functions returned an error (indicated by "Error:" prefix)
        if isinstance(response_text, str) and response_text.startswith("Error:"):
            # Log the error from the LLM utility and relay it to the client
            print(f"LLM Utility Error ({provider}): {response_text}")
            # The error message from the util is assumed to be user-friendly enough.
            raise HTTPException(status_code=500, detail=response_text)
            
        print(f"Successfully generated response using {provider}.")
        return {"response": response_text}

    except HTTPException as e:
        # Re-raise HTTPExceptions that were thrown intentionally (e.g., API key missing, invalid provider, LLM util error)
        raise e 
    except Exception as e:
        # Catch any other unexpected errors during the AI interaction process
        print(f"Unexpected backend error during AI call ({provider}): {e}")
        # Provide a generic error message to the client for unexpected issues
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred while contacting the AI provider: {str(e)}")

# --- Main Block (for direct execution) ---
if __name__ == "__main__":
    import uvicorn
    print("--- Starting FastAPI Development Server ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Loading .env from: {DOTENV_PATH}")
    
    # For debugging: Check if API keys are loaded (without printing the keys themselves)
    # This helps quickly diagnose .env loading issues.
    print(f"OpenAI Key Loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No. Ensure OPENAI_API_KEY is in .env'}")
    print(f"Google Key Loaded: {'Yes' if os.getenv('GOOGLE_API_KEY') else 'No. Ensure GOOGLE_API_KEY is in .env'}")
    
    uvicorn.run(
        "main:app",          # App entry point (module_name:app_instance_name)
        host="0.0.0.0",      # Listen on all available network interfaces
        port=8000,           # Standard port for the backend
        reload=True          # Enable auto-reload for development (watches for file changes)
    )
    print("--- FastAPI Development Server Stopped ---")