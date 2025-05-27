# AI Multi-Tool Template

This project is a foundational template for building AI-powered applications. It includes a Python FastAPI backend, a simple HTML/JavaScript frontend, and utility modules for interacting with OpenAI and Google Generative AI models. The interface allows users to send text prompts and optional files to their chosen AI provider and view the generated response.

## Project Structure

The project is organized as follows:

```plaintext
.
├── backend/
│   └── main.py         # FastAPI application for handling API requests.
├── frontend/
│   └── index.html      # Single-page HTML frontend with Tailwind CSS and JavaScript.
├── llm_utils/
│   ├── __init__.py     # Makes the llm_utils directory a Python package.
│   ├── openai_utils.py # Utilities for OpenAI API interaction.
│   └── google_utils.py # Utilities for Google Generative AI API interaction.
├── .gitignore          # Specifies intentionally untracked files that Git should ignore.
├── config.json         # Centralized configuration for AI model names.
├── README.md           # This file: provides an overview and instructions.
├── requirements.txt    # Lists Python dependencies for the backend.
└── .env                # For API keys (you need to create this file manually).
```

## Features

-   **Dual AI Provider Support:** Seamlessly switch between OpenAI (e.g., GPT-4o, GPT-3.5-turbo) and Google (e.g., Gemini 1.5 Flash, Gemini Pro) models.
-   **Configurable Models:** AI model names are specified in `config.json`, allowing easy updates without code changes. Fallbacks are in place if the configuration is missing or invalid.
-   **File Uploads:** Supports including file content with prompts:
    -   **OpenAI:** File content (text snippet) is added to the system message for context.
    -   **Google Gemini:**
        -   Uses the configured vision model (e.g., `gemini-1.5-flash-latest`) if an image MIME type is detected.
        -   For other file types, text content is extracted and appended to the user's prompt.
-   **Simple Web Interface:** Built with HTML, Tailwind CSS for styling, and vanilla JavaScript for interactivity.
-   **FastAPI Backend:** A robust and efficient Python backend using FastAPI, including detailed error handling and API key management.
-   **Modular LLM Utilities:** Well-commented and refactored utility functions in `llm_utils` for clear and maintainable AI interactions, including robust error reporting.
-   **Environment-based API Key Management:** Securely manages API keys using a `.env` file at the project root.
-   **Comprehensive Error Handling:** Both frontend and backend provide informative error messages for API issues, network problems, and missing configurations.

## Setup Instructions

Follow these steps to set up and run the project:

1.  **Clone or Download Project:**
    If this were a Git repository, you would clone it. Otherwise, ensure all project files are downloaded and organized in a local directory.

2.  **Create Python Virtual Environment:**
    Using a virtual environment is crucial for managing dependencies and isolating project environments.
    Open your terminal or command prompt and navigate to the project's root directory.
    ```bash
    # For Python 3
    python3 -m venv venv
    # Or, if 'python' defaults to Python 3
    python -m venv venv
    ```

3.  **Activate Virtual Environment:**
    -   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    -   **Windows (Command Prompt/PowerShell):**
        ```bash
        venv\Scripts\activate
        ```
    Your terminal prompt should change to indicate the active virtual environment (e.g., `(venv)`).

4.  **Install Dependencies:**
    With the virtual environment active, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set Up API Keys:**
    Create a file named `.env` in the **root** of your project directory (the same directory as `requirements.txt` and the `backend` folder). This file will store your secret API keys.

    Add your API keys to the `.env` file as follows:
    ```env
    OPENAI_API_KEY="sk-your_openai_api_key_here"
    GOOGLE_API_KEY="your_google_api_key_here"
    ```
    -   Replace `sk-your_openai_api_key_here` with your actual API key from the [OpenAI Platform](https://platform.openai.com/account/api-keys).
    -   Replace `your_google_api_key_here` with your actual API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

    **Important:** Do not commit the `.env` file to version control if you are using Git. The `.gitignore` file is already configured to ignore it.

## Running the Application

1.  **Start the Backend Server:**
    Ensure your virtual environment is activated. In your terminal, from the project root directory, run:
    ```bash
    python backend/main.py
    ```
    The backend server will start, typically listening on `http://127.0.0.1:8000`. The console output will confirm the address. The server uses `uvicorn` with `reload=True`, so changes to `backend/main.py` should automatically restart the server during development.

2.  **Open the Frontend:**
    Navigate to the `frontend` directory and open the `index.html` file directly in your preferred web browser (e.g., by double-clicking it or using "File > Open" in the browser menu).

3.  **Interact with the AI:**
    -   The web page will load, allowing you to select an AI provider (OpenAI or Google).
    -   Enter your prompt in the text area.
    -   Optionally, upload a file to provide additional context.
    -   Click "Send Prompt". The AI's response will appear below the form. If any errors occur, they will be displayed in a designated error section.

## LLM Utilities (`llm_utils`)

The `llm_utils` directory contains modules for interacting with the different AI providers. Model names are loaded from `config.json`, with fallbacks to default models if the file is missing or misconfigured. Both utility files now include logging for warnings and errors.

-   **`openai_utils.py` (`get_openai_chat_response`)**:
    -   Interacts with OpenAI chat completion models (e.g., `gpt-4o`, `gpt-3.5-turbo`).
    -   Loads the `OPENAI_API_KEY` from the environment.
    -   Handles file inputs by creating a system message that includes the filename and a snippet of its decoded text content.
    -   Provides detailed error handling for various API exceptions (connection, rate limit, authentication, etc.), returning messages in the "Error: <details>" format.

-   **`google_utils.py` (`get_google_gemini_response`)**:
    -   Interacts with Google Gemini models (e.g., `gemini-1.5-flash-latest`, `gemini-1.5-pro-latest`).
    -   Loads the `GOOGLE_API_KEY` from the environment and configures the Google GenAI SDK.
    -   Handles file inputs:
        -   Uses the configured vision model if an image MIME type (e.g., `image/jpeg`, `image/png`) is detected. The prompt and image are sent as separate parts.
        -   For other file types, decodes the file content as text and appends it to the user's prompt, along with the filename and MIME type.
    -   Provides robust error handling, including specific messages for API errors (invalid arguments, permission denied, resource exhausted) and issues with prompt feedback (e.g., safety blocks). Error messages follow the "Error: <details>" format.

## Extending the Project

This template is designed to be extensible. Here are some ways you might build upon it:

### Adding a New AI Provider

1.  **Create New Utility File:**
    In the `llm_utils/` directory, create a Python file for your new provider (e.g., `newprovider_utils.py`).
2.  **Implement Provider Logic:**
    -   Write a function (e.g., `get_newprovider_response()`) that takes `prompt`, `file_content`, `filename`, etc., as arguments.
    -   Handle authentication: This usually involves fetching an API key from an environment variable (e.g., `NEWPROVIDER_API_KEY`).
    -   Make the API call to the provider using their SDK or HTTP requests.
    -   Process the response and return it, or return a formatted error string like "Error: <details>".
3.  **Update Backend (`backend/main.py`):**
    -   Import your new utility function: `from llm_utils.newprovider_utils import get_newprovider_response`.
    -   In the `generate_text` endpoint, add an `elif provider == "newprovider":` condition to call your new function.
    -   Add an API key check for `NEWPROVIDER_API_KEY` in the `_check_api_key` helper function or directly in the endpoint.
4.  **Update Frontend (`frontend/index.html`):**
    -   Add an `<option value="newprovider">New Provider Name</option>` to the provider `select` dropdown.
5.  **Update `.env` File:**
    -   Add the new API key to your local `.env` file: `NEWPROVIDER_API_KEY="your_key_here"`.
    -   If you were to create an `.env.example` file, add it there too.
6.  **Update `requirements.txt`:**
    -   If the new provider requires a Python SDK, add it to `requirements.txt` and reinstall dependencies (`pip install -r requirements.txt`).
7.  **Update `config.json` (Optional):**
    -   If the new provider uses configurable model names, add a section for it in `config.json` and update its utility to read from there.

### Modifying the Frontend

The `frontend/index.html` file contains all the HTML structure, Tailwind CSS classes for styling, and JavaScript for interactivity. You can:
-   Change the appearance by modifying Tailwind classes or adding custom CSS.
-   Add new UI elements (e.g., input fields for more parameters, display areas for different types of content).
-   Extend JavaScript functionality (e.g., client-side validation, more complex state management).

### Customizing the Backend

The `backend/main.py` FastAPI application can be extended:
-   Add new API endpoints for different functionalities (e.g., user accounts, history).
-   Integrate with databases or other services.
-   Implement more sophisticated business logic.

This template provides a solid starting point. Feel free to adapt and expand it to fit your specific AI project needs!