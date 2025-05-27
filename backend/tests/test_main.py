import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure the FastAPI app can be imported.
# This might require adjusting sys.path if tests are run from a different root
# For now, assume backend.main can be found or PYTHONPATH is set.
# If PROJECT_ROOT is not set, we might need to set it here for llm_utils import in main.py
# However, main.py itself adds PROJECT_ROOT to sys.path, so it should be fine if main is imported.

# Must patch environment variables BEFORE importing main and TestClient
# as FastAPI might read them at import time.
MOCKED_ENV_VARS = {
    "OPENAI_API_KEY": "fake_openai_key",
    "GOOGLE_API_KEY": "fake_google_key"
}

# Patch environment variables that main.py will try to load via dotenv
# This needs to happen before `from backend.main import app`
# We also need to ensure that the PROJECT_ROOT and DOTENV_PATH in main.py
# are correctly handled or mocked if they cause issues during test setup.
# Given that main.py calculates PROJECT_ROOT, we assume it works.
# The crucial part is that os.getenv inside main.py gets the mocked values.

@patch.dict(os.environ, MOCKED_ENV_VARS)
class TestMainAPI(unittest.TestCase):

    def setUp(self):
        # This import needs to happen *after* os.environ is patched.
        from fastapi.testclient import TestClient
        from backend.main import app # Import the FastAPI app instance

        self.client = TestClient(app)
        
        # We need to patch the functions *where they are looked up*, which is in backend.main
        self.patcher_openai = patch('backend.main.get_openai_chat_response')
        self.patcher_google = patch('backend.main.get_google_gemini_response')
        
        self.mock_get_openai_chat_response = self.patcher_openai.start()
        self.mock_get_google_gemini_response = self.patcher_google.start()

    def tearDown(self):
        self.patcher_openai.stop()
        self.patcher_google.stop()

    def test_generate_text_openai_success_no_file(self):
        # Configure the mock for OpenAI
        self.mock_get_openai_chat_response.return_value = "Mocked OpenAI response"

        response = self.client.post(
            "/api/generate",
            data={"provider": "openai", "text": "Hello OpenAI"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "Mocked OpenAI response"})
        self.mock_get_openai_chat_response.assert_called_once_with(
            prompt="Hello OpenAI", file_content=None, filename=None
        )

    def test_generate_text_google_success_no_file(self):
        # Configure the mock for Google
        self.mock_get_google_gemini_response.return_value = "Mocked Google response"

        response = self.client.post(
            "/api/generate",
            data={"provider": "google", "text": "Hello Google"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "Mocked Google response"})
        self.mock_get_google_gemini_response.assert_called_once_with(
            prompt="Hello Google", file_content=None, filename=None, mime_type=None
        )

    def test_generate_text_openai_success_with_file(self):
        self.mock_get_openai_chat_response.return_value = "OpenAI file response"
        file_content = b"This is a test file."
        
        response = self.client.post(
            "/api/generate",
            data={"provider": "openai", "text": "Analyze this file"},
            files={"file": ("test.txt", file_content, "text/plain")}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "OpenAI file response"})
        self.mock_get_openai_chat_response.assert_called_once_with(
            prompt="Analyze this file",
            file_content=file_content,
            filename="test.txt"
        )

    def test_generate_text_google_success_with_file(self):
        self.mock_get_google_gemini_response.return_value = "Google file response"
        file_content = b"This is another test file."

        response = self.client.post(
            "/api/generate",
            data={"provider": "google", "text": "Analyze this please"},
            files={"file": ("test_doc.txt", file_content, "text/plain")}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "Google file response"})
        self.mock_get_google_gemini_response.assert_called_once_with(
            prompt="Analyze this please",
            file_content=file_content,
            filename="test_doc.txt",
            mime_type="text/plain"
        )

    def test_generate_text_invalid_provider(self):
        response = self.client.post(
            "/api/generate",
            data={"provider": "unknown_provider", "text": "Hello"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid AI provider specified", response.json()["detail"])

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_google_key"}, clear=True) # No OPENAI_API_KEY
    def test_generate_text_missing_openai_api_key(self):
        # Reload TestClient and app if necessary, or ensure patch is effective for the app instance
        # The @patch.dict on the class should handle this, but let's be sure the app "sees" it.
        # For this test, we specifically want os.getenv("OPENAI_API_KEY") to be None within the endpoint.
        # The class-level patch might be overridden by subsequent specific patches or imports.
        # The _check_api_key function in main.py uses os.getenv.
        
        # To ensure the app reflects this state, we might need to re-initialize client with app re-imported under this patch
        # However, the current structure of main.py's _check_api_key should be covered by patching os.environ at the test method level.
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}): # More direct way to simulate missing key for this call
            from fastapi.testclient import TestClient
            from backend.main import app # Re-import app to catch fresh env var state
            client = TestClient(app)
            
            response = client.post(
                "/api/generate",
                data={"provider": "openai", "text": "Hello OpenAI"}
            )
            self.assertEqual(response.status_code, 500)
            self.assertIn("OPENAI_API_KEY not configured", response.json()["detail"])

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_openai_key"}, clear=True) # No GOOGLE_API_KEY
    def test_generate_text_missing_google_api_key(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}):
            from fastapi.testclient import TestClient
            from backend.main import app 
            client = TestClient(app)

            response = client.post(
                "/api/generate",
                data={"provider": "google", "text": "Hello Google"}
            )
            self.assertEqual(response.status_code, 500)
            self.assertIn("GOOGLE_API_KEY not configured", response.json()["detail"])

    def test_generate_text_llm_util_returns_openai_error(self):
        self.mock_get_openai_chat_response.return_value = "Error: OpenAI specific problem"
        
        response = self.client.post(
            "/api/generate",
            data={"provider": "openai", "text": "Test error"}
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Error: OpenAI specific problem")

    def test_generate_text_llm_util_returns_google_error(self):
        self.mock_get_google_gemini_response.return_value = "Error: Google specific problem"

        response = self.client.post(
            "/api/generate",
            data={"provider": "google", "text": "Test error google"}
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Error: Google specific problem")

    def test_file_read_error_handling(self):
        # This test assumes that if file.read() itself fails, it's caught.
        # The current main.py has a try-except around file.read()
        
        mock_file = MagicMock()
        mock_file.filename = "error_file.txt"
        mock_file.content_type = "text/plain"
        mock_file.read.side_effect = Exception("Failed to read file") # Simulate read error
        # We also need to ensure file.close() is called, so make it a MagicMock too.
        mock_file.close = MagicMock()


        # The TestClient's files argument expects a tuple or a file-like object.
        # To inject our mock that raises an error on read, we need a bit more setup.
        # One way is to patch 'UploadFile' if it's type-hinted and used for specific methods,
        # or more directly, patch the part of the code that calls 'await file.read()'.
        # Let's try to patch the `file.read()` call inside the endpoint, if possible,
        # or ensure the data for `files` triggers this.
        # The TestClient handles file creation, so we can't easily inject a faulty file object directly.
        # Instead, we will patch 'backend.main.UploadFile' if it were used explicitly,
        # but since it's a type hint, FastAPI handles its creation.
        # A more robust way is to ensure the file processing path can be triggered to fail.
        # Let's assume the structure of `main.py`'s file handling.
        # The test for this is more complex due to how TestClient handles file uploads.
        # A simpler approach for this specific test might be to patch the `read` method
        # of the `UploadFile` class globally for the scope of this test.

        with patch('fastapi.UploadFile.read', side_effect=Exception("Simulated file read error")):
            response = self.client.post(
                "/api/generate",
                data={"provider": "openai", "text": "Analyze this"},
                files={"file": ("error_file.txt", b"some content", "text/plain")}
            )
            self.assertEqual(response.status_code, 400) # main.py raises 400 for file processing error
            self.assertIn("Error processing uploaded file: Simulated file read error", response.json()["detail"])


if __name__ == '__main__':
    unittest.main()
