import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json

# Import the function to test
from llm_utils.google_utils import get_google_gemini_response

# Import specific Google errors and types for mocking
from google.generativeai import types as google_genai_types
from google.generativeai import errors as google_genai_errors


# Default config for testing, similar to what's in google_utils.py
DEFAULT_TEST_CONFIG = {
    "google": {
        "default_model": "gemini-test-default",
        "vision_model": "gemini-test-vision"
    }
}

class TestGoogleUtils(unittest.TestCase):

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_google_key"})
    @patch('llm_utils.google_utils.google_genai_sdk.configure')
    @patch('llm_utils.google_utils.google_genai_sdk.GenerativeModel')
    @patch('llm_utils.google_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_get_google_gemini_response_success_text_only(self, mock_model_config, mock_generative_model, mock_configure):
        # Configure mocks
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Test Google response"
        mock_response.prompt_feedback = None # Simulate no blocking
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance

        prompt = "Hello Google"
        response = get_google_gemini_response(prompt)

        self.assertEqual(response, "Test Google response")
        mock_configure.assert_called_once_with(api_key="fake_google_key")
        mock_generative_model.assert_called_once_with(model_name=DEFAULT_TEST_CONFIG["google"]["default_model"])
        mock_model_instance.generate_content.assert_called_once_with(contents=[prompt])

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_google_key"})
    @patch('llm_utils.google_utils.google_genai_sdk.configure')
    @patch('llm_utils.google_utils.google_genai_sdk.GenerativeModel')
    @patch('llm_utils.google_utils.google_genai_types.Part') # Mock Part creation
    @patch('llm_utils.google_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_get_google_gemini_response_success_with_image_file(self, mock_model_config, mock_part_constructor, mock_generative_model, mock_configure):
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Google image processed"
        mock_response.prompt_feedback = None
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance
        
        # Mock what Part.from_bytes returns
        mock_image_part_instance = MagicMock(spec=google_genai_types.Part)
        mock_part_constructor.from_bytes.return_value = mock_image_part_instance

        prompt = "Describe this image"
        file_content = b"fake_image_bytes"
        filename = "test_image.png"
        mime_type = "image/png"

        response = get_google_gemini_response(prompt, file_content=file_content, filename=filename, mime_type=mime_type)

        self.assertEqual(response, "Google image processed")
        mock_configure.assert_called_once_with(api_key="fake_google_key")
        # Check that the vision model was selected
        mock_generative_model.assert_called_once_with(model_name=DEFAULT_TEST_CONFIG["google"]["vision_model"])
        mock_part_constructor.from_bytes.assert_called_once_with(data=file_content, mime_type=mime_type)
        # Check that the contents were structured correctly for multimodal input
        mock_model_instance.generate_content.assert_called_once_with(contents=[prompt, mock_image_part_instance])


    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_google_key"})
    @patch('llm_utils.google_utils.google_genai_sdk.configure')
    @patch('llm_utils.google_utils.google_genai_sdk.GenerativeModel')
    @patch('llm_utils.google_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_get_google_gemini_response_success_with_text_file(self, mock_model_config, mock_generative_model, mock_configure):
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Google text file processed"
        mock_response.prompt_feedback = None
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance

        prompt = "Summarize this document"
        file_content = b"This is a test text document."
        filename = "test_doc.txt"
        mime_type = "text/plain"
        
        file_text_snippet = file_content[:50000].decode(errors='replace')
        expected_enhanced_prompt = (
            f"{prompt}\n\n"
            f"--- User Uploaded File: {filename} (MIME type: {mime_type}) ---\n"
            f"{file_text_snippet}\n"
            f"--- End of File Content ---"
        )

        response = get_google_gemini_response(prompt, file_content=file_content, filename=filename, mime_type=mime_type)

        self.assertEqual(response, "Google text file processed")
        mock_configure.assert_called_once_with(api_key="fake_google_key")
        # Default model should be used for text files
        mock_generative_model.assert_called_once_with(model_name=DEFAULT_TEST_CONFIG["google"]["default_model"])
        mock_model_instance.generate_content.assert_called_once_with(contents=[expected_enhanced_prompt])

    @patch.dict(os.environ, {}, clear=True) # No API Key
    @patch('llm_utils.google_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG) # Patch config
    def test_get_google_gemini_response_missing_api_key(self, mock_model_config):
        response = get_google_gemini_response("Hello")
        self.assertTrue(response.startswith("Error: GOOGLE_API_KEY not found"))

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_google_key"})
    @patch('llm_utils.google_utils.google_genai_sdk.configure') # Mock configure to avoid it raising error itself
    @patch('llm_utils.google_utils.google_genai_sdk.GenerativeModel')
    @patch('llm_utils.google_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_get_google_gemini_response_permission_denied(self, mock_model_config, mock_generative_model, mock_configure):
        mock_model_instance = MagicMock()
        # Simulate PermissionDenied error from the SDK
        mock_model_instance.generate_content.side_effect = google_genai_errors.PermissionDenied(message="Fake permission denied")
        mock_generative_model.return_value = mock_model_instance

        response = get_google_gemini_response("Test permission error")
        self.assertTrue(response.startswith("Error: Google API Error - Fake permission denied"))
        self.assertIn("This indicates an issue with your GOOGLE_API_KEY", response)

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_google_key"})
    @patch('llm_utils.google_utils.google_genai_sdk.configure')
    @patch('llm_utils.google_utils.google_genai_sdk.GenerativeModel')
    @patch('llm_utils.google_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_get_google_gemini_response_invalid_argument(self, mock_model_config, mock_generative_model, mock_configure):
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.side_effect = google_genai_errors.InvalidArgumentError(message="Fake invalid argument")
        mock_generative_model.return_value = mock_model_instance

        response = get_google_gemini_response("Test invalid argument")
        self.assertTrue(response.startswith("Error: Google API Invalid Argument - Fake invalid argument"))

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_google_key"})
    @patch('llm_utils.google_utils.google_genai_sdk.configure')
    @patch('llm_utils.google_utils.google_genai_sdk.GenerativeModel')
    @patch('llm_utils.google_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_get_google_gemini_response_prompt_blocked(self, mock_model_config, mock_generative_model, mock_configure):
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        
        # Simulate a blocked prompt
        mock_response.prompt_feedback = MagicMock()
        mock_response.prompt_feedback.block_reason = MagicMock()
        mock_response.prompt_feedback.block_reason.name = "SAFETY" # Example reason
        mock_response.prompt_feedback.safety_ratings = [
            MagicMock(category=google_genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, probability=google_genai_types.HarmProbability.HIGH)
        ]
        # If text is accessed on a blocked prompt, it might raise ValueError or be empty.
        # Let's ensure .text is not accessed or handled.
        # The code checks prompt_feedback first.
        
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance

        response = get_google_gemini_response("A problematic prompt")
        self.assertTrue(response.startswith("Error: Prompt blocked by Google API. Reason: SAFETY"))
        self.assertIn("HARM_CATEGORY_DANGEROUS_CONTENT: HIGH", response)

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_google_key"})
    @patch('llm_utils.google_utils.google_genai_sdk.configure')
    @patch('llm_utils.google_utils.google_genai_sdk.GenerativeModel')
    @patch('llm_utils.google_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_get_google_gemini_response_empty_text_finish_reason_other(self, mock_model_config, mock_generative_model, mock_configure):
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "" # Empty text
        mock_response.prompt_feedback = None # Not blocked by prompt_feedback

        # Simulate finish reason OTHER and some safety ratings
        mock_candidate = MagicMock()
        mock_candidate.finish_reason.name = "OTHER"
        mock_candidate.safety_ratings = [
            MagicMock(category=google_genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT, probability=google_genai_types.HarmProbability.NEGLIGIBLE)
        ]
        mock_response.candidates = [mock_candidate]
        
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance

        response = get_google_gemini_response("A prompt that might yield no text")
        self.assertTrue(response.startswith("Error: Google Gemini returned no content. Finish Reason: OTHER."))
        self.assertIn("HARM_CATEGORY_HARASSMENT: NEGLIGIBLE", response)

    def test_get_google_gemini_response_file_without_filename_or_mimetype(self):
        # This test doesn't need full mocking of the API calls as it should fail validation early.
        response = get_google_gemini_response(
            prompt="Test", 
            file_content=b"some content", 
            filename=None, # Missing filename
            mime_type="text/plain"
        )
        self.assertEqual(response, "Error: File content provided without filename or MIME type. Both are required for Google Gemini.")

        response = get_google_gemini_response(
            prompt="Test", 
            file_content=b"some content", 
            filename="test.txt", 
            mime_type=None # Missing mime_type
        )
        self.assertEqual(response, "Error: File content provided without filename or MIME type. Both are required for Google Gemini.")
        
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_google_key"})
    @patch('llm_utils.google_utils.google_genai_sdk.configure')
    @patch('llm_utils.google_utils.google_genai_sdk.GenerativeModel')
    @patch('llm_utils.google_utils.MODEL_CONFIG', {"google": {"default_model": "custom-from-config", "vision_model": "custom-vision-from-config"}})
    def test_uses_model_from_patched_config(self, mock_model_config, mock_generative_model, mock_configure):
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Custom model response"
        mock_response.prompt_feedback = None
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance

        # Test with text only (should use default_model)
        get_google_gemini_response("Test default model from config")
        mock_generative_model.assert_called_with(model_name="custom-from-config")
        
        # Test with image (should use vision_model)
        mock_part_constructor = MagicMock() # Need to mock Part for image part
        with patch('llm_utils.google_utils.google_genai_types.Part', mock_part_constructor):
            get_google_gemini_response("Test vision model from config", file_content=b"img", filename="i.png", mime_type="image/png")
            mock_generative_model.assert_called_with(model_name="custom-vision-from-config")

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_google_key"})
    @patch('llm_utils.google_utils.google_genai_sdk.configure')
    @patch('llm_utils.google_utils.google_genai_sdk.GenerativeModel')
    @patch('llm_utils.google_utils.MODEL_CONFIG', {}) # Empty config
    @patch('llm_utils.google_utils.DEFAULT_MODEL_CONFIG', {"google": {"default_model": "fallback-gemini-model", "vision_model": "fallback-gemini-vision"}})
    def test_uses_fallback_model_if_config_empty(self, mock_default_config_val, mock_model_config_val, mock_generative_model, mock_configure):
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Fallback model response"
        mock_response.prompt_feedback = None
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance

        # This test relies on the module-level constants being influenced by the patched MODEL_CONFIG and DEFAULT_MODEL_CONFIG.
        # The current google_utils.py loads these at module import.
        # To effectively test this, we need to ensure GOOGLE_DEFAULT_MODEL and GOOGLE_VISION_MODEL are re-evaluated.
        # Patching them directly is the most straightforward way for this test.
        with patch('llm_utils.google_utils.GOOGLE_DEFAULT_MODEL', "direct-fallback-default"), \
             patch('llm_utils.google_utils.GOOGLE_VISION_MODEL', "direct-fallback-vision"):

            # Test text-only uses patched default fallback
            get_google_gemini_response("Test fallback default")
            mock_generative_model.assert_called_with(model_name="direct-fallback-default")

            # Test image uses patched vision fallback
            mock_part_constructor = MagicMock()
            with patch('llm_utils.google_utils.google_genai_types.Part', mock_part_constructor):
                get_google_gemini_response("Test fallback vision", file_content=b"img", filename="i.png", mime_type="image/png")
                mock_generative_model.assert_called_with(model_name="direct-fallback-vision")

if __name__ == '__main__':
    unittest.main()
