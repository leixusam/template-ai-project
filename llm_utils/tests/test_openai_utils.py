import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json

# Import the function to test
from llm_utils.openai_utils import get_openai_chat_response

# Default config for testing, similar to what's in openai_utils.py
DEFAULT_TEST_CONFIG = {
    "openai": {
        "default_model": "gpt-test-default",
        "vision_model": "gpt-test-vision" 
    }
}

class TestOpenAIUtils(unittest.TestCase):

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    @patch('llm_utils.openai_utils.openai.OpenAI')
    @patch('llm_utils.openai_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG) # Patch config directly
    def test_get_openai_chat_response_success_no_file(self, mock_openai_client_constructor, mock_config_actual_not_used):
        # Configure the mock client and its methods
        mock_client_instance = MagicMock()
        mock_chat_completion = MagicMock()
        mock_chat_completion.choices = [MagicMock()]
        mock_chat_completion.choices[0].message = MagicMock()
        mock_chat_completion.choices[0].message.content = "Test OpenAI response"
        
        mock_client_instance.chat.completions.create.return_value = mock_chat_completion
        mock_openai_client_constructor.return_value = mock_client_instance

        prompt = "Hello OpenAI"
        response = get_openai_chat_response(prompt)

        self.assertEqual(response, "Test OpenAI response")
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model=DEFAULT_TEST_CONFIG["openai"]["default_model"], # Check if model from patched config is used
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024 
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    @patch('llm_utils.openai_utils.openai.OpenAI')
    @patch('llm_utils.openai_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_get_openai_chat_response_success_with_file(self, mock_openai_client_constructor, mock_config_actual_not_used):
        mock_client_instance = MagicMock()
        mock_chat_completion = MagicMock()
        mock_chat_completion.choices = [MagicMock()]
        mock_chat_completion.choices[0].message = MagicMock()
        mock_chat_completion.choices[0].message.content = "OpenAI file processed"
        
        mock_client_instance.chat.completions.create.return_value = mock_chat_completion
        mock_openai_client_constructor.return_value = mock_client_instance

        prompt = "Describe this file"
        file_content = b"This is a test file content."
        filename = "test_file.txt"
        
        # Expected system message content snippet
        file_text_snippet = file_content[:2000].decode(errors='ignore')
        expected_system_message = (
            f"The user has uploaded a file named '{filename}'. "
            f"Its content (first 2000 characters) is: {file_text_snippet}"
        )

        response = get_openai_chat_response(prompt, file_content=file_content, filename=filename)

        self.assertEqual(response, "OpenAI file processed")
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model=DEFAULT_TEST_CONFIG["openai"]["default_model"],
            messages=[
                {"role": "system", "content": expected_system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024
        )

    @patch.dict(os.environ, {}, clear=True) # No API Key
    @patch('llm_utils.openai_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_get_openai_chat_response_missing_api_key(self, mock_config_actual_not_used):
        response = get_openai_chat_response("Hello")
        self.assertTrue(response.startswith("Error: OPENAI_API_KEY not found"))

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    @patch('llm_utils.openai_utils.openai.OpenAI')
    @patch('llm_utils.openai_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_get_openai_chat_response_api_connection_error(self, mock_openai_client_constructor, mock_config_actual_not_used):
        # Import openai specific errors locally for patching
        from openai import APIConnectionError

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())
        mock_openai_client_constructor.return_value = mock_client_instance
        
        response = get_openai_chat_response("Hello")
        self.assertTrue(response.startswith("Error: OpenAI API Connection Error"))

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    @patch('llm_utils.openai_utils.openai.OpenAI')
    @patch('llm_utils.openai_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_get_openai_chat_response_rate_limit_error(self, mock_openai_client_constructor, mock_config_actual_not_used):
        from openai import RateLimitError
        
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = RateLimitError(message="Rate limited", response=MagicMock(), body=None)
        mock_openai_client_constructor.return_value = mock_client_instance

        response = get_openai_chat_response("Hello")
        self.assertTrue(response.startswith("Error: OpenAI API Rate Limit Exceeded"))

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_invalid_key"}) # Key is present but invalid
    @patch('llm_utils.openai_utils.openai.OpenAI')
    @patch('llm_utils.openai_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_get_openai_chat_response_authentication_error(self, mock_openai_client_constructor, mock_config_actual_not_used):
        from openai import AuthenticationError
        
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = AuthenticationError(message="Invalid API Key", response=MagicMock(), body=None)
        mock_openai_client_constructor.return_value = mock_client_instance

        response = get_openai_chat_response("Hello")
        self.assertTrue(response.startswith("Error: OpenAI API Authentication Error"))
        self.assertIn("Ensure your OPENAI_API_KEY is correct and active", response)


    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    @patch('llm_utils.openai_utils.openai.OpenAI')
    @patch('llm_utils.openai_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_get_openai_chat_response_generic_api_error(self, mock_openai_client_constructor, mock_config_actual_not_used):
        from openai import APIError # Generic API error
        
        mock_client_instance = MagicMock()
        # Need to mock the structure of an APIError if its attributes are accessed
        error_response = MagicMock()
        error_response.status_code = 500
        # error_response.type = "server_error" # This might not be how type is set on APIError directly
        
        api_error_instance = APIError(message="Generic API Error", request=MagicMock(), body=None)
        # Manually set attributes if they are not part of constructor and are accessed in error formatting
        api_error_instance.status_code = 500 
        api_error_instance.type = "internal_server_error"

        mock_client_instance.chat.completions.create.side_effect = api_error_instance
        mock_openai_client_constructor.return_value = mock_client_instance

        response = get_openai_chat_response("Hello")
        self.assertTrue(response.startswith("Error: OpenAI API Error - Generic API Error"))
        self.assertIn("(Status Code: 500, Type: internal_server_error)", response)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    @patch('llm_utils.openai_utils.openai.OpenAI')
    @patch('llm_utils.openai_utils.MODEL_CONFIG', {"openai": {"default_model": "gpt-test-from-config"}})
    def test_uses_model_from_patched_config(self, mock_openai_client_constructor, mock_config_actual_not_used):
        mock_client_instance = MagicMock()
        mock_chat_completion = MagicMock()
        mock_chat_completion.choices = [MagicMock()]
        mock_chat_completion.choices[0].message = MagicMock()
        mock_chat_completion.choices[0].message.content = "Config model response"
        
        mock_client_instance.chat.completions.create.return_value = mock_chat_completion
        mock_openai_client_constructor.return_value = mock_client_instance

        get_openai_chat_response("Test model config")
        
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="gpt-test-from-config", # Verify this model name is used
            messages=[{"role": "user", "content": "Test model config"}],
            max_tokens=1024
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    @patch('llm_utils.openai_utils.openai.OpenAI')
    @patch('llm_utils.openai_utils.MODEL_CONFIG', {}) # Empty config
    @patch('llm_utils.openai_utils.DEFAULT_MODEL_CONFIG', {"openai": {"default_model": "fallback-gpt-model"}}) # Patch the fallback default
    def test_uses_fallback_model_if_config_empty(self, mock_openai_client_constructor, mock_default_config, mock_model_config):
        mock_client_instance = MagicMock()
        mock_chat_completion = MagicMock()
        mock_chat_completion.choices = [MagicMock()]
        mock_chat_completion.choices[0].message = MagicMock()
        mock_chat_completion.choices[0].message.content = "Fallback model response"
        
        mock_client_instance.chat.completions.create.return_value = mock_chat_completion
        mock_openai_client_constructor.return_value = mock_client_instance

        # We need to reload openai_utils or re-evaluate OPENAI_DEFAULT_MODEL
        # The current openai_utils.py loads config at module level.
        # For this test to be effective, OPENAI_DEFAULT_MODEL needs to be re-evaluated
        # This can be done by re-importing or by directly patching OPENAI_DEFAULT_MODEL
        
        with patch('llm_utils.openai_utils.OPENAI_DEFAULT_MODEL', "fallback-gpt-model-direct-patch"):
             get_openai_chat_response("Test fallback")
        
             mock_client_instance.chat.completions.create.assert_called_once_with(
                 model="fallback-gpt-model-direct-patch", # Check if the directly patched model is used
                 messages=[{"role": "user", "content": "Test fallback"}],
                 max_tokens=1024
             )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    @patch('llm_utils.openai_utils.openai.OpenAI')
    @patch('llm_utils.openai_utils.MODEL_CONFIG', DEFAULT_TEST_CONFIG)
    def test_file_decoding_error(self, mock_openai_client_constructor, mock_config_actual_not_used):
        # This tests the scenario where file_content.decode() fails
        mock_client_instance = MagicMock() # Not actually called if decode fails early
        mock_openai_client_constructor.return_value = mock_client_instance

        prompt = "Test"
        file_content = b'\x80abc' # Invalid start byte for UTF-8
        filename = "bad_encoding.txt"

        response = get_openai_chat_response(prompt, file_content=file_content, filename=filename)
        self.assertTrue(response.startswith("Error: Could not decode file content for OpenAI."))


if __name__ == '__main__':
    unittest.main()
