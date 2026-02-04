"""
Unit tests for analysis module
"""

import pytest
from unittest.mock import Mock, patch
from src.analysis import ConversationAnalyzer


@pytest.fixture
def mock_api_key():
    return 'test_api_key'


class TestConversationAnalyzer:
    
    @patch('src.analysis.genai')
    def test_analyzer_initialization(self, mock_genai, mock_api_key):
        """Test analyzer initialization"""
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        
        analyzer = ConversationAnalyzer(api_key=mock_api_key)
        
        # The default model should be gemini-2.0-flash (when 'gemini' is in the name)
        assert 'flash' in analyzer.model_name
        assert analyzer.max_workers == 32
        mock_genai.Client.assert_called_once_with(api_key=mock_api_key)
    
    @patch('src.analysis.genai')
    def test_analyze_conversation(self, mock_genai, mock_api_key):
        """Test conversation analysis"""
        # Setup mock
        mock_response = Mock()
        mock_response.text.strip.return_value = '{"customer_summary": "Test"}'
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client
        
        # Test
        analyzer = ConversationAnalyzer(api_key=mock_api_key)
        result = analyzer.analyze_conversation("Test transcription")
        
        assert result == '{"customer_summary": "Test"}'
        mock_client.models.generate_content.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
