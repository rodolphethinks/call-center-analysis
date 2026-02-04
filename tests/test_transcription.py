"""
Unit tests for transcription module
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.transcription import AudioTranscriber


@pytest.fixture
def mock_config():
    return {
        'transcription': {
            'model': 'openai/whisper-large-v3-turbo',
            'language': 'Korean',
            'device': 'cpu',
            'batch_size': 24,
            'chunk_length': 30,
            'quality_verification': False,
            'max_retries': 3
        }
    }


@pytest.fixture
def mock_env():
    return {
        'GEMINI_API_KEY': 'test_key',
        'HUGGINGFACE_TOKEN': 'test_token'
    }


class TestAudioTranscriber:
    
    @patch('src.transcription.AutoModelForSpeechSeq2Seq')
    @patch('src.transcription.AutoProcessor')
    def test_transcriber_initialization(
        self, 
        mock_processor_class, 
        mock_model_class,
        mock_env
    ):
        """Test transcriber initialization"""
        # Setup mocks
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        transcriber = AudioTranscriber(
            huggingface_token=mock_env['HUGGINGFACE_TOKEN'],
            quality_verification=False
        )
        
        assert transcriber.model_id == 'openai/whisper-large-v3-turbo'
        assert transcriber.language == 'Korean'
        assert transcriber.device == 'cpu'
    
    @patch('src.transcription.librosa')
    @patch('src.transcription.AutoModelForSpeechSeq2Seq')
    @patch('src.transcription.AutoProcessor')
    def test_transcribe_audio(
        self,
        mock_processor_class,
        mock_model_class,
        mock_librosa,
        mock_env
    ):
        """Test audio transcription"""
        # Setup librosa mock
        import numpy as np
        mock_audio = np.array([0.1, 0.2, 0.3])
        mock_librosa.load.return_value = (mock_audio, 16000)
        
        # Setup processor mock
        mock_processor = Mock()
        mock_input_features = Mock()
        mock_input_features.to.return_value = mock_input_features
        mock_attention_mask = Mock()
        mock_attention_mask.to.return_value = mock_attention_mask
        
        # Create a dict-like mock for inputs
        mock_inputs = {
            'input_features': mock_input_features,
            'attention_mask': mock_attention_mask
        }
        # Make it both dict-like and attribute accessible
        mock_inputs_obj = Mock()
        mock_inputs_obj.input_features = mock_input_features
        mock_inputs_obj.attention_mask = mock_attention_mask
        mock_inputs_obj.__getitem__ = lambda self, key: mock_inputs[key]
        mock_inputs_obj.__contains__ = lambda self, key: key in mock_inputs
        
        mock_processor.return_value = mock_inputs_obj
        mock_processor.batch_decode.return_value = ['Test transcription']
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        # Setup model mock
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.generate.return_value = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        transcriber = AudioTranscriber(
            huggingface_token=mock_env['HUGGINGFACE_TOKEN'],
            quality_verification=False
        )
        
        # Test transcription
        result = transcriber.transcribe_audio('test.wav')
        
        assert result == 'Test transcription'
        mock_model.generate.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
