from unittest.mock import Mock, patch
import pytest

@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch("src.controllers.main_controller.ConfigLoader"), \
         patch("src.controllers.main_controller.KafkaConfigLoader"), \
         patch("src.controllers.main_controller.FraudService"), \
         patch("src.controllers.main_controller.FraudListener") as mock_listener, \
         patch("src.controllers.main_controller.threading.Thread") as mock_thread:
             
        from src.controllers.main_controller import lifespan, app
        
        yield lifespan, mock_listener, mock_thread

@pytest.mark.asyncio
async def test_lifespan(mock_dependencies):
    lifespan, mock_listener, mock_thread = mock_dependencies
    
    mock_app = Mock()
    
    async with lifespan(mock_app):
        # Before yield
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()
    
    # After yield
    pass
