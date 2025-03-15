import unittest
from diagnostic import run_diagnostics
from logging_config import setup_logging
import os
from dotenv import load_load_dotenv

class TestRiskBot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_load_dotenv()
        cls.logger = setup_logging()
    
    def test_diagnostics(self):
        """Test if all diagnostic checks pass"""
        self.assertTrue(run_diagnostics())
    
    def test_api_credentials(self):
        """Test if API credentials are properly set"""
        self.assertIsNotNone(os.getenv('BINANCE_API_KEY'))
        self.assertIsNotNone(os.getenv('BINANCE_API_SECRET'))
    
    def test_binance_connection(self):
        """Test Binance API connection"""
        try:
            from binance.client import Client
            client = Client(
                os.getenv('BINANCE_API_KEY'),
                os.getenv('BINANCE_API_SECRET')
            )
            # Try to get account information
            info = client.get_account()
            self.assertIsNotNone(info)
        except Exception as e:
            self.fail(f"Binance connection failed: {str(e)}")

if __name__ == '__main__':
    unittest.main()
