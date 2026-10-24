import unittest
from src.config.settings import Config


class TestSetup(unittest.TestCase):
    """Verify setup and environment configuration defaults."""

    def test_config_paths(self):
        """Verify default data and directory paths."""
        self.assertEqual(Config.DATA_DIR.name, "data")
        self.assertEqual(Config.RAW_DATA_DIR.name, "raw")
        self.assertEqual(Config.PROCESSED_DATA_DIR.name, "processed")

    def test_api_token_validation(self):
        """Verify API token validation behaves predictably without leaking secrets."""
        valid = Config.validate_api_token()
        self.assertIsInstance(valid, bool)


if __name__ == "__main__":
    unittest.main()

