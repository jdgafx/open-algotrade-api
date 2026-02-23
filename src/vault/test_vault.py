import unittest
from unittest.mock import MagicMock, patch
from src.vault.vault_manager import VaultManager


class TestVaultManager(unittest.TestCase):
    @patch("src.vault.vault_manager.Exchange")
    @patch("src.vault.vault_manager.Info")
    @patch("src.vault.vault_manager.Account")
    @patch("os.getenv")
    def test_vault_manager_initialization(
        self, mock_getenv, mock_account, mock_info, mock_exchange
    ):
        mock_getenv.side_effect = (
            lambda k,
            d=None: "0x0000000000000000000000000000000000000000000000000000000000000001"
            if k == "HYPERLIQUID_PRIVATE_KEY"
            else d
        )
        mock_account.from_key.return_value = MagicMock(address="0xUserAddress")

        manager = VaultManager()

        self.assertIsNotNone(manager.exchange)
        self.assertIsNotNone(manager.info)
        self.assertEqual(manager.account.address, "0xUserAddress")

    @patch("src.vault.vault_manager.Exchange")
    @patch("src.vault.vault_manager.Info")
    @patch("src.vault.vault_manager.Account")
    @patch("os.getenv")
    def test_create_vault_if_not_exists_existing(
        self, mock_getenv, mock_account, mock_info, mock_exchange
    ):
        mock_getenv.side_effect = (
            lambda k,
            d=None: "0x0000000000000000000000000000000000000000000000000000000000000001"
            if k == "HYPERLIQUID_PRIVATE_KEY"
            else d
        )
        mock_info_instance = mock_info.return_value
        mock_info_instance.query_sub_accounts.return_value = [
            {"subAccountUser": "0xVaultAddress"}
        ]

        manager = VaultManager()
        vault_addr = manager.create_vault_if_not_exists()

        self.assertEqual(vault_addr, "0xVaultAddress")
        self.assertEqual(manager.vault_address, "0xVaultAddress")

    @patch("src.vault.vault_manager.Exchange")
    @patch("src.vault.vault_manager.Info")
    @patch("src.vault.vault_manager.Account")
    @patch("os.getenv")
    def test_get_vault_equity(
        self, mock_getenv, mock_account, mock_info, mock_exchange
    ):
        mock_getenv.side_effect = (
            lambda k,
            d=None: "0x0000000000000000000000000000000000000000000000000000000000000001"
            if k == "HYPERLIQUID_PRIVATE_KEY"
            else d
        )
        mock_info_instance = mock_info.return_value
        mock_info_instance.user_state.return_value = {
            "marginSummary": {"accountValue": "1000.50"}
        }

        manager = VaultManager()
        manager.vault_address = "0xVaultAddress"

        equity = manager.get_vault_equity()
        self.assertEqual(equity, 1000.50)


if __name__ == "__main__":
    unittest.main()
