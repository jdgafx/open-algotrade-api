import os
import logging
from typing import Optional
from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VaultManager:
    def __init__(self):
        load_dotenv()
        self.private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
        self.network = os.getenv("HYPERLIQUID_NETWORK", "testnet").lower()
        self.vault_address = os.getenv("VAULT_ADDRESS")

        if not self.private_key:
            raise ValueError(
                "HYPERLIQUID_PRIVATE_KEY not found in environment variables"
            )

        self.account = Account.from_key(self.private_key)
        self.base_url = (
            constants.TESTNET_API_URL
            if self.network == "testnet"
            else constants.MAINNET_API_URL
        )

        self.info = Info(self.base_url, skip_ws=True)
        self.exchange = Exchange(self.account, self.base_url)

        logger.info(
            f"Initialized VaultManager for {self.network} with account {self.account.address}"
        )

    def create_vault_if_not_exists(self, name: str = "default_vault") -> str:
        """
        Checks if a vault (subaccount) exists. If not, creates one.
        Returns the vault address.
        """
        if self.vault_address:
            logger.info(f"Using configured vault address: {self.vault_address}")
            return self.vault_address

        logger.info("No VAULT_ADDRESS configured. Checking for existing subaccounts...")
        sub_accounts = self.info.query_sub_accounts(self.account.address)

        if sub_accounts:
            self.vault_address = sub_accounts[0]["subAccountUser"]
            logger.info(f"Found existing subaccount/vault: {self.vault_address}")
            return self.vault_address

        logger.info(
            f"No subaccounts found. Creating new vault/subaccount with name '{name}'..."
        )
        try:
            result = self.exchange.create_sub_account(name)
            if result["status"] == "ok":
                logger.info(
                    "Subaccount creation transaction sent. Fetching new address..."
                )
                import time

                time.sleep(2)
                sub_accounts = self.info.query_sub_accounts(self.account.address)
                if sub_accounts:
                    self.vault_address = sub_accounts[-1]["subAccountUser"]
                    logger.info(f"Created and selected new vault: {self.vault_address}")
                    return self.vault_address
                else:
                    raise Exception(
                        "Failed to retrieve new subaccount address after creation."
                    )
            else:
                raise Exception(f"Failed to create subaccount: {result}")
        except Exception as e:
            logger.error(f"Error creating vault: {e}")
            raise

    def get_vault_equity(self) -> float:
        """
        Retrieves the equity of the managed vault.
        """
        if not self.vault_address:
            raise ValueError(
                "Vault address not set. Call create_vault_if_not_exists() first."
            )

        try:
            user_state = self.info.user_state(self.vault_address)
            equity = float(user_state["marginSummary"]["accountValue"])
            logger.info(f"Vault {self.vault_address} Equity: {equity}")
            return equity
        except Exception as e:
            logger.error(f"Error getting vault equity: {e}")
            return 0.0

    def deposit_to_vault(self, amount_usd: float):
        """
        Deposits USD into the vault from the main account.
        """
        if not self.vault_address:
            raise ValueError(
                "Vault address not set. Call create_vault_if_not_exists() first."
            )

        logger.info(f"Depositing {amount_usd} USD to vault {self.vault_address}...")
        try:
            from hyperliquid.utils.signing import float_to_usd_int

            amount_int = float_to_usd_int(amount_usd)

            result = self.exchange.sub_account_transfer(
                self.vault_address, True, amount_int
            )
            logger.info(f"Deposit result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error depositing to vault: {e}")
            raise


if __name__ == "__main__":
    try:
        manager = VaultManager()
        vault_addr = manager.create_vault_if_not_exists()
        print(f"Managed Vault Address: {vault_addr}")

        equity = manager.get_vault_equity()
        print(f"Current Vault Equity: {equity}")

    except Exception as e:
        print(f"Execution failed: {e}")
