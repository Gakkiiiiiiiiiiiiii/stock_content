"""Credential adapters; references only, never secret values."""

from stock_content.adapters.credentials.file_secret_provider import FileSecretProvider, SecretUnavailable

__all__ = ["FileSecretProvider", "SecretUnavailable"]
