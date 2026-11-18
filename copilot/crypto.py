"""Decrypts the LLM API key the HSC console stores in copilot_llm_settings.

Format (written by the PHP console, AdminSetupCopilotService::encrypt):
    "v1:" + base64(iv[12 bytes] + tag[16 bytes] + ciphertext)
AES-256-GCM with COPILOT_ENC_KEY = base64 of 32 random bytes, the same value in
the customers service .env and this service's .env.

The key never lives in this repo (it is public) - only in the DB, encrypted.
"""
import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class CryptoError(Exception):
    pass


def _key():
    raw = os.environ.get('COPILOT_ENC_KEY', '').strip()
    if not raw:
        raise CryptoError("COPILOT_ENC_KEY is not set on the Copilot service")
    try:
        key = base64.b64decode(raw)
    except Exception:
        raise CryptoError("COPILOT_ENC_KEY is not valid base64")
    if len(key) != 32:
        raise CryptoError("COPILOT_ENC_KEY must be 32 bytes (base64 encoded)")
    return key


def decrypt(value):
    if not value:
        return None
    if not value.startswith('v1:'):
        raise CryptoError("Unknown API key format")
    blob = base64.b64decode(value[3:])
    iv, tag, ciphertext = blob[:12], blob[12:28], blob[28:]
    return AESGCM(_key()).decrypt(iv, ciphertext + tag, None).decode('utf-8')


def encrypt(plain):
    """Same format as the PHP side; used by tests and local setup only."""
    iv = os.urandom(12)
    sealed = AESGCM(_key()).encrypt(iv, plain.encode('utf-8'), None)
    ciphertext, tag = sealed[:-16], sealed[-16:]
    return 'v1:' + base64.b64encode(iv + tag + ciphertext).decode('ascii')
