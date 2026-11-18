"""LLM settings, edited from the HSC console, cached here for 60 seconds.

copilot_llm_settings: the row with customer_id NULL is the global default; a
customer's row overrides it field by field (NULL = inherit). So switching
Groq -> Claude for everyone, or for one customer, is a console change with no
deploy.

Dev fallback only: when the DB has no usable provider/key and
COPILOT_LLM_API_KEY is set in .env, that env config is used.
"""
import logging
import os
import threading
import time

from . import crypto, db

log = logging.getLogger('copilot.settings')

CACHE_SECONDS = 60

DEFAULT_BASE_URLS = {
    'groq': 'https://api.groq.com/openai/v1',
    'openai': 'https://api.openai.com/v1',
    'anthropic': 'https://api.anthropic.com',
}
PROVIDERS = ('groq', 'openai', 'anthropic', 'openai_compatible')

FIELDS = ('provider', 'base_url', 'model', 'api_key_enc', 'temperature', 'max_tokens', 'daily_message_limit')
DEFAULTS = {'temperature': 0.2, 'max_tokens': 1024, 'daily_message_limit': 200}

_cache = {}
_lock = threading.Lock()


class SettingsError(Exception):
    pass


def _load_rows(customer_id):
    with db.read() as conn:
        return db.fetch_all(conn, """
            SELECT * FROM copilot_llm_settings
            WHERE is_active = 1 AND (customer_id IS NULL OR customer_id = %s)
            ORDER BY customer_id IS NULL DESC, id DESC
        """, (customer_id,))


def _merge(rows):
    merged = {}
    global_rows = [r for r in rows if r['customer_id'] is None][:1]
    customer_rows = [r for r in rows if r['customer_id'] is not None][:1]
    for row in global_rows + customer_rows:  # customer row last, so it wins
        for f in FIELDS:
            if row.get(f) not in (None, ''):
                merged[f] = row[f]
    return merged


def _env_fallback():
    key = os.environ.get('COPILOT_LLM_API_KEY')
    if not key:
        return None
    return {
        'provider': os.environ.get('COPILOT_LLM_PROVIDER', 'groq'),
        'base_url': os.environ.get('COPILOT_LLM_BASE_URL') or None,
        'model': os.environ.get('COPILOT_LLM_MODEL', 'openai/gpt-oss-120b'),
        'api_key': key,
        'source': 'env',
    }


def _build(customer_id):
    try:
        merged = _merge(_load_rows(customer_id))
    except Exception as e:  # table missing before the SQL is run, DB down...
        log.warning("copilot_llm_settings unreadable: %s", e)
        merged = {}

    conf = {k: merged.get(k, DEFAULTS.get(k)) for k in FIELDS}
    conf['source'] = 'db'
    try:
        conf['api_key'] = crypto.decrypt(conf.pop('api_key_enc')) if merged.get('api_key_enc') else None
    except Exception as e:
        log.error("Could not decrypt the Copilot API key: %s", e)
        conf['api_key'] = None
        conf['key_error'] = str(e)

    if not conf.get('provider') or not conf.get('model') or not conf.get('api_key'):
        env = _env_fallback()
        if env:
            env.update({k: conf.get(k) or DEFAULTS.get(k) for k in ('temperature', 'max_tokens', 'daily_message_limit')})
            conf = env

    conf['temperature'] = float(conf.get('temperature') if conf.get('temperature') is not None else 0.2)
    conf['max_tokens'] = int(conf.get('max_tokens') or 1024)
    conf['daily_message_limit'] = int(conf.get('daily_message_limit') or 0)
    if conf.get('provider') and not conf.get('base_url'):
        conf['base_url'] = DEFAULT_BASE_URLS.get(conf['provider'])
    return conf


def for_customer(customer_id):
    now = time.time()
    with _lock:
        hit = _cache.get(customer_id)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    conf = _build(customer_id)
    with _lock:
        _cache[customer_id] = (now, conf)
    return conf


def require_ready(conf):
    """Raises SettingsError with a message the console admin can act on."""
    if conf.get('provider') not in PROVIDERS:
        raise SettingsError("Copilot LLM provider is not configured. Set it in the HSC console › Patho Assist.")
    if not conf.get('model'):
        raise SettingsError("Copilot LLM model is not configured.")
    if not conf.get('api_key'):
        raise SettingsError(conf.get('key_error') or "Copilot LLM API key is not set.")
    if not conf.get('base_url'):
        raise SettingsError("Copilot LLM base URL is required for an OpenAI-compatible provider.")


def public_view(conf):
    """What the CRM widget may know about the config (never the key)."""
    return {'provider': conf.get('provider'), 'model': conf.get('model'),
            'ready': bool(conf.get('provider') and conf.get('model') and conf.get('api_key'))}
