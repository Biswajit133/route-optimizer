"""LLM providers behind one call: chat(conf, messages, tools).

Messages and tools use the OpenAI chat format everywhere in the Copilot:
  {"role": "system"|"user"|"assistant"|"tool", "content": ...,
   "tool_calls": [...] (assistant), "tool_call_id": ... (tool)}
Two adapters:
- OpenAI-compatible: Groq, OpenAI, and any other /chat/completions server
  (Together, local Ollama ...) - only base_url, key and model differ.
- Anthropic: converts to/from the Messages API.

Returns {"content": str, "tool_calls": [{"id", "name", "arguments": dict}],
         "usage": {"prompt": int, "completion": int}}.
"""
import json
import logging

import requests

log = logging.getLogger('copilot.providers')

TIMEOUT_SECONDS = 30   # one model step; a hung provider should fail fast (one retry follows)


class LLMError(Exception):
    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


def chat(conf, messages, tools):
    if conf['provider'] == 'anthropic':
        return _anthropic(conf, messages, tools)
    return _openai_compatible(conf, messages, tools)


def _post(url, headers, body):
    try:
        res = requests.post(url, headers=headers, json=body, timeout=TIMEOUT_SECONDS)
    except requests.Timeout:
        raise LLMError("The AI service took too long to answer. Please try again.", retryable=True)
    except requests.RequestException as e:
        raise LLMError(f"Could not reach the AI service: {e.__class__.__name__}", retryable=True)

    if res.status_code >= 400:
        try:
            detail = res.json()
        except ValueError:
            detail = {'raw': res.text[:300]}
        err = detail.get('error') if isinstance(detail, dict) else None
        message = err.get('message') if isinstance(err, dict) else (err or str(detail)[:300])
        code = err.get('code') if isinstance(err, dict) else None
        log.warning("LLM HTTP %s: %s", res.status_code, str(detail)[:500])
        if res.status_code == 429:
            raise LLMError("The AI service is busy (rate limit). Please try again in a minute.", retryable=False)
        if res.status_code in (401, 403):
            raise LLMError("The AI service rejected the API key. Check the key in the HSC console.")
        # Groq returns 400 tool_use_failed when the model emitted a malformed tool call.
        raise LLMError(f"AI service error ({res.status_code}): {message}",
                       retryable=code == 'tool_use_failed' or res.status_code >= 500)
    return res.json()


# --------------------------------------------------------------------------- OpenAI-compatible

def _openai_compatible(conf, messages, tools):
    body = {
        'model': conf['model'],
        'messages': messages,
        'temperature': conf['temperature'],
        'max_tokens': conf['max_tokens'],
    }
    if tools:
        body['tools'] = tools
        body['tool_choice'] = 'auto'

    data = _post(conf['base_url'].rstrip('/') + '/chat/completions',
                 {'Authorization': f"Bearer {conf['api_key']}", 'Content-Type': 'application/json'}, body)

    try:
        msg = data['choices'][0]['message']
    except (KeyError, IndexError, TypeError):
        raise LLMError("The AI service sent an empty answer.", retryable=True)

    calls = []
    for c in msg.get('tool_calls') or []:
        fn = c.get('function') or {}
        try:
            args = json.loads(fn.get('arguments') or '{}')
        except ValueError:
            args = {}
        # Anything else on the call (Gemini 3's extra_content.google.thought_signature)
        # must go back unchanged with the next request, or Gemini rejects it.
        extra = {k: v for k, v in c.items() if k not in ('id', 'type', 'function', 'index')}
        calls.append({'id': c.get('id') or fn.get('name'), 'name': fn.get('name'), 'arguments': args, 'extra': extra})

    usage = data.get('usage') or {}
    return {
        'content': msg.get('content') or '',
        'tool_calls': calls,
        'usage': {'prompt': int(usage.get('prompt_tokens') or 0), 'completion': int(usage.get('completion_tokens') or 0)},
    }


# --------------------------------------------------------------------------- Anthropic

def _to_anthropic(messages):
    system, out = [], []

    def push(role, block):
        if out and out[-1]['role'] == role:
            out[-1]['content'].append(block)
        else:
            out.append({'role': role, 'content': [block]})

    for m in messages:
        role = m['role']
        if role == 'system':
            system.append(m['content'])
        elif role == 'user':
            push('user', {'type': 'text', 'text': m['content'] or ''})
        elif role == 'assistant':
            if m.get('content'):
                push('assistant', {'type': 'text', 'text': m['content']})
            for c in m.get('tool_calls') or []:
                fn = c['function']
                push('assistant', {'type': 'tool_use', 'id': c['id'], 'name': fn['name'],
                                   'input': json.loads(fn.get('arguments') or '{}')})
        elif role == 'tool':
            push('user', {'type': 'tool_result', 'tool_use_id': m['tool_call_id'], 'content': m['content']})
    return '\n\n'.join(system), out


def _anthropic(conf, messages, tools):
    system, converted = _to_anthropic(messages)
    body = {
        'model': conf['model'],
        'system': system,
        'messages': converted,
        'max_tokens': conf['max_tokens'],
        'temperature': conf['temperature'],
    }
    if tools:
        body['tools'] = [{'name': t['function']['name'], 'description': t['function']['description'],
                          'input_schema': t['function']['parameters']} for t in tools]

    data = _post(conf['base_url'].rstrip('/') + '/v1/messages', {
        'x-api-key': conf['api_key'],
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
    }, body)

    text, calls = [], []
    for block in data.get('content') or []:
        if block.get('type') == 'text':
            text.append(block.get('text') or '')
        elif block.get('type') == 'tool_use':
            calls.append({'id': block['id'], 'name': block['name'], 'arguments': block.get('input') or {}})

    usage = data.get('usage') or {}
    return {
        'content': ''.join(text),
        'tool_calls': calls,
        'usage': {'prompt': int(usage.get('input_tokens') or 0), 'completion': int(usage.get('output_tokens') or 0)},
    }
