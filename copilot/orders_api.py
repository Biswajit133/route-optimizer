"""Creates an order through the orders service create-order API.

The CRM user's own JWT (the Authorization header of the confirm request) is
forwarded, so the order is created as that user, with the same validation,
city access, lab and phlebo auto-assign rules as any other create-order call.
Nothing here writes to the hsc database directly.

ORDERS_API_URL = base URL of the orders service as the CRM uses it
(environment.OrderApiendpoint), e.g. https://api.example.com/cust/orders
"""
import json
import logging
import os

import requests

log = logging.getLogger('copilot.orders_api')

TIMEOUT_SECONDS = 60


class OrderApiError(Exception):
    """uncertain=True: we could not tell whether the order was created
    (timeout / connection dropped after sending)."""

    def __init__(self, message, uncertain=False):
        super().__init__(message)
        self.uncertain = uncertain


def base_url():
    return (os.environ.get('ORDERS_API_URL') or '').rstrip('/')


def form_fields(payload):
    """Flatten the draft payload into the form fields create-order reads
    (patients[0][fullName], patients[0][test] as a JSON string, ...)."""
    fields = []
    for key, value in payload.items():
        if key == 'patients':
            continue
        fields.append((key, '' if value is None else str(value)))
    for i, patient in enumerate(payload.get('patients') or []):
        for key, value in patient.items():
            if key == 'test':
                value = json.dumps(value, ensure_ascii=False)
            fields.append((f'patients[{i}][{key}]', '' if value is None else str(value)))
    return fields


def _message(body, fallback):
    if not isinstance(body, dict):
        return fallback
    error = body.get('error') if body.get('error') is not None else body.get('message')
    if isinstance(error, str) and error.strip():
        return error.strip()
    if isinstance(error, dict) and error:
        first = next(iter(error.values()))
        return str(first[0] if isinstance(first, list) and first else first)
    # Lumen validation failure (HTTP 422): {"field": ["message", ...], ...}
    for value in body.values():
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0]
    return fallback


def create_order(payload, authorization):
    """Returns create-order's data: {id (visit), order_id, order_number, amount, ...}."""
    url = base_url()
    if not url:
        raise OrderApiError("Order creation is not configured on the Patho Assist service (ORDERS_API_URL).")
    try:
        res = requests.post(f'{url}/create-order', data=form_fields(payload),
                            headers={'Authorization': authorization, 'Accept': 'application/json'},
                            timeout=TIMEOUT_SECONDS)
    except requests.Timeout:
        raise OrderApiError("The orders service took too long to answer. Check the Orders list before trying "
                            "again - the order may have been created.", uncertain=True)
    except requests.ConnectionError as e:
        log.error("create-order connection failed: %s", e)
        raise OrderApiError("Could not reach the orders service. Please try again.")

    try:
        body = res.json()
    except ValueError:
        log.error("create-order returned non-JSON (HTTP %s): %s", res.status_code, res.text[:300])
        raise OrderApiError(f"The orders service returned an unexpected answer (HTTP {res.status_code}). "
                            "Check the Orders list before trying again.", uncertain=res.status_code < 500)

    if res.status_code in (401, 403):
        raise OrderApiError(_message(body, "Your session has expired. Please sign in again."))
    if not body.get('success') or not isinstance(body.get('data'), dict):
        raise OrderApiError(_message(body, "The order could not be created."))
    return body['data']
