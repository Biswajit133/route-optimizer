"""Calls the orders service for the order-action cards (tools_actions).

Same idea as orders_api: the CRM user's own JWT is forwarded, so each call runs
as that user with the same checks as the CRM buttons. Nothing here writes to
the hsc database directly.

The orders service returns business errors as HTTP 201 with
{"success": false, "error": "..."}, so success is read from the body.
"""
import logging

import requests

from .orders_api import _message, base_url

log = logging.getLogger('copilot.actions_api')

TIMEOUT_SECONDS = 45


class ActionApiError(Exception):
    """uncertain=True: we could not tell whether the change was made."""

    def __init__(self, message, uncertain=False):
        super().__init__(message)
        self.uncertain = uncertain


def call(method, path, authorization, body=None, params=None):
    """Returns the response body's `data` (may be None). Raises ActionApiError."""
    url = base_url()
    if not url:
        raise ActionApiError("Order actions are not configured on the Patho Assist service (ORDERS_API_URL).")
    try:
        res = requests.request(method, f'{url}{path}', json=body, params=params,
                               headers={'Authorization': authorization, 'Accept': 'application/json'},
                               timeout=TIMEOUT_SECONDS)
    except requests.Timeout:
        raise ActionApiError("The orders service took too long to answer. Check the order before trying again - "
                             "the change may have been made.", uncertain=True)
    except requests.ConnectionError as e:
        log.error("%s %s connection failed: %s", method, path, e)
        raise ActionApiError("Could not reach the orders service. Please try again.")

    try:
        data = res.json()
    except ValueError:
        log.error("%s %s returned non-JSON (HTTP %s): %s", method, path, res.status_code, res.text[:300])
        raise ActionApiError(f"The orders service returned an unexpected answer (HTTP {res.status_code}). "
                             "Check the order before trying again.", uncertain=res.status_code < 500)

    if res.status_code in (401, 403):
        raise ActionApiError(_message(data, "Your session has expired. Please sign in again."))
    if not isinstance(data, dict) or not data.get('success'):
        raise ActionApiError(_message(data, "The orders service did not accept the change."))
    return data.get('data')


def roster_phlebos(authorization, visit_id, area_code_id):
    """Phlebos the CRM's Assign Phlebo box offers for this visit
    (GET /roster_phlebo_listing/{area_code}?visit_id=: roster, leave, overlap,
    area/radius rules). Returns [] when none is available."""
    url = base_url()
    if not url:
        return []
    try:
        res = requests.get(f'{url}/roster_phlebo_listing/{int(area_code_id or 0)}',
                           params={'visit_id': visit_id},
                           headers={'Authorization': authorization, 'Accept': 'application/json'},
                           timeout=20)
        data = res.json()
    except (requests.RequestException, ValueError) as e:
        log.warning("roster listing for visit %s failed: %s", visit_id, e)
        return []
    if not isinstance(data, dict) or not data.get('success') or not isinstance(data.get('data'), list):
        return []
    return data['data']
