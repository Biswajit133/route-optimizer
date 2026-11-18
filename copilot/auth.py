"""Who is asking: verifies the CRM user's JWT and loads what scoping needs.

customer_id and user_id come ONLY from the verified token, never from the
request body or from the model. Mirrors the PHP CustomerAccessGuard so a lab
that is blocked in the CRM is blocked here too (same 401 codes, which the
Angular account-blocked interceptor already understands).
"""
import os
import time
from dataclasses import dataclass, field
from datetime import date

import jwt
from flask import request

from . import db

# Keep in sync with CustomerAccessGuard::MAX_SESSION_DAYS and the gateway TTL.
MAX_SESSION_DAYS = 7

BLOCKED = 'ACCOUNT_BLOCKED'
EXPIRED = 'SESSION_EXPIRED'
EXPIRED_MESSAGE = 'Your session has expired. Please sign in again.'


class AuthError(Exception):
    def __init__(self, message, status=401, code=None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


@dataclass
class Principal:
    user_id: int
    customer_id: int
    user_name: str
    role_id: int
    user_lab: str
    is_facility: int
    user_facility: str
    customer: dict
    city_ids: list = field(default_factory=list)
    lab_location_ids: list = field(default_factory=list)

    @property
    def customer_name(self):
        return self.customer.get('name') or ''

    @property
    def tz_offset(self):
        """Customer timezone as '+05:30'. Most customers are IST; NULL means IST."""
        return self.customer.get('timezone') or '+05:30'


def _claims():
    secret = os.environ.get('JWT_SECRET')
    if not secret:
        raise AuthError("JWT_SECRET is not configured on this service", 500)

    header = request.headers.get('Authorization', '').strip()
    if header[:7].lower() != 'bearer ':
        raise AuthError("Token is required")
    token = header[7:].strip()
    if token.lower() in ('', 'null', 'undefined'):
        raise AuthError("Token is required")

    try:
        # tymon puts the user id in `sub` as a JSON number; PyJWT >= 2.10 rejects
        # a non-string sub unless told not to check it.
        claims = jwt.decode(token, secret, algorithms=['HS256'], options={'verify_sub': False})
    except jwt.ExpiredSignatureError:
        raise AuthError(EXPIRED_MESSAGE, 401, EXPIRED)
    except jwt.InvalidTokenError:
        raise AuthError("Token is invalid")

    issued_at = int(claims.get('iat') or 0)
    if issued_at and issued_at < time.time() - MAX_SESSION_DAYS * 86400:
        raise AuthError(EXPIRED_MESSAGE, 401, EXPIRED)

    return claims


def block_reason(customer):
    """Same rules as CustomerAccessGuard::reasonFor."""
    if not customer:
        return 'This account has been deactivated. Please contact support.'
    if customer.get('deleted_at') or str(customer.get('status') or '').strip().lower() == 'inactive':
        return 'This account has been deactivated. Please contact support.'
    if customer.get('is_trial') and customer.get('trial_ends_at'):
        ends = customer['trial_ends_at']
        ends = ends.isoformat() if isinstance(ends, date) else str(ends)[:10]
        if ends < date.today().isoformat():
            return 'Your trial has expired. Please contact support to continue.'
    return None


def authenticate():
    """Returns a Principal or raises AuthError. One small read per request."""
    claims = _claims()

    # CRM tokens are minted by the gateway with user_type 'admin'. Phlebo,
    # runner and patient-app tokens are not allowed to use the Copilot.
    if claims.get('user_type') != 'admin':
        raise AuthError("Patho Assist is available to CRM users only", 403)

    customer_id = claims.get('customer_id')
    user_id = claims.get('sub') or claims.get('id')
    if not customer_id or not user_id:
        raise AuthError("Token carries no customer or user")

    with db.read() as conn:
        customer = db.fetch_one(conn, "SELECT * FROM customers WHERE id = %s", (customer_id,))
        reason = block_reason(customer)
        if reason:
            raise AuthError(reason, 401, BLOCKED)

        user = db.fetch_one(conn, """
            SELECT id, name, first_name, last_name, customer_id, role_id, status,
                   user_lab, is_facility, user_facility
            FROM users WHERE id = %s AND deleted_at IS NULL
        """, (user_id,))
        if not user or int(user['customer_id'] or 0) != int(customer_id):
            raise AuthError("User not found for this account")
        if str(user['status']) == '0':
            raise AuthError('Your account has been disabled. Please contact your administrator.', 401, BLOCKED)

        if 'copilot_feature' not in customer:
            raise AuthError("Patho Assist is not set up yet (customers.copilot_feature is missing)", 403, 'COPILOT_DISABLED')
        if not customer.get('copilot_feature'):
            raise AuthError("Patho Assist is not enabled for your account", 403, 'COPILOT_DISABLED')

        cities = db.fetch_all(conn, """
            SELECT city_id FROM user_cities WHERE user_id = %s AND deleted_at IS NULL
        """, (user_id,))
        lab_locations = []
        if int(user['role_id'] or 0) == 25:
            lab_locations = db.fetch_all(conn, """
                SELECT lab_location_id FROM user_lab_locations WHERE user_id = %s AND deleted_at IS NULL
            """, (user_id,))

    name = (user.get('name') or ' '.join(filter(None, [user.get('first_name'), user.get('last_name')]))).strip()

    return Principal(
        user_id=int(user['id']),
        customer_id=int(customer_id),
        user_name=name,
        role_id=int(user['role_id'] or 0),
        user_lab=str(user['user_lab'] or ''),
        is_facility=int(user['is_facility'] or 0),
        user_facility=str(user['user_facility'] or ''),
        customer=customer,
        city_ids=[int(r['city_id']) for r in cities if r['city_id']],
        lab_location_ids=[int(r['lab_location_id']) for r in lab_locations if r['lab_location_id']],
    )
