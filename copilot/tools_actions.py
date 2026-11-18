"""Order actions from chat: reschedule, cancel, assign phlebo, payment link.

Same pattern as order create (tools_order): the model can only call a
prepare_* tool, which checks everything and saves a 'pending' row in
copilot_action_drafts. The CRM shows the row as a card; the change happens
only when the user clicks the card's button (routes.confirm_action ->
confirm() here -> actions_api with the user's own JWT, i.e. the same orders
API the CRM buttons call).

Who may do what mirrors the CRM order list buttons (order-list.component):
- reschedule: 'orders-reschedule'; never once the visit reached Completed,
  Sample Collected, Cancelled, Enroute To Lab or Dropped In Lab (current
  status or history); customer 95 non-admins only while Accepted/Rescheduled.
- cancel: 'orders-cancel'; same status rules; not the second visit of a
  two-visit test. Cancels the whole order, like the CRM.
- assign: 'orders-assign_phlebo'; status Accepted / Phlebo Assigned /
  Rescheduled. Phlebos come from the CRM's own roster listing API.
- payment link: 'orders-edit'; amount to collect > 0 and not paid.
Device-service customers are refused (the CRM needs extra confirm steps).
"""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

from flask import request

from . import actions_api, clock, db
from .scope import visit_scope
from .tools_order import (DRAFT_MINUTES, _area_slot_full, _check_date, _clean, _norm_slot, _slots_for,
                          currency_symbol)
from .tools_read import PATIENT_NAME_SQL, mask_phone, order_link

log = logging.getLogger('copilot.actions')

MAX_ASSIGN = 15          # orders on one assign card
MAX_PHLEBO_OPTIONS = 5
DEFAULT_SLOT_CAP = 3     # setting.phlebo_assign: visits per phlebo per slot (assign_phlebo)
LOCKED_STATUSES = {'completed', 'sample collected', 'cancelled', 'enroute to lab', 'dropped in lab'}
ASSIGNABLE = ('Accepted', 'Phlebo Assigned', 'Rescheduled')
PAID = {'paid', 'success', 'successful', 'completed', 'captured'}   # visits.payment_status; 'success' = cash taken / marked paid / paid online
ACTION_WORDS = {'reschedule': 'rescheduled', 'cancel': 'cancelled'}

VISIT_SELECT = f"""
    SELECT v.id AS visit_id, v.order_id, v.order_number, v.status, v.slot_date, v.slot_time,
           {PATIENT_NAME_SQL} AS patient_name, v.patient_phone, v.phlebo_id,
           TRIM(CONCAT_WS(' ', ph.first_name, ph.last_name)) AS phlebo_name,
           v.payment_status, v.amount_to_collect, v.total_order_value, v.visit_number,
           v.patient_pincode, v.patient_areacode_id, v.city_id, c.city_name AS city,
           (SELECT pc.area_code_id FROM pincodes pc
             WHERE pc.pincode = v.patient_pincode AND pc.customer_id = v.customer_id AND pc.deleted_at IS NULL
             ORDER BY pc.id LIMIT 1) AS area_code_id,
           (SELECT COUNT(*) FROM visit_patients vp WHERE vp.visit_id = v.id AND vp.deleted_at IS NULL) AS patient_count,
           o.payment_type AS order_payment_type, o.razorpay_payment_id
    FROM visits v
    LEFT JOIN phlebos ph ON ph.id = v.phlebo_id
    LEFT JOIN cities c ON c.id = v.city_id
    LEFT JOIN orders o ON o.id = v.order_id
"""


# ------------------------------------------------------------------ small helpers

def _as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], '%Y-%m-%d').date()
    except (TypeError, ValueError):
        return None


def _label(day):
    day = _as_date(day)
    return day.strftime('%a, %d %b') if day else ''


def _area(v):
    return ' '.join(x for x in [str(v.get('city') or '').strip(), str(v.get('patient_pincode') or '').strip()] if x)


def _patient(v):
    name = (v.get('patient_name') or '').strip() or 'Patient'
    more = int(v.get('patient_count') or 0) - 1
    return f"{name} + {more} more" if more > 0 else name


def _utc_now():
    return datetime.utcnow().replace(microsecond=0)


def has_permission(p, name):
    """The role's permission, the same lookup as the CRM menu: role_has_permissions
    for this customer, else the customer_id 0 defaults."""
    with db.read() as conn:
        rows = db.fetch_all(conn, """
            SELECT r.customer_id, MAX(pm.name = %s) AS allowed
            FROM role_has_permissions r
            JOIN permissions pm ON pm.id = r.permission_id AND pm.deleted_at IS NULL
            WHERE r.role_id = %s AND r.customer_id IN (%s, 0)
            GROUP BY r.customer_id
        """, (name, p.role_id, p.customer_id))
    by_customer = {int(r['customer_id']): int(r['allowed'] or 0) for r in rows}
    return bool(by_customer[p.customer_id] if p.customer_id in by_customer else by_customer.get(0, 0))


PERMISSIONS = {
    'reschedule': ('orders-reschedule', 'reschedule orders'),
    'cancel': ('orders-cancel', 'cancel orders'),
    'assign': ('orders-assign_phlebo', 'assign phlebos'),
    'payment_link': ('orders-edit', 'send payment links'),
}


def allowed(p, action):
    """(ok, reason)."""
    if p.customer.get('is_device_service'):
        return False, "Orders of device-service accounts are changed from the Orders page; Patho Assist cannot do it yet."
    perm, words = PERMISSIONS[action]
    if not has_permission(p, perm):
        return False, f"This user's role does not have permission to {words}."
    return True, None


# ------------------------------------------------------------------ visits

def _find_visit(conn, p, ref):
    """(visit row, problem). Only visits this user can open in the CRM."""
    ref = str(ref or '').strip()
    if not ref:
        return None, "Which order? Give the order number."
    scope_sql, scope_params = visit_scope(p)
    match, params = ["v.order_number = %s", "v.order_ref_number = %s"], [ref, ref]
    if ref.isdigit():
        match += ["v.order_number LIKE %s", "v.id = %s", "v.order_id = %s"]
        params += ['%-' + ref, int(ref), int(ref)]
    rows = db.fetch_all(conn, VISIT_SELECT + f"""
        WHERE {scope_sql} AND ({' OR '.join(match)})
        ORDER BY v.visit_number = 'second', v.id DESC LIMIT 5
    """, scope_params + params)
    if not rows:
        return None, f'No order matching "{ref}" that this user can open.'
    numbers = list(dict.fromkeys(r['order_number'] for r in rows))
    if len(numbers) > 1:
        return None, f'"{ref}" matches several orders ({", ".join(numbers[:4])}). Ask which one.'
    return rows[0], None


def _history(conn, visit_id):
    rows = db.fetch_all(conn, "SELECT status FROM visit_status WHERE visit_id = %s AND deleted_at IS NULL", (visit_id,))
    return {str(r['status'] or '').strip().lower() for r in rows}


def _status_problem(conn, p, v, action):
    status = str(v.get('status') or '').strip()
    number = v['order_number']
    if action in ('reschedule', 'cancel'):
        if int(p.customer_id) == 95 and p.role_id != 11 and status not in ('Accepted', 'Rescheduled'):
            return f"Order {number} is {status}; it can be changed only while Accepted or Rescheduled."
        reached = (_history(conn, v['visit_id']) | {status.lower()}) & LOCKED_STATUSES
        if reached:
            word = sorted(reached)[0].title()
            return f"Order {number} has reached {word}, so it can no longer be {ACTION_WORDS[action]}."
        if action == 'cancel' and v.get('visit_number') == 'second':
            return f"Order {number} is the second visit of a two-visit test. Cancel the main visit instead."
    elif action == 'assign':
        if status not in ASSIGNABLE:
            return f"Order {number} is {status or 'not open'}; a phlebo can be assigned only while it is Accepted, " \
                   f"Phlebo Assigned or Rescheduled."
    elif action == 'payment_link':
        if status.lower() == 'cancelled':
            return f"Order {number} is cancelled."
        if str(v.get('payment_status') or '').strip().lower() in PAID:
            return f"Order {number} is already paid."
        if float(v.get('amount_to_collect') or 0) <= 0:
            return f"Order {number} has nothing to collect (amount to collect is 0)."
    return None


def _paid_online(v):
    return bool(v.get('razorpay_payment_id')) or str(v.get('payment_status') or '').strip().lower() in PAID


# ------------------------------------------------------------------ reasons

REASON_TABLES = {'reschedule': 'reschedule_reason', 'cancel': 'cancellation_reason'}


def _reasons(conn, p, action):
    """The lab's own reasons (Settings > Visit Reasons), as the CRM dropdown lists them."""
    return [{'id': int(r['id']), 'reason': str(r['reason']).strip()} for r in db.fetch_all(conn, f"""
        SELECT id, reason FROM {REASON_TABLES[action]}
        WHERE customer_id = %s AND for_customer = 0 AND deleted_at IS NULL ORDER BY id
    """, (p.customer_id,)) if str(r['reason'] or '').strip()]


_WORD = re.compile(r'[a-z]{3,}')


def _pick_reason(reasons, text):
    """The reason that best matches what the user said, or None (the user picks on the card)."""
    text = str(text or '').strip().lower()
    if not text or not reasons:
        return None
    for r in reasons:
        if r['reason'].lower() == text:
            return r['id']
    words = set(_WORD.findall(text))
    best, score = None, 0
    for r in reasons:
        s = len(words & set(_WORD.findall(r['reason'].lower())))
        if s > score:
            best, score = r['id'], s
    return best


def _reason_name(summary):
    return next((r['reason'] for r in summary.get('reasons') or [] if r['id'] == summary.get('reason_id')), None)


# ------------------------------------------------------------------ drafts

def _save(p, ctx, action, payload, summary):
    expires = _utc_now() + timedelta(minutes=DRAFT_MINUTES)
    with db.write() as conn:
        db.execute(conn, """
            UPDATE copilot_action_drafts SET status = 'replaced', updated_at = NOW()
            WHERE conversation_id = %s AND customer_id = %s AND user_id = %s AND status = 'pending'
        """, (ctx['conversation_id'], p.customer_id, p.user_id))
        draft_id = db.execute(conn, """
            INSERT INTO copilot_action_drafts
                (customer_id, user_id, conversation_id, action, payload, summary, status, expires_at, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s, NOW(), NOW())
        """, (p.customer_id, p.user_id, ctx['conversation_id'], action,
              json.dumps(payload, ensure_ascii=False, default=str), json.dumps(summary, ensure_ascii=False, default=str),
              expires.strftime('%Y-%m-%d %H:%M:%S')))
    ctx.setdefault('actions', []).append(draft_id)
    return draft_id


def card_from_row(row):
    if not row:
        return None
    card = json.loads(row['summary'] or '{}')
    status = row['status']
    if status == 'pending' and row['expires_at'] and row['expires_at'] < _utc_now():
        status = 'expired'
    card.update({
        'action_id': int(row['id']),
        'action': row['action'],
        'status': status,
        'error': row.get('error'),
        'result': json.loads(row['result']) if row.get('result') else None,
        'expires_at': row['expires_at'].strftime('%Y-%m-%dT%H:%M:%SZ') if row.get('expires_at') else None,
    })
    return card


def get_draft(p, draft_id):
    with db.write() as conn:
        return db.fetch_one(conn, """
            SELECT * FROM copilot_action_drafts WHERE id = %s AND customer_id = %s AND user_id = %s
        """, (draft_id, p.customer_id, p.user_id))


def attach_message(draft_ids, message_id):
    if not draft_ids or not message_id:
        return
    marks, params = db.in_list(draft_ids)
    with db.write() as conn:
        db.execute(conn, f"UPDATE copilot_action_drafts SET message_id = %s WHERE id IN ({marks})",
                   (message_id, *params))


def _rows_for_messages(message_ids):
    if not message_ids:
        return []
    marks, params = db.in_list(message_ids)
    with db.write() as conn:
        return db.fetch_all(conn, f"SELECT * FROM copilot_action_drafts WHERE message_id IN ({marks}) ORDER BY id", params)


def cards_for_messages(message_ids):
    return {int(r['message_id']): card_from_row(r) for r in _rows_for_messages(message_ids)}


STATUS_NOTES = {
    'pending': 'card shown, waiting for the user to click (NOT done yet)',
    'running': 'user clicked, change in progress',
    'done': 'user confirmed, DONE',
    'partial': 'user confirmed, PARTLY done',
    'failed': 'user confirmed but it FAILED',
    'cancelled': 'user closed the card, nothing changed',
    'replaced': 'replaced by a newer card',
    'expired': 'card expired, nothing changed',
}
ACTION_NAMES = {'reschedule': 'Reschedule', 'cancel': 'Cancel order', 'assign': 'Assign phlebo',
                'payment_link': 'Payment link'}


def notes_for_messages(message_ids):
    """{message_id: '[... card: ...]'} so the model knows what became of each card."""
    out = {}
    for r in _rows_for_messages(message_ids):
        card = card_from_row(r)
        what = card.get('order_number') or f"{len(card.get('rows') or [])} orders"
        note = STATUS_NOTES.get(card['status'], card['status'])
        if card.get('error'):
            note += f": {card['error']}"
        out[int(r['message_id'])] = f"[{ACTION_NAMES.get(r['action'], r['action'])} card for {what}: {note}]"
    return out


# ------------------------------------------------------------------ prepare: reschedule

def _slot_problem(conn, p, v, day, wanted_text):
    """(slot, problem) for a reschedule date + slot."""
    slots = _slots_for(conn, p, day)
    wanted = _norm_slot(wanted_text)
    slot = next((s for s in slots if _norm_slot(s) == wanted), None)
    if not wanted:
        return None, "New slot is missing."
    if not slot:
        return None, f"Slot '{wanted_text}' is not available on {day.isoformat()}. Call available_slots and offer one of those."
    if _as_date(v['slot_date']) == day and _norm_slot(v['slot_time']) == wanted:
        return None, f"Order {v['order_number']} is already on {day.isoformat()} {slot}."
    loc = {'city_id': v.get('city_id'), 'area_code_id': v.get('patient_areacode_id')}
    if loc['city_id'] and _area_slot_full(conn, p, loc, day, slot):
        return None, f"Slot {slot} is full for this area on {day.isoformat()}. Offer another slot."
    return slot, None


def prepare_reschedule(p, args, ctx):
    ok, reason = allowed(p, 'reschedule')
    if not ok:
        return {'error': reason}
    problems = []
    with db.read() as conn:
        v, problem = _find_visit(conn, p, args.get('order_ref'))
        if problem:
            return {'ok': False, 'problems': [problem]}
        problem = _status_problem(conn, p, v, 'reschedule')
        if problem:
            return {'ok': False, 'problems': [problem]}
        day, problem = _check_date(p, args.get('date'))
        slot = None
        if problem:
            problems.append(problem)
        else:
            slot, problem = _slot_problem(conn, p, v, day, args.get('slot'))
            if problem:
                problems.append(problem)
        reasons = _reasons(conn, p, 'reschedule')
    if problems:
        return {'ok': False, 'problems': problems,
                'note': 'No card was shown. Fix these with the user (or use available_slots), then call prepare_reschedule again.'}
    if not reasons:
        return {'error': 'This lab has no reschedule reasons yet. Ask an admin to add one in Settings > Visit Reasons.'}

    reason_id = _pick_reason(reasons, args.get('reason'))
    note = _clean(args.get('note'), 500)
    today = clock.today_local(p.tz_offset)
    payload = {'order_id': v['order_id'], 'visit_id': v['visit_id'], 'date': day.isoformat(), 'slot': slot,
               'reason_id': reason_id, 'reschedule_note': note}
    summary = {
        'order_number': v['order_number'], 'link': order_link(v['order_id'], v['visit_id']),
        'patient': _patient(v), 'area': _area(v), 'phlebo': v.get('phlebo_name') or None,
        'was_label': f"{_label(v['slot_date'])} · {v['slot_time'] or ''}".strip(' ·'),
        'date': day.isoformat(), 'date_label': _label(day), 'slot': slot,
        'reason_id': reason_id, 'reasons': reasons, 'note': note or None,
        'min_date': today.isoformat(), 'max_date': (today + timedelta(days=15)).isoformat(),
    }
    _save(p, ctx, 'reschedule', payload, summary)
    msg = ('The Reschedule card is now shown below your reply. Nothing is changed yet: the user must click Reschedule '
           'on it. Reply in one or two lines.')
    if not reason_id:
        msg += ' No reason matched: tell them to pick the reason on the card (do not ask in chat).'
    return {'ok': True, 'card_shown': True, 'order': v['order_number'],
            'from': summary['was_label'], 'to': f"{summary['date_label']} · {slot}",
            'reason': _reason_name(summary), 'note': msg}


# ------------------------------------------------------------------ prepare: cancel

def prepare_cancel(p, args, ctx):
    ok, reason = allowed(p, 'cancel')
    if not ok:
        return {'error': reason}
    with db.read() as conn:
        v, problem = _find_visit(conn, p, args.get('order_ref'))
        if problem:
            return {'ok': False, 'problems': [problem]}
        problem = _status_problem(conn, p, v, 'cancel')
        if problem:
            return {'ok': False, 'problems': [problem]}
        reasons = _reasons(conn, p, 'cancel')
        others = db.fetch_one(conn, """
            SELECT COUNT(*) AS n FROM visits
            WHERE order_id = %s AND id != %s AND deleted_at IS NULL AND (status IS NULL OR status != 'Cancelled')
        """, (v['order_id'], v['visit_id']))['n']
    if not reasons:
        return {'error': 'This lab has no cancellation reasons yet. Ask an admin to add one in Settings > Visit Reasons.'}

    reason_id = _pick_reason(reasons, args.get('reason'))
    note = _clean(args.get('note'), 500)
    currency = currency_symbol(p)
    warnings = []
    if _paid_online(v):
        warnings.append("Already paid. Cancelling does not refund it; use Cancellation Refund on the order after this.")
    if int(others or 0):
        warnings.append(f"This order has {int(others) + 1} visits; all of them are cancelled.")
    amount = float(v.get('total_order_value') or 0)
    payload = {'order_id': v['order_id'], 'visit_id': v['visit_id'], 'reason_id': reason_id, 'cancel_note': note}
    summary = {
        'order_number': v['order_number'], 'link': order_link(v['order_id'], v['visit_id']),
        'patient': _patient(v), 'area': _area(v),
        'slot_label': f"{_label(v['slot_date'])} · {v['slot_time'] or ''}".strip(' ·'),
        'amount': amount, 'currency': currency,
        'payment': ' · '.join(x for x in [v.get('order_payment_type'), 'paid' if _paid_online(v) else None] if x) or None,
        'reason_id': reason_id, 'reasons': reasons, 'note': note or None, 'warnings': warnings,
    }
    _save(p, ctx, 'cancel', payload, summary)
    msg = ('The Cancel card is now shown below your reply. The order is NOT cancelled yet: the user must click '
           '"Cancel order" on it. Reply in one or two lines.')
    if not reason_id:
        msg += ' No reason matched: tell them to pick the reason on the card.'
    if warnings:
        msg += ' Mention the warning on the card in a few words.'
    return {'ok': True, 'card_shown': True, 'order': v['order_number'], 'reason': _reason_name(summary),
            'warnings': warnings, 'note': msg}


# ------------------------------------------------------------------ prepare: payment link

def prepare_payment_link(p, args, ctx):
    ok, reason = allowed(p, 'payment_link')
    if not ok:
        return {'error': reason}
    with db.read() as conn:
        v, problem = _find_visit(conn, p, args.get('order_ref'))
        if problem:
            return {'ok': False, 'problems': [problem]}
        problem = _status_problem(conn, p, v, 'payment_link')
        if problem:
            return {'ok': False, 'problems': [problem]}
    sms_on = bool(p.customer.get('sms_feature'))
    summary = {
        'order_number': v['order_number'], 'link': order_link(v['order_id'], v['visit_id']),
        'patient': _patient(v), 'phone': mask_phone(v.get('patient_phone')),
        'amount': float(v.get('amount_to_collect') or 0), 'currency': currency_symbol(p), 'sms': sms_on,
    }
    _save(p, ctx, 'payment_link', {'visit_id': v['visit_id']}, summary)
    if sms_on:
        msg = 'The Payment link card is shown. Nothing is sent yet: the user must click "Send link" on it.'
    else:
        msg = ('The Payment link card is shown. SMS is off for this lab, so the link is only created and shown on the '
               'card for the user to copy and share; say that in a few words.')
    return {'ok': True, 'card_shown': True, 'order': v['order_number'],
            'amount': f"{summary['currency']}{summary['amount']:g}", 'note': msg}


# ------------------------------------------------------------------ prepare: assign

def _slot_cap(conn):
    row = db.fetch_one(conn, "SELECT value FROM setting WHERE keyword = 'phlebo_assign' AND deleted_at IS NULL LIMIT 1")
    try:
        return int(row['value']) if row else DEFAULT_SLOT_CAP
    except (TypeError, ValueError):
        return DEFAULT_SLOT_CAP


def _phlebo_name(r):
    name = (r.get('name') or ' '.join(x for x in [r.get('first_name'), r.get('last_name')] if x) or '').strip()
    return name or f"Phlebo {r.get('id')}"


def _unassigned(conn, p, day):
    scope_sql, scope_params = visit_scope(p)
    return db.fetch_all(conn, VISIT_SELECT + f"""
        WHERE {scope_sql} AND DATE(v.slot_date) = %s AND v.status IN ('Accepted', 'Rescheduled')
          AND (v.phlebo_id IS NULL OR v.phlebo_id = 0)
        ORDER BY v.slot_time, v.id LIMIT {MAX_ASSIGN + 1}
    """, scope_params + [day.isoformat()])


def prepare_assign(p, args, ctx):
    ok, reason = allowed(p, 'assign')
    if not ok:
        return {'error': reason}
    authorization = request.headers.get('Authorization', '')
    refs = [str(r).strip() for r in (args.get('order_refs') or []) if str(r).strip()]
    more = False
    with db.read() as conn:
        visits, problems = [], []
        if refs:
            for ref in list(dict.fromkeys(refs))[:MAX_ASSIGN]:
                v, problem = _find_visit(conn, p, ref)
                problem = problem or _status_problem(conn, p, v, 'assign')
                if problem:
                    problems.append(problem)
                elif all(x['visit_id'] != v['visit_id'] for x in visits):
                    visits.append(v)
            if problems:
                return {'ok': False, 'problems': problems, 'note': 'No card was shown. Tell the user, then try again with the right orders.'}
        elif args.get('unassigned'):
            day = clock.resolve_date(args.get('date'), p.tz_offset, clock.today_local(p.tz_offset))
            visits = _unassigned(conn, p, day)
            more = len(visits) > MAX_ASSIGN
            visits = visits[:MAX_ASSIGN]
            if not visits:
                return {'ok': True, 'card_shown': False, 'note': f'No order without a phlebo on {day.isoformat()} '
                                                                  '(among the orders this user can see). Tell the user.'}
        else:
            return {'error': 'Give order_refs, or set unassigned=true (with a date) for all orders without a phlebo.'}
        cap = _slot_cap(conn)

    # The CRM's own roster listing, once per visit (roster, leave, overlap, area rules).
    with ThreadPoolExecutor(max_workers=5) as pool:
        rosters = list(pool.map(lambda v: actions_api.roster_phlebos(authorization, v['visit_id'], v.get('area_code_id')),
                                visits))

    ids = sorted({int(r['id']) for roster in rosters for r in roster if str(r.get('id') or '').isdigit()})
    days = sorted({_as_date(v['slot_date']).isoformat() for v in visits if _as_date(v['slot_date'])})
    day_load, slot_load, pin_load = {}, {}, {}
    if ids and days:
        id_marks, id_params = db.in_list(ids)
        day_marks, day_params = db.in_list(days)
        with db.read() as conn:
            for r in db.fetch_all(conn, f"""
                SELECT phlebo_id, DATE(slot_date) AS d, slot_time, patient_pincode, COUNT(*) AS n FROM visits
                WHERE customer_id = %s AND phlebo_id IN ({id_marks}) AND DATE(slot_date) IN ({day_marks})
                  AND is_active = 1 AND deleted_at IS NULL AND (status IS NULL OR status != 'Cancelled')
                GROUP BY phlebo_id, DATE(slot_date), slot_time, patient_pincode
            """, [p.customer_id] + id_params + day_params):
                pid, d, n = int(r['phlebo_id']), _as_date(r['d']).isoformat(), int(r['n'])
                day_load[(pid, d)] = day_load.get((pid, d), 0) + n
                slot_key = (pid, d, _norm_slot(r['slot_time']))
                slot_load[slot_key] = slot_load.get(slot_key, 0) + n
                pin_key = (pid, d, str(r['patient_pincode'] or '').strip())
                pin_load[pin_key] = pin_load.get(pin_key, 0) + n

    # Greedy: earliest slot first; each suggestion counts toward the next rows,
    # so one free phlebo does not get every order.
    rows = []
    for v, roster in zip(visits, rosters):
        d = _as_date(v['slot_date']).isoformat() if _as_date(v['slot_date']) else ''
        slot = _norm_slot(v['slot_time'])
        pin = str(v.get('patient_pincode') or '').strip()
        current = int(v['phlebo_id'] or 0)
        options = []
        for r in roster:
            if not str(r.get('id') or '').isdigit():
                continue
            pid = int(r['id'])
            if pid != current and slot_load.get((pid, d, slot), 0) >= cap:
                continue
            load, near = day_load.get((pid, d), 0), pin_load.get((pid, d, pin), 0)
            hint = f"{load} order{'s' if load != 1 else ''} that day"
            if near and pin:
                hint += f" · {near} in {pin}"
            options.append({'phlebo_id': pid, 'name': _phlebo_name(r), 'load': load, 'near': near, 'hint': hint,
                            'current': pid == current})
        options.sort(key=lambda o: (-1 if o['near'] else 0, o['load'], o['name'].lower()))
        options = options[:MAX_PHLEBO_OPTIONS]
        pick = next((o for o in options if not o['current']), None)
        if pick:
            pick['suggested'] = True
            day_load[(pick['phlebo_id'], d)] = day_load.get((pick['phlebo_id'], d), 0) + 1
            slot_load[(pick['phlebo_id'], d, slot)] = slot_load.get((pick['phlebo_id'], d, slot), 0) + 1
            pin_load[(pick['phlebo_id'], d, pin)] = pin_load.get((pick['phlebo_id'], d, pin), 0) + 1
        rows.append({
            'visit_id': int(v['visit_id']), 'order_id': int(v['order_id']), 'order_number': v['order_number'],
            'link': order_link(v['order_id'], v['visit_id']), 'patient': _patient(v), 'area': _area(v),
            'date_label': _label(v['slot_date']), 'slot': v['slot_time'] or '',
            'current': v.get('phlebo_name') or None, 'options': options,
            'phlebo_id': pick['phlebo_id'] if pick else None,
            'note': None if options else 'No phlebo on the roster for this slot and area.',
        })

    summary = {'mode': 'single' if len(rows) == 1 else 'bulk', 'rows': rows, 'more': more,
               'date_label': rows[0]['date_label'] if len({r['date_label'] for r in rows}) == 1 else None}
    payload = {'rows': [{'visit_id': r['visit_id'], 'order_number': r['order_number'], 'phlebo_id': r['phlebo_id']}
                        for r in rows]}
    _save(p, ctx, 'assign', payload, summary)

    model_rows = []
    for r in rows:
        pick = next((o for o in r['options'] if o.get('suggested')), None)
        model_rows.append({'order': r['order_number'], 'slot': r['slot'], 'area': r['area'],
                           'suggested': f"{pick['name']} ({pick['hint']})" if pick else None,
                           'other_choices': max(0, len(r['options']) - 1), 'problem': r['note']})
    msg = ('The Assign card is now shown below your reply with a suggested phlebo per order (same-pincode orders that '
           'day first, then the lightest load). Nothing is assigned yet: the user can change any choice and must click '
           'Assign on the card. Reply in one or two lines; name the suggestion for a single order.')
    if more:
        msg += f' Only the first {MAX_ASSIGN} orders fit on the card; say there are more.'
    return {'ok': True, 'card_shown': True, 'orders': model_rows, 'note': msg}


# ------------------------------------------------------------------ card edits (no LLM)

class ActionProblem(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _pending(p, draft_id):
    row = get_draft(p, draft_id)
    if not row:
        raise ActionProblem("Card not found.", 404)
    card = card_from_row(row)
    if card['status'] != 'pending':
        raise ActionProblem("This card can no longer be changed." if card['status'] != 'expired'
                            else "This card has expired. Ask Patho Assist again.", 409)
    return row


def update(p, draft_id, body):
    """One change from the card. Returns the new card."""
    row = _pending(p, draft_id)
    action = row['action']
    payload = json.loads(row['payload'] or '{}')
    summary = json.loads(row['summary'] or '{}')
    changed = False

    if action in ('reschedule', 'cancel') and 'reason_id' in body:
        try:
            reason_id = int(body.get('reason_id'))
        except (TypeError, ValueError):
            raise ActionProblem("Pick a reason.")
        if reason_id not in {r['id'] for r in summary.get('reasons') or []}:
            raise ActionProblem("That reason is not in the list.")
        payload['reason_id'] = summary['reason_id'] = reason_id
        changed = True

    if action in ('reschedule', 'cancel') and 'note' in body:
        note = _clean(body.get('note'), 500)
        payload['reschedule_note' if action == 'reschedule' else 'cancel_note'] = note
        summary['note'] = note or None
        changed = True

    if action == 'reschedule' and (body.get('date') or body.get('slot')):
        day, problem = _check_date(p, body.get('date') or summary.get('date'))
        if problem:
            raise ActionProblem(problem)
        with db.read() as conn:
            v, problem = _find_visit(conn, p, str(payload['visit_id']))
            if problem:
                raise ActionProblem(problem)
            slot, problem = _slot_problem(conn, p, v, day, body.get('slot') or summary.get('slot'))
        if problem:
            raise ActionProblem(problem)
        payload.update(date=day.isoformat(), slot=slot)
        summary.update(date=day.isoformat(), date_label=_label(day), slot=slot)
        changed = True

    if action == 'assign' and isinstance(body.get('assign'), dict):
        try:
            visit_id, phlebo_id = int(body['assign'].get('visit_id')), int(body['assign'].get('phlebo_id') or 0)
        except (TypeError, ValueError):
            raise ActionProblem("Say which order and phlebo.")
        row_s = next((r for r in summary.get('rows') or [] if r['visit_id'] == visit_id), None)
        row_p = next((r for r in payload.get('rows') or [] if r['visit_id'] == visit_id), None)
        if not row_s or not row_p:
            raise ActionProblem("That order is not on this card.")
        if phlebo_id and phlebo_id not in {o['phlebo_id'] for o in row_s['options']}:
            raise ActionProblem("That phlebo is not available for this order.")
        row_s['phlebo_id'] = row_p['phlebo_id'] = phlebo_id or None
        changed = True

    if not changed:
        raise ActionProblem("Nothing to change.")
    with db.write() as conn:
        db.execute(conn, """
            UPDATE copilot_action_drafts SET payload = %s, summary = %s, updated_at = NOW()
            WHERE id = %s AND status = 'pending'
        """, (json.dumps(payload, ensure_ascii=False, default=str), json.dumps(summary, ensure_ascii=False, default=str),
              draft_id))
    return card_from_row(get_draft(p, draft_id))


def slot_choices(p, draft_id, date_value):
    row = _pending(p, draft_id)
    if row['action'] != 'reschedule':
        raise ActionProblem("Slots are only for a reschedule card.")
    day, problem = _check_date(p, date_value)
    if problem:
        raise ActionProblem(problem)
    with db.read() as conn:
        slots = _slots_for(conn, p, day)
    return {'date': day.isoformat(), 'date_label': _label(day), 'slots': slots}


# ------------------------------------------------------------------ confirm

def _claim(p, draft_id):
    """pending -> running, atomically, so a double click cannot run it twice."""
    row = get_draft(p, draft_id)
    if not row:
        raise ActionProblem("Card not found.", 404)
    card = card_from_row(row)
    if card['status'] == 'expired':
        raise ActionProblem("This card has expired. Ask Patho Assist again.", 409)
    if row['status'] != 'pending':
        raise ActionProblem({'running': "This is already in progress.", 'done': "This was already done.",
                             'partial': "This was already done.", 'cancelled': "This card was closed.",
                             'replaced': "A newer card replaced this one. Use the latest card."}
                            .get(row['status'], "This card can no longer be used."), 409)
    payload = json.loads(row['payload'] or '{}')
    if row['action'] in ('reschedule', 'cancel') and not payload.get('reason_id'):
        raise ActionProblem("Pick a reason first.", 409)
    if row['action'] == 'assign' and not any(r.get('phlebo_id') for r in payload.get('rows') or []):
        raise ActionProblem("Pick a phlebo for at least one order.", 409)
    with db.write() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE copilot_action_drafts SET status = 'running', confirmed_at = NOW(), updated_at = NOW()
                WHERE id = %s AND status = 'pending' AND expires_at >= UTC_TIMESTAMP()
            """, (draft_id,))
            if cur.rowcount != 1:
                raise ActionProblem("This card can no longer be used. Refresh the chat.", 409)
    return row, payload


def _finish(draft_id, status, result=None, error=None):
    with db.write() as conn:
        db.execute(conn, """
            UPDATE copilot_action_drafts SET status = %s, result = %s, error = %s, updated_at = NOW() WHERE id = %s
        """, (status, json.dumps(result, ensure_ascii=False, default=str) if result is not None else None,
              (error or '')[:500] or None, draft_id))


def confirm(p, draft_id, authorization):
    """Runs the card's change through the orders API. Returns the card;
    raises ActionProblem for a card that cannot run (nothing was sent)."""
    if not has_permission(p, PERMISSIONS[(get_draft(p, draft_id) or {}).get('action', 'cancel')][0]):
        raise ActionProblem("This user's role does not have permission for this.", 403)
    row, payload = _claim(p, draft_id)
    action = row['action']
    try:
        if action == 'reschedule':
            actions_api.call('POST', '/rescheduleOrder', authorization, {
                'order_id': payload['order_id'], 'visit_id': payload['visit_id'], 'date': payload['date'],
                'slot': payload['slot'], 'reason_id': payload['reason_id'],
                'reschedule_note': payload.get('reschedule_note') or None})
            _finish(draft_id, 'done')
        elif action == 'cancel':
            actions_api.call('POST', '/cancelOrder', authorization, {
                'order_id': payload['order_id'], 'visit_id': payload['visit_id'], 'reason_id': payload['reason_id'],
                'cancel_note': payload.get('cancel_note') or None})
            _finish(draft_id, 'done')
        elif action == 'payment_link':
            link = actions_api.call('POST', f"/send-paymentlink-customer/{int(payload['visit_id'])}", authorization)
            _finish(draft_id, 'done', result={'link': link if isinstance(link, str) else None})
        elif action == 'assign':
            results = []
            for r in payload.get('rows') or []:
                if not r.get('phlebo_id'):
                    results.append({'visit_id': r['visit_id'], 'order_number': r['order_number'], 'ok': None})
                    continue
                try:
                    actions_api.call('PUT', '/assign_phlebo', authorization,
                                     {'visit_id': r['visit_id'], 'phlebo_id': r['phlebo_id']})
                    results.append({'visit_id': r['visit_id'], 'order_number': r['order_number'], 'ok': True})
                except actions_api.ActionApiError as e:
                    results.append({'visit_id': r['visit_id'], 'order_number': r['order_number'], 'ok': False,
                                    'error': str(e)})
            done = sum(1 for x in results if x['ok'])
            tried = sum(1 for x in results if x['ok'] is not None)
            failed = [x for x in results if x['ok'] is False]
            status = 'done' if done == tried else ('partial' if done else 'failed')
            _finish(draft_id, status, result={'rows': results, 'done': done, 'tried': tried},
                    error=f"{len(failed)} not assigned: {failed[0]['error']}" if failed else None)
        log.info("Copilot action %s (%s) by user %s of customer %s finished", draft_id, action, p.user_id, p.customer_id)
    except actions_api.ActionApiError as e:
        log.warning("Copilot action %s (%s) failed: %s", draft_id, action, e)
        _finish(draft_id, 'failed', error=str(e))
    except Exception as e:
        log.exception("Copilot action %s (%s) crashed", draft_id, action)
        _finish(draft_id, 'failed', error="Something went wrong. Check the order before trying again.")
    return card_from_row(get_draft(p, draft_id))


def close(p, draft_id):
    with db.write() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE copilot_action_drafts SET status = 'cancelled', updated_at = NOW()
                WHERE id = %s AND customer_id = %s AND user_id = %s AND status = 'pending'
            """, (draft_id, p.customer_id, p.user_id))
    return card_from_row(get_draft(p, draft_id))


# ------------------------------------------------------------------ registry

_ORDER_REF = {'type': 'string', 'description': 'Order number as the user wrote it, e.g. "NOV-3156623" or "3156623".'}

TOOLS = {
    'prepare_reschedule': {
        'handler': prepare_reschedule,
        'description': "Check a reschedule of one order to a new date and slot, and show the user a Reschedule card. "
                       "This does NOT reschedule: the user must click Reschedule on the card. Run available_slots for "
                       "the new date first and pass an exact slot from it.",
        'parameters': {'type': 'object', 'properties': {
            'order_ref': _ORDER_REF,
            'date': {'type': 'string', 'description': "New date: YYYY-MM-DD, 'today' or 'tomorrow'."},
            'slot': {'type': 'string', 'description': 'Exact slot text from available_slots.'},
            'reason': {'type': 'string', 'description': 'Why, in the user\'s words, if they gave it. Matched to the lab\'s reasons.'},
            'note': {'type': 'string', 'description': 'Optional note.'},
        }, 'required': ['order_ref', 'date', 'slot']},
    },
    'prepare_cancel': {
        'handler': prepare_cancel,
        'description': "Check a cancellation of one order and show the user a Cancel card. This does NOT cancel: the "
                       "user must click Cancel order on the card. The whole order is cancelled.",
        'parameters': {'type': 'object', 'properties': {
            'order_ref': _ORDER_REF,
            'reason': {'type': 'string', 'description': 'Why, in the user\'s words, if they gave it. Matched to the lab\'s reasons.'},
            'note': {'type': 'string', 'description': 'Optional note, e.g. what the patient said.'},
        }, 'required': ['order_ref']},
    },
    'prepare_assign': {
        'handler': prepare_assign,
        'description': "Suggest phlebos from the roster and show an Assign card. Either for given orders (order_refs), "
                       "or for every order without a phlebo on a date (unassigned=true). This does NOT assign: the user "
                       "can change the choices and must click Assign on the card. Also use it to answer 'who should "
                       "I give this order to?'.",
        'parameters': {'type': 'object', 'properties': {
            'order_refs': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': MAX_ASSIGN,
                           'description': 'Order numbers to assign.'},
            'unassigned': {'type': 'boolean', 'description': 'True = all orders without a phlebo on `date`.'},
            'date': {'type': 'string', 'description': "With unassigned: YYYY-MM-DD or 'today'/'tomorrow'. Default today."},
        }},
    },
    'prepare_payment_link': {
        'handler': prepare_payment_link,
        'description': "Show a card to send the patient a payment link (SMS) for the amount still to collect on one "
                       "order. Nothing is sent until the user clicks Send link on the card.",
        'parameters': {'type': 'object', 'properties': {'order_ref': _ORDER_REF}, 'required': ['order_ref']},
    },
}
