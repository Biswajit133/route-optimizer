"""Edits on a pending order card, made by clicks on the card (no LLM call).

- address: switch to another saved address of the same phone (customer_history refs A1..A3)
- payment: Prepaid / Postpaid (Confirm needs one, see tools_order.claim)
- test: swap a test for one of the other matches of the same search ("CBC" -> "CBC with ESR")

prepare_order calls card_extras() so the card knows its choices; the CRM then
calls POST /copilot/drafts/<id>/update and gets the new card back.
"""
import json
from datetime import datetime, timedelta

from . import db

MAX_TEST_OPTIONS = 6
PAYMENT_TYPES = ('Prepaid', 'Postpaid')


class EditProblem(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ choices for a new card

def _lookup(p, conversation_id):
    with db.write() as conn:
        row = db.fetch_one(conn, """
            SELECT phone, data FROM copilot_customer_lookups
            WHERE conversation_id = %s AND customer_id = %s AND user_id = %s
            ORDER BY id DESC LIMIT 1
        """, (conversation_id, p.customer_id, p.user_id))
    if not row:
        return None, []
    return row['phone'], (json.loads(row['data'] or '{}').get('card') or {}).get('addresses') or []


def address_choices(p, conversation_id, phone):
    """Saved addresses the card may switch to: only those looked up for this same phone."""
    try:
        looked_up, addresses = _lookup(p, conversation_id)
    except Exception:  # copilot_customer_lookups not created yet
        return []
    if not looked_up or str(looked_up)[-10:] != str(phone or '')[-10:]:
        return []
    return [{'ref': a['ref'], 'number': a['number'], 'line': a['line'], 'place': a['place']} for a in addresses]


def _option(r):
    return {'test_id': int(r['id']), 'name': str(r['test_name']).strip(), 'code': str(r.get('test_code') or '').strip() or None,
            'price': float(r['discount_price'] or 0)}


def test_options(p, conn, ctx, patients):
    """Adds 'options' (other bookable tests of the same search) to each test that has some."""
    from .tools_order import _test_rows
    searched = (ctx or {}).get('test_options') or {}
    ids = {t['test_id'] for pt in patients for t in pt['tests']}
    codes = {}
    if ids:
        marks, params = db.in_list(ids)
        codes = {int(r['id']): str(r['test_code'] or '').strip()
                 for r in db.fetch_all(conn, f"SELECT id, test_code FROM customer_dos WHERE id IN ({marks})", params)}
    for pt in patients:
        for t in pt['tests']:
            options = searched.get(t['test_id'])
            if options is None and len(codes.get(t['test_id'], '')) >= 2:
                # searched in an earlier turn: same lab tests whose name or code has this test's code
                code = codes[t['test_id']]
                options = [_option(r) for r in _test_rows(conn, p.customer_id, "test_name LIKE %s OR test_code LIKE %s",
                                                          (f'%{code}%', f'%{code}%'), MAX_TEST_OPTIONS + 1)]
            options = [o for o in (options or []) if o['test_id'] != t['test_id']][:MAX_TEST_OPTIONS - 1]
            if options:
                current = next((o for o in (searched.get(t['test_id']) or []) if o['test_id'] == t['test_id']), None)
                t['options'] = [{'test_id': t['test_id'], 'name': t['name'], 'price': t['price'],
                                 'code': (current or {}).get('code') or codes.get(t['test_id']) or None}] + options


def remember_search(ctx, rows):
    """search_tests: keep every match, so the card can offer the others."""
    if ctx is None or len(rows) < 2:
        return
    options = [_option(r) for r in rows][:MAX_TEST_OPTIONS]
    store = ctx.setdefault('test_options', {})
    for o in options:
        store[o['test_id']] = options


# ------------------------------------------------------------------ edit a pending card

def _pending(p, draft_id):
    from .tools_order import card_from_row, get_draft
    row = get_draft(p, draft_id)
    if not row:
        raise EditProblem("Order card not found.", 404)
    status = card_from_row(row)['status']
    if status == 'expired':
        raise EditProblem("This card has expired. Ask Patho Assist to prepare the order again.", 409)
    if status != 'pending':
        raise EditProblem("This card can no longer be changed.", 409)
    return row


def _set_address(p, row, payload, summary, ref):
    from .tools_customer import saved_address
    from .tools_order import _area_slot_full, _resolve_location
    entry, phone, problem = saved_address(p, row['conversation_id'], ref)
    if problem:
        raise EditProblem(problem, 409)
    if str(phone or '')[-10:] != str(payload.get('contact_number') or '')[-10:]:
        raise EditProblem("That saved address belongs to another phone number.", 409)
    with db.read() as conn:
        loc, problem = _resolve_location(conn, p, entry.get('pincode'), entry.get('city'))
        if problem:
            raise EditProblem(problem, 409)
        day = datetime.strptime(payload['date'], '%Y-%m-%d').date()
        if _area_slot_full(conn, p, loc, day, payload['slot']):
            raise EditProblem(f"Slot {payload['slot']} is full for that area. Ask Patho Assist for another slot.", 409)
    street = str(entry.get('street') or '')[:250]
    address_id = int(entry['address_id']) if entry.get('address_id') else 0
    address_line = ', '.join(x for x in [street, loc['city'], loc['state'], loc['pincode']] if x)
    payload.update(address_id=address_id, contact_address='' if address_id else address_line, address=street,
                   city=loc['city'], state=loc['state'], pincode=loc['pincode'])
    summary.update(address=address_line, saved_address=True, address_ref=entry['ref'])


def _set_test(p, payload, summary, patient, from_id, to_id):
    from .tools_order import _test_rows
    patients = summary.get('patients') or []
    if not (0 <= patient < len(patients)):
        raise EditProblem("Patient not found on this card.")
    tests = patients[patient]['tests']
    test = next((t for t in tests if t['test_id'] == from_id), None)
    if not test:
        raise EditProblem("That test is not on this card any more.")
    if to_id == from_id:
        return
    if to_id not in {o['test_id'] for o in test.get('options') or []}:
        raise EditProblem("Pick one of the listed tests.")
    if any(t['test_id'] == to_id for t in tests):
        raise EditProblem("This patient already has that test.")
    with db.read() as conn:
        rows = _test_rows(conn, p.customer_id, "id = %s", (to_id,), 1)
    if not rows:
        raise EditProblem("That test is not bookable any more.", 409)
    new = _option(rows[0])
    test.update(test_id=new['test_id'], name=new['name'], price=new['price'])
    for o in test['options']:  # keep the list, with today's price for the picked one
        if o['test_id'] == new['test_id']:
            o.update(name=new['name'], price=new['price'], code=new['code'])

    pt = patients[patient]
    pt['amount'] = round(sum(t['price'] for t in pt['tests']), 2)
    summary['total'] = round(sum(x['amount'] for x in patients), 2)
    out = payload['patients'][patient]
    out['amount'] = pt['amount']
    out['test'] = [{'value': t['test_id'], 'text': t['name']} for t in pt['tests']]


def _clean(value, limit):
    return ' '.join(str(value or '').split())[:limit]


def _patient_fields(data, base=None):
    """name / age / gender from the card's inputs."""
    from .tools_order import GENDERS
    out = dict(base or {})
    if 'name' in data:
        name = _clean(data.get('name'), 120)
        if not name:
            raise EditProblem("Patient name is needed.")
        out['name'] = name
    if 'age' in data:
        age = str(data.get('age') if data.get('age') is not None else '').strip()
        if age and not (age.isdigit() and 0 <= int(age) <= 120):
            raise EditProblem("Age must be a number from 0 to 120.")
        out['age'] = int(age) if age else None
    if 'gender' in data:
        out['gender'] = GENDERS.get(str(data.get('gender') or '').strip().lower())
    return out


def _bookable(p, test_id):
    from .tools_order import _test_rows
    with db.read() as conn:
        rows = _test_rows(conn, p.customer_id, "id = %s", (int(test_id),), 1)
    if not rows:
        raise EditProblem("That test is not bookable for this lab.", 409)
    return _option(rows[0])


def _patient(summary, index):
    patients = summary.get('patients') or []
    try:
        index = int(index)
    except (TypeError, ValueError):
        raise EditProblem("Patient not found on this card.")
    if not (0 <= index < len(patients)):
        raise EditProblem("Patient not found on this card.")
    return patients[index]


def _set_slot(p, payload, summary, date_value, slot_value):
    from .tools_order import _area_slot_full, _check_date, _norm_slot, _resolve_location, _slots_for
    day, problem = _check_date(p, date_value)
    if problem:
        raise EditProblem(problem)
    with db.read() as conn:
        slot = next((s for s in _slots_for(conn, p, day) if _norm_slot(s) == _norm_slot(slot_value)), None)
        if not slot:
            raise EditProblem(f"Slot {slot_value} is not available on {day.strftime('%d %b')}.", 409)
        loc, _ = _resolve_location(conn, p, payload.get('pincode'), payload.get('city'))
        if loc.get('city_id') and _area_slot_full(conn, p, loc, day, slot):
            raise EditProblem(f"Slot {slot} is full for this area on {day.strftime('%d %b')}.", 409)
    payload.update(date=day.isoformat(), slot=slot)
    summary.update(date=day.isoformat(), date_label=day.strftime('%a, %d %b %Y'), slot=slot)


def _sync(p, payload, summary):
    """Totals, the create-order patients list and warnings, from the card's patients."""
    patients = summary['patients']
    for pt in patients:
        pt['amount'] = round(sum(t['price'] for t in pt['tests']), 2)
    summary['total'] = round(sum(pt['amount'] for pt in patients), 2)
    phone = payload.get('contact_number') or ''
    payload['patients'] = [{
        'fullName': pt['name'],
        'age': '' if pt.get('age') is None else pt['age'],
        'gender': pt.get('gender') or '',
        'phone': phone,
        'amount': pt['amount'],
        'status': 'Pending',
        'test': [{'value': t['test_id'], 'text': t['name']} for t in pt['tests']],
    } for pt in patients]
    first = patients[0]['name'] if patients else ''
    payload['relationship'] = 'Self' if first.lower() == str(payload.get('contact_name') or '').lower() else 'Other'

    warnings = []
    with db.read() as conn:
        dup = db.fetch_one(conn, """
            SELECT order_number, slot_time FROM visits
            WHERE customer_id = %s AND patient_phone LIKE %s AND DATE(slot_date) = %s
              AND is_active = 1 AND deleted_at IS NULL AND (status IS NULL OR status != 'Cancelled')
            ORDER BY id DESC LIMIT 1
        """, (p.customer_id, '%' + phone[-10:], payload['date'])) if len(phone) >= 7 else None
    if dup:
        warnings.append(f"This phone number already has order {dup['order_number']} on this date ({dup['slot_time']}).")
    if any(pt.get('age') is None for pt in patients):
        warnings.append("Age is not given for every patient.")
    summary['warnings'] = warnings


def update_draft(p, draft_id, body):
    """Applies one edit from the card (usually one row's Save). Returns the new card."""
    from .tools_order import DRAFT_MINUTES, MAX_PATIENTS, MAX_TESTS_PER_PATIENT, card_from_row, get_draft
    row = _pending(p, draft_id)
    payload = json.loads(row['payload'] or '{}')
    summary = json.loads(row['summary'] or '{}')
    changed = False

    if 'payment_type' in body:
        payment = str(body.get('payment_type') or '').strip().capitalize()
        if payment not in PAYMENT_TYPES:
            raise EditProblem("Payment type must be Prepaid or Postpaid.")
        payload['payment_type'] = payment
        summary['payment_type'] = payment
        changed = True

    if body.get('address_ref'):
        _set_address(p, row, payload, summary, body['address_ref'])
        changed = True

    if body.get('date') or body.get('slot'):
        _set_slot(p, payload, summary, body.get('date') or summary.get('date'), body.get('slot') or summary.get('slot'))
        changed = True

    if 'contact_name' in body:
        name = _clean(body.get('contact_name'), 120)
        if not name:
            raise EditProblem("Contact name is needed.")
        payload['contact_name'] = name
        summary['contact_name'] = name
        changed = True

    if 'special_instruction' in body:
        note = _clean(body.get('special_instruction'), 500)
        payload['special_instruction'] = note
        summary['special_instruction'] = note or None
        changed = True

    if body.get('test'):
        t = body['test'] if isinstance(body['test'], dict) else {}
        try:
            patient, from_id, to_id = int(t.get('patient')), int(t.get('from_test_id')), int(t.get('to_test_id'))
        except (TypeError, ValueError):
            raise EditProblem("Say which test to change.")
        _set_test(p, payload, summary, patient, from_id, to_id)
        changed = True

    if isinstance(body.get('patient'), dict):
        pt = _patient(summary, body['patient'].get('index'))
        pt.update(_patient_fields(body['patient']))
        changed = True

    if isinstance(body.get('add_test'), dict):
        pt = _patient(summary, body['add_test'].get('patient'))
        new = _bookable(p, body['add_test'].get('test_id'))
        if any(t['test_id'] == new['test_id'] for t in pt['tests']):
            raise EditProblem(f"{pt['name']} already has {new['name']}.")
        if len(pt['tests']) >= MAX_TESTS_PER_PATIENT:
            raise EditProblem(f"One patient can have at most {MAX_TESTS_PER_PATIENT} tests.")
        pt['tests'].append({'test_id': new['test_id'], 'name': new['name'], 'price': new['price']})
        changed = True

    if isinstance(body.get('remove_test'), dict):
        pt = _patient(summary, body['remove_test'].get('patient'))
        try:
            test_id = int(body['remove_test'].get('test_id'))
        except (TypeError, ValueError):
            raise EditProblem("Say which test to remove.")
        if not any(t['test_id'] == test_id for t in pt['tests']):
            raise EditProblem("That test is not on this card any more.")
        if len(pt['tests']) == 1:
            raise EditProblem("A patient needs at least one test. Remove the patient instead.")
        pt['tests'] = [t for t in pt['tests'] if t['test_id'] != test_id]
        changed = True

    if isinstance(body.get('add_patient'), dict):
        data = body['add_patient']
        if len(summary['patients']) >= MAX_PATIENTS:
            raise EditProblem(f"One visit can have at most {MAX_PATIENTS} patients.")
        pt = _patient_fields({'name': data.get('name'), 'age': data.get('age'), 'gender': data.get('gender')})
        new = _bookable(p, data.get('test_id')) if data.get('test_id') else None
        if not new:
            raise EditProblem("Pick at least one test for the new patient.")
        pt['tests'] = [{'test_id': new['test_id'], 'name': new['name'], 'price': new['price']}]
        summary['patients'].append(pt)
        changed = True

    if 'remove_patient' in body:
        _patient(summary, body['remove_patient'])
        if len(summary['patients']) == 1:
            raise EditProblem("The order needs at least one patient.")
        summary['patients'].pop(int(body['remove_patient']))
        changed = True

    if not changed:
        raise EditProblem("Nothing to change.")
    _sync(p, payload, summary)

    # The user is still working on the card: give it a fresh 10 minutes.
    expires = (datetime.utcnow() + timedelta(minutes=DRAFT_MINUTES)).replace(microsecond=0)
    with db.write() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE copilot_order_drafts SET payload = %s, summary = %s, expires_at = %s, updated_at = NOW()
                WHERE id = %s AND customer_id = %s AND user_id = %s AND status = 'pending'
            """, (json.dumps(payload, ensure_ascii=False), json.dumps(summary, ensure_ascii=False),
                  expires.strftime('%Y-%m-%d %H:%M:%S'), draft_id, p.customer_id, p.user_id))
            if cur.rowcount != 1:
                raise EditProblem("This card can no longer be changed. Refresh the chat.", 409)
    return card_from_row(get_draft(p, draft_id))


# ------------------------------------------------------------------ lookups for the card's edit rows

def slot_choices(p, draft_id, date_value):
    """Every slot of the lab on a date, marked bookable / passed / full (for the draft's area)."""
    from .tools_order import _area_slot_full, _check_date, _resolve_location, _slots_for
    row = _pending(p, draft_id)
    payload = json.loads(row['payload'] or '{}')
    day, problem = _check_date(p, date_value)
    if problem:
        raise EditProblem(problem)
    with db.read() as conn:
        every = [str(r['slot']).strip() for r in db.fetch_all(conn, """
            SELECT slot FROM customer_slot WHERE customer_id = %s AND deleted_at IS NULL ORDER BY id
        """, (p.customer_id,)) if r['slot'] and str(r['slot']).strip()]
        open_now = set(_slots_for(conn, p, day))
        loc, _ = _resolve_location(conn, p, payload.get('pincode'), payload.get('city'))
        out = []
        for s in every:
            state = 'ok' if s in open_now else 'passed'
            if state == 'ok' and loc.get('city_id') and _area_slot_full(conn, p, loc, day, s):
                state = 'full'
            out.append({'slot': s, 'state': state})
    return {'date': day.isoformat(), 'date_label': day.strftime('%a, %d %b %Y'), 'slots': out}


def find_tests(p, query):
    """Test search for the card's "Add test" box: bookable tests, no AI."""
    from .tools_order import _test_rows
    query = ' '.join(str(query or '').split())[:60]
    if len(query) < 2:
        return []
    with db.read() as conn:
        rows = _test_rows(conn, p.customer_id, "test_name LIKE %s OR test_code LIKE %s",
                          (f'%{query}%', f'%{query}%'), 8)
    exact = query.lower()
    rows.sort(key=lambda r: (str(r['test_name']).strip().lower() != exact,
                             str(r['test_code'] or '').strip().lower() != exact))
    return [_option(r) for r in rows]
