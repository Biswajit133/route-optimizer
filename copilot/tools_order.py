"""Order-create tools (phase 2).

The model can look things up (search_tests, available_slots) and ask for a
draft (prepare_order). It can NOT create an order: prepare_order only checks
everything and saves a 'pending' row in copilot_order_drafts. The CRM shows
that row as a card, and the order is created only when the user clicks
Confirm (routes.confirm_draft -> orders_api.create_order with the user's JWT).

Checks here mirror the orders service create-order API (OrdersController
createOrder*), so most problems are caught before the card is shown. The API
still runs its own validation on confirm.
"""
import json
import re
from datetime import datetime, timedelta

from . import clock, db
from .draft_edit import address_choices, remember_search, test_options
from .tools_read import mask_phone, order_link

DRAFT_MINUTES = 10
BOOKING_DAYS_AHEAD = 15     # orders config services.orderBooking.accepted_future_booking
MAX_PATIENTS = 8            # createOrderRules patients max:8
MAX_TESTS_PER_PATIENT = 25
HOME_VISIT = 1              # visit_types.id for Home Visit

FILLER_WORDS = {'test', 'tests', 'package', 'packages', 'checkup', 'check', 'the', 'and', 'of', 'koro', 'karo'}

GENDERS = {'m': 'Male', 'male': 'Male', 'f': 'Female', 'female': 'Female', 'o': 'Other', 'other': 'Other'}


class OrderCreateNotAllowed(Exception):
    pass


# ------------------------------------------------------------------ access

def can_create_orders(p):
    """(ok, reason). Same rule as the CRM's Add Order button: the role's
    'orders-create' permission (role_has_permissions for this customer, else
    the customer_id 0 defaults - gateway AuthController)."""
    if p.customer.get('is_device_service'):
        return False, ("Orders for device-service accounts need the Add Order page "
                       "(case ID and service type). Patho Assist cannot create them yet.")
    with db.read() as conn:
        rows = db.fetch_all(conn, """
            SELECT r.customer_id, MAX(p.name = 'orders-create') AS allowed
            FROM role_has_permissions r
            JOIN permissions p ON p.id = r.permission_id AND p.deleted_at IS NULL
            WHERE r.role_id = %s AND r.customer_id IN (%s, 0)
            GROUP BY r.customer_id
        """, (p.role_id, p.customer_id))
    by_customer = {int(r['customer_id']): int(r['allowed'] or 0) for r in rows}
    allowed = by_customer[p.customer_id] if p.customer_id in by_customer else by_customer.get(0, 0)
    if not allowed:
        return False, "This user's role does not have permission to create orders."
    return True, None


def _country(p):
    with db.read() as conn:
        return db.fetch_one(conn, "SELECT name, symbol FROM countries WHERE id = %s",
                            (p.customer.get('country_id'),)) or {}


def currency_symbol(p):
    return _country(p).get('symbol') or '₹'


def normalize_phone(p, value):
    """Digits only. Indian labs store 10-digit mobiles, so drop a +91 / 0 prefix."""
    digits = re.sub(r'\D', '', str(value or ''))
    if str(_country(p).get('name') or '').strip().lower() == 'india':
        if len(digits) == 12 and digits.startswith('91'):
            digits = digits[2:]
        elif len(digits) == 11 and digits.startswith('0'):
            digits = digits[1:]
    return digits


# ------------------------------------------------------------------ tests

def _test_rows(conn, customer_id, where, params, limit):
    return db.fetch_all(conn, f"""
        SELECT id, test_name, test_code, test_price, discount_price, is_package, preparation
        FROM customer_dos
        WHERE customer_id = %s AND is_booking = 1 AND status = 'Active' AND deleted_at IS NULL
          AND ({where})
        ORDER BY is_popular DESC, test_name
        LIMIT {int(limit)}
    """, (customer_id, *params))


def search_tests(p, args, ctx=None):
    query = ' '.join(str(args.get('query') or '').split())
    if len(query) < 2:
        return {'error': 'Give at least 2 letters of the test name or code.'}
    words = [w for w in re.split(r'[\s,]+', query) if w and w.lower() not in FILLER_WORDS][:5]
    with db.read() as conn:
        rows = _test_rows(conn, p.customer_id, "test_name LIKE %s OR test_code LIKE %s",
                          (f'%{query}%', f'%{query}%'), 8)
        if not rows and words:
            # "lipid profile test" -> every word must appear somewhere in the name
            clause = ' AND '.join(['test_name LIKE %s'] * len(words))
            rows = _test_rows(conn, p.customer_id, clause, [f'%{w}%' for w in words], 8)
    exact = query.lower()
    rows.sort(key=lambda r: (str(r['test_name']).strip().lower() != exact,
                             str(r['test_code'] or '').strip().lower() != exact))
    # The user typed a test's exact code or name ("CBC"): that is the test they
    # mean, so return only it instead of a list for the model to ask about.
    exact_match = bool(rows) and exact in (str(rows[0]['test_name']).strip().lower(),
                                           str(rows[0]['test_code'] or '').strip().lower())
    remember_search(ctx, rows)  # the card offers the other matches as a switch
    if exact_match:
        rows = rows[:1]
    return {
        'query': query,
        'exact_match': exact_match,
        'matches': [{
            'test_id': int(r['id']),
            'name': str(r['test_name']).strip(),
            'code': r['test_code'] or None,
            'price': float(r['discount_price'] or 0),
            'mrp': float(r['test_price'] or 0),
            'package': bool(r['is_package']),
            'fasting': r['preparation'] in ('fasting', 'both'),
        } for r in rows],
        'note': ('Exact match for what the user typed: use it without asking.' if exact_match else
                 None if rows else 'No bookable test matched. Ask the user for another name or the test code.'),
    }


# ------------------------------------------------------------------ slots

def _minutes(text, which=0):
    found = re.findall(r'(\d{1,2}):(\d{2})\s*([AaPp][Mm])?', str(text or ''))
    if not found:
        return None
    hour, minute, ampm = found[which]
    hour, minute = int(hour), int(minute)
    if ampm:
        ampm = ampm.lower()
        if ampm == 'pm' and hour < 12:
            hour += 12
        if ampm == 'am' and hour == 12:
            hour = 0
    return hour * 60 + minute


def _norm_slot(text):
    return re.sub(r'\s+', '', str(text or '')).upper()


def _slots_for(conn, p, day):
    """Bookable slot strings for a day - get_slots_for_booking_New: the
    customer's customer_slot rows, minus started ones today when
    hide_past_slot is on (plus buffer_time when slot_buffer_feature is on)."""
    rows = db.fetch_all(conn, """
        SELECT slot FROM customer_slot WHERE customer_id = %s AND deleted_at IS NULL ORDER BY id
    """, (p.customer_id,))
    slots = [str(r['slot']).strip() for r in rows if r['slot'] and str(r['slot']).strip()]

    now = clock.now_local(p.tz_offset)
    if p.customer.get('hide_past_slot') and day == now.date():
        cutoff = now.hour * 60 + now.minute
        if p.customer.get('slot_buffer_feature'):
            buf = db.fetch_one(conn, "SELECT buffer_time FROM customer_slot_config WHERE customer_id = %s",
                               (p.customer_id,))
            cutoff += int((buf or {}).get('buffer_time') or 0)
        slots = [s for s in slots if (_minutes(s) is None or _minutes(s) > cutoff)]
    return slots


def _check_date(p, value):
    """(date, problem)."""
    today = clock.today_local(p.tz_offset)
    day = clock.resolve_date(value, p.tz_offset)
    if not day:
        return None, f"Date '{value}' is not clear. Use YYYY-MM-DD, today or tomorrow."
    if day < today:
        return None, f"{day.isoformat()} is in the past."
    if day > today + timedelta(days=BOOKING_DAYS_AHEAD):
        return None, f"Orders can only be booked up to {BOOKING_DAYS_AHEAD} days ahead."
    return day, None


def available_slots(p, args):
    day, problem = _check_date(p, args.get('date') or 'today')
    if problem:
        return {'error': problem}
    with db.read() as conn:
        slots = _slots_for(conn, p, day)
    return {
        'date': day.isoformat(),
        'weekday': day.strftime('%A'),
        'slots': slots,
        'note': None if slots else 'No slot left on this date. Suggest another date.',
    }


# ------------------------------------------------------------------ location

def _resolve_location(conn, p, pincode, city):
    """Same lookup as createOrderAddressParts: pincode first, then city name.
    Returns (loc dict, problem)."""
    pincode = re.sub(r'\s+', '', str(pincode or ''))
    city = ' '.join(str(city or '').split())
    loc = {'pincode': pincode, 'city': city, 'state': '', 'city_id': None, 'area_code_id': None, 'pincode_known': False}

    if pincode:
        row = db.fetch_one(conn, """
            SELECT pc.city_id, pc.area_code_id, c.city_name, s.state_name
            FROM pincodes pc
            LEFT JOIN cities c ON c.id = pc.city_id
            LEFT JOIN states s ON s.id = pc.state_id
            WHERE pc.customer_id = %s AND pc.pincode = %s AND pc.status = 'Active' AND pc.deleted_at IS NULL
            ORDER BY pc.id LIMIT 1
        """, (p.customer_id, pincode))
        if row:
            loc.update(city_id=int(row['city_id']), area_code_id=row['area_code_id'], pincode_known=True,
                       city=(row['city_name'] or city).strip(), state=(row['state_name'] or '').strip())

    if not loc['city_id'] and city:
        row = db.fetch_one(conn, """
            SELECT c.id, c.city_name, s.state_name FROM cities c
            LEFT JOIN states s ON s.id = c.state_id
            WHERE c.customer_id = %s AND TRIM(c.city_name) = %s AND c.deleted_at IS NULL
            ORDER BY c.id LIMIT 1
        """, (p.customer_id, city))
        if row:
            loc.update(city_id=int(row['id']), city=row['city_name'].strip(), state=(row['state_name'] or '').strip())

    if not loc['city_id']:
        if pincode:
            return loc, f"Pincode {pincode} is not in this lab's service area."
        if city:
            return loc, f"'{city}' is not one of this lab's service cities."
        return loc, "Pincode (or city) of the address is needed."

    if p.customer.get('restrict_order_by_user_city'):
        if not loc['pincode_known']:
            return loc, "A serviceable pincode is required for this lab."
        if p.city_ids and loc['city_id'] not in p.city_ids:
            return loc, "This user cannot create orders in that city."
    return loc, None


def _area_slot_full(conn, p, loc, day, slot):
    """assertCreateOrderAreaSlotAvailable."""
    if not loc.get('area_code_id') or not p.customer.get('area_slot_restriction'):
        return False
    cap = db.fetch_one(conn, "SELECT slots FROM slots WHERE area_code_id = %s AND deleted_at IS NULL LIMIT 1",
                       (loc['area_code_id'],))
    if not cap or cap['slots'] is None:
        return False
    booked = db.fetch_one(conn, """
        SELECT COUNT(*) AS n FROM visits
        WHERE customer_id = %s AND patient_areacode_id = %s AND is_active = 1
          AND (status IS NULL OR status != 'Cancelled')
          AND DATE(slot_date) = %s AND slot_time = %s
    """, (p.customer_id, loc['area_code_id'], day.isoformat(), slot))
    return int(booked['n']) >= int(cap['slots'])


# ------------------------------------------------------------------ prepare

def _clean(value, limit=200):
    return ' '.join(str(value or '').split())[:limit]


def prepare_order(p, args, ctx):
    ok, reason = can_create_orders(p)
    if not ok:
        return {'error': reason}
    if not ctx or not ctx.get('conversation_id'):
        return {'error': 'No conversation to attach the order card to.'}

    problems = []
    contact_name = _clean(args.get('contact_name'), 120)
    phone = normalize_phone(p, args.get('contact_phone'))
    street = _clean(args.get('address'), 250)
    instruction = _clean(args.get('special_instruction'), 500)
    payment = str(args.get('payment_type') or '').strip().capitalize()
    if payment and payment not in ('Prepaid', 'Postpaid'):
        payment = ''

    # A saved address from customer_history (address_ref "A1"): the full address
    # and its address_id come from copilot_customer_lookups, not from the model.
    saved = None
    pincode_in, city_in = args.get('pincode'), args.get('city')
    if str(args.get('address_ref') or '').strip():
        from .tools_customer import saved_address  # here, to avoid a circular import
        saved, looked_up_phone, problem = saved_address(p, ctx['conversation_id'], args.get('address_ref'))
        if problem:
            problems.append(problem)
        if saved:
            street, pincode_in, city_in = saved['street'][:250], saved.get('pincode'), saved.get('city')
            contact_name = contact_name or _clean(saved.get('contact_name'), 120)
        if not phone and looked_up_phone:
            phone = looked_up_phone

    if not contact_name:
        problems.append("Contact name is missing.")
    if not (7 <= len(phone) <= 15):
        problems.append("Contact phone number is missing or not valid.")
    if not street and not args.get('address_ref'):
        problems.append("Address (house / street / area) is missing.")

    patients_in = args.get('patients') if isinstance(args.get('patients'), list) else []
    if not patients_in:
        problems.append("At least one patient with tests is needed.")
    if len(patients_in) > MAX_PATIENTS:
        problems.append(f"One visit can have at most {MAX_PATIENTS} patients.")
        patients_in = patients_in[:MAX_PATIENTS]

    day, problem = _check_date(p, args.get('date'))
    if problem:
        problems.append(problem)

    with db.read() as conn:
        loc, problem = _resolve_location(conn, p, pincode_in, city_in)
        if problem:
            problems.append(problem)

        slot = None
        if day:
            slots = _slots_for(conn, p, day)
            wanted = _norm_slot(args.get('slot'))
            slot = next((s for s in slots if _norm_slot(s) == wanted), None)
            if not wanted:
                problems.append("Slot is missing.")
            elif not slot:
                problems.append(f"Slot '{args.get('slot')}' is not available on {day.isoformat()}. "
                                f"Call available_slots and offer one of those.")
            elif loc.get('city_id') and _area_slot_full(conn, p, loc, day, slot):
                problems.append(f"Slot {slot} is full for this area on {day.isoformat()}. Offer another slot.")

        all_ids = {int(t) for pt in patients_in if isinstance(pt, dict)
                   for t in (pt.get('test_ids') or []) if str(t).isdigit()}
        tests = {}
        if all_ids:
            marks, params = db.in_list(all_ids)
            tests = {int(r['id']): r for r in _test_rows(conn, p.customer_id, f"id IN ({marks})", params, 200)}

        patients = []
        for i, pt in enumerate(patients_in, 1):
            pt = pt if isinstance(pt, dict) else {}
            name = _clean(pt.get('name'), 120)
            if not name:
                problems.append(f"Patient {i}: name is missing.")
            age = pt.get('age')
            age = int(age) if str(age).strip().isdigit() and 0 <= int(age) <= 120 else None
            gender = GENDERS.get(str(pt.get('gender') or '').strip().lower())
            ids = [int(t) for t in (pt.get('test_ids') or []) if str(t).isdigit()][:MAX_TESTS_PER_PATIENT]
            ids = list(dict.fromkeys(ids))
            missing = [t for t in ids if t not in tests]
            if not ids:
                problems.append(f"Patient {i} ({name or 'unnamed'}): no test chosen. Use search_tests to get test_ids.")
            if missing:
                problems.append(f"Patient {i}: test id(s) {missing} are not bookable for this lab. Use search_tests.")
            rows = [tests[t] for t in ids if t in tests]
            patients.append({
                'name': name, 'age': age, 'gender': gender,
                'tests': [{'test_id': int(r['id']), 'name': str(r['test_name']).strip(),
                           'price': float(r['discount_price'] or 0)} for r in rows],
            })

        duplicate = None
        if day and len(phone) >= 7:
            duplicate = db.fetch_one(conn, """
                SELECT id, order_id, order_number, slot_time FROM visits
                WHERE customer_id = %s AND patient_phone LIKE %s AND DATE(slot_date) = %s
                  AND is_active = 1 AND deleted_at IS NULL AND (status IS NULL OR status != 'Cancelled')
                ORDER BY id DESC LIMIT 1
            """, (p.customer_id, '%' + phone[-10:], day.isoformat()))

        if not problems:
            test_options(p, conn, ctx, patients)

    if problems:
        return {'ok': False, 'problems': problems,
                'note': 'No card was shown. Fix these with the user (ask, or use the lookup tools), then call prepare_order again.'}

    for pt in patients:
        pt['amount'] = round(sum(t['price'] for t in pt['tests']), 2)
    total = round(sum(pt['amount'] for pt in patients), 2)

    warnings = []
    if duplicate:
        warnings.append(f"This phone number already has order {duplicate['order_number']} on this date "
                        f"({duplicate['slot_time']}).")
    if any(pt['age'] is None for pt in patients):
        warnings.append("Age is not given for every patient.")

    address_line = ', '.join(x for x in [street, loc['city'], loc['state'], loc['pincode']] if x)
    saved_id = int(saved['address_id']) if saved and saved.get('address_id') else 0
    first = patients[0]['name'] if patients else ''
    payload = {
        'type': HOME_VISIT,
        # A saved address goes by its id with no contact_address, so create-order
        # reuses that contact + address row (lat/lng included) untouched.
        'address_id': saved_id,
        'contact_address': '' if saved_id else address_line,
        'address': street,
        'city': loc['city'],
        'state': loc['state'],
        'pincode': loc['pincode'],
        'lat': '', 'lng': '',
        'access_notes': '',
        'special_instruction': instruction,
        'contact_name': contact_name,
        'relationship': 'Self' if first.lower() == contact_name.lower() else 'Other',
        'contact_number': phone,
        'date': day.isoformat(),
        'slot': slot,
        'phlebo_id': '',
        'patients': [{
            'fullName': pt['name'],
            'age': '' if pt['age'] is None else pt['age'],
            'gender': pt['gender'] or '',
            'phone': phone,
            'amount': pt['amount'],
            'status': 'Pending',
            'test': [{'value': t['test_id'], 'text': t['name']} for t in pt['tests']],
        } for pt in patients],
    }
    if payment:  # create-order: nullable|in:Prepaid,Postpaid - leave it out when not given
        payload['payment_type'] = payment
    summary = {
        'contact_name': contact_name,
        'contact_phone': phone,
        'address': address_line,
        'saved_address': bool(saved),
        'address_ref': saved['ref'] if saved else None,
        'address_choices': address_choices(p, ctx['conversation_id'], phone),
        'date': day.isoformat(),
        'date_label': day.strftime('%a, %d %b %Y'),
        'slot': slot,
        'visit_type': 'Home visit',
        'payment_type': payment or None,
        'special_instruction': instruction or None,
        'patients': patients,
        'total': total,
        'currency': currency_symbol(p),
        'warnings': warnings,
    }

    expires = (datetime.utcnow() + timedelta(minutes=DRAFT_MINUTES)).replace(microsecond=0)
    with db.write() as conn:
        db.execute(conn, """
            UPDATE copilot_order_drafts SET status = 'replaced', updated_at = NOW()
            WHERE conversation_id = %s AND customer_id = %s AND user_id = %s AND status = 'pending'
        """, (ctx['conversation_id'], p.customer_id, p.user_id))
        draft_id = db.execute(conn, """
            INSERT INTO copilot_order_drafts
                (customer_id, user_id, conversation_id, payload, summary, status, expires_at, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, 'pending', %s, NOW(), NOW())
        """, (p.customer_id, p.user_id, ctx['conversation_id'],
              json.dumps(payload, ensure_ascii=False), json.dumps(summary, ensure_ascii=False),
              expires.strftime('%Y-%m-%d %H:%M:%S')))
    ctx.setdefault('drafts', []).append(draft_id)

    # The model sees a masked phone; the full number goes only to the card.
    public = {k: v for k, v in summary.items() if k != 'address_choices'}
    public['contact_phone'] = mask_phone(phone)
    public['patients'] = [dict(pt, tests=[{k: v for k, v in t.items() if k != 'options'} for t in pt['tests']])
                          for pt in patients]
    note = ('The confirm card is now shown to the user below your reply. The order is NOT created yet: '
            'tell the user to check the card and click Confirm. Do not repeat every detail.')
    if not payment:
        note += ' Payment type is not set: tell them to pick Prepaid or Postpaid on the card (do not ask in chat).'
    if any(t.get('options') for pt in patients for t in pt['tests']):
        note += ' Other tests matching the name can be switched on the card.'
    return {'ok': True, 'card_shown': True, 'summary': public, 'note': note}


# ------------------------------------------------------------------ drafts

def _utc_now():
    return datetime.utcnow().replace(microsecond=0)


def card_from_row(row):
    """What the widget renders for a draft row."""
    if not row:
        return None
    summary = json.loads(row['summary'] or '{}')
    status = row['status']
    if status == 'pending' and row['expires_at'] and row['expires_at'] < _utc_now():
        status = 'expired'
    card = dict(summary)
    card.update({
        'draft_id': int(row['id']),
        'status': status,
        'error': row.get('error'),
        'order_number': row.get('order_number'),
        'link': order_link(row['order_id'], row['visit_id']) if row.get('order_id') and row.get('visit_id') else None,
        'expires_at': row['expires_at'].strftime('%Y-%m-%dT%H:%M:%SZ') if row.get('expires_at') else None,
    })
    return card


def get_draft(p, draft_id):
    with db.write() as conn:
        return db.fetch_one(conn, """
            SELECT * FROM copilot_order_drafts WHERE id = %s AND customer_id = %s AND user_id = %s
        """, (draft_id, p.customer_id, p.user_id))


def attach_message(draft_ids, message_id):
    if not draft_ids or not message_id:
        return
    marks, params = db.in_list(draft_ids)
    with db.write() as conn:
        db.execute(conn, f"UPDATE copilot_order_drafts SET message_id = %s WHERE id IN ({marks})",
                   (message_id, *params))


def cards_for_messages(message_ids):
    """{message_id: card} - the latest draft each message showed."""
    if not message_ids:
        return {}
    marks, params = db.in_list(message_ids)
    with db.write() as conn:
        rows = db.fetch_all(conn, f"""
            SELECT * FROM copilot_order_drafts WHERE message_id IN ({marks}) ORDER BY id
        """, params)
    return {int(r['message_id']): card_from_row(r) for r in rows}


def claim(p, draft_id):
    """pending -> creating, atomically, so a double click cannot create two orders.
    Returns (row, problem)."""
    row = get_draft(p, draft_id)
    if not row:
        return None, ("Order card not found.", 404)
    card = card_from_row(row)
    if card['status'] == 'expired':
        return row, ("This card has expired. Ask Patho Assist to prepare the order again.", 409)
    if row['status'] != 'pending':
        messages = {
            'creating': "This order is already being created.",
            'created': "This order was already created.",
            'cancelled': "This card was cancelled.",
            'replaced': "A newer card replaced this one. Use the latest card.",
            'failed': "This card already failed. Ask Patho Assist to prepare the order again.",
        }
        return row, (messages.get(row['status'], "This card can no longer be confirmed."), 409)
    if not json.loads(row['payload'] or '{}').get('payment_type'):
        return row, ("Pick the payment type (Prepaid or Postpaid) first.", 409)
    with db.write() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE copilot_order_drafts SET status = 'creating', confirmed_at = NOW(), updated_at = NOW()
                WHERE id = %s AND status = 'pending' AND expires_at >= UTC_TIMESTAMP()
            """, (draft_id,))
            if cur.rowcount != 1:
                return row, ("This card can no longer be confirmed. Refresh the chat.", 409)
    return row, None


def finish(draft_id, status, order=None, error=None):
    order = order or {}
    with db.write() as conn:
        db.execute(conn, """
            UPDATE copilot_order_drafts
            SET status = %s, order_id = %s, visit_id = %s, order_number = %s, error = %s, updated_at = NOW()
            WHERE id = %s
        """, (status, order.get('order_id'), order.get('id'), order.get('order_number'),
              (error or '')[:500] or None, draft_id))


def cancel(p, draft_id):
    with db.write() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE copilot_order_drafts SET status = 'cancelled', updated_at = NOW()
                WHERE id = %s AND customer_id = %s AND user_id = %s AND status = 'pending'
            """, (draft_id, p.customer_id, p.user_id))
            return cur.rowcount


# ------------------------------------------------------------------ registry

TOOLS = {
    'search_tests': {
        'handler': lambda p, args, ctx: search_tests(p, args, ctx),
        'description': "Find bookable tests/packages of this lab by name or code, with test_id and price. "
                       "Call it for every test the user names before prepare_order; if several match, ask which one.",
        'parameters': {'type': 'object', 'properties': {
            'query': {'type': 'string', 'description': 'Test name or code, e.g. "CBC", "lipid", "thyroid profile".'},
        }, 'required': ['query']},
    },
    'available_slots': {
        'handler': lambda p, args, ctx: available_slots(p, args),
        'description': "Bookable collection time slots for a date. Use the slot text exactly as returned.",
        'parameters': {'type': 'object', 'properties': {
            'date': {'type': 'string', 'description': "YYYY-MM-DD, 'today' or 'tomorrow'."},
        }, 'required': ['date']},
    },
    'prepare_order': {
        'handler': prepare_order,
        'description': "Check a new HOME VISIT order and show the user a confirm card. This does NOT create the order; "
                       "the user must click Confirm on the card. Call only when you have: contact name and phone, "
                       "address with pincode (or city) OR an address_ref from customer_history, date, an exact slot "
                       "from available_slots, and for every patient a name, age, gender and test_ids (from "
                       "search_tests or customer_history last_tests). If it returns problems, fix them with the user "
                       "and call it again.",
        'parameters': {'type': 'object', 'properties': {
            'contact_name': {'type': 'string', 'description': 'Person to contact at the address. With an '
                                                              'address_ref it defaults to the saved contact.'},
            'contact_phone': {'type': 'string', 'description': 'Contact phone number exactly as the user gave it.'},
            'address_ref': {'type': 'string', 'description': 'Saved address from customer_history, e.g. "A1". '
                                                             'When given, leave address, pincode and city out.'},
            'address': {'type': 'string', 'description': 'House / street / area / landmark (without city and pincode). '
                                                         'Only for a new address.'},
            'pincode': {'type': 'string'},
            'city': {'type': 'string'},
            'date': {'type': 'string', 'description': 'YYYY-MM-DD'},
            'slot': {'type': 'string', 'description': 'Exact slot text from available_slots.'},
            'payment_type': {'type': 'string', 'enum': ['Prepaid', 'Postpaid'],
                             'description': 'Only if the user said it.'},
            'special_instruction': {'type': 'string', 'description': 'Optional note for the lab / phlebo.'},
            'patients': {'type': 'array', 'minItems': 1, 'maxItems': MAX_PATIENTS, 'items': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'age': {'type': 'integer'},
                    'gender': {'type': 'string', 'enum': ['Male', 'Female', 'Other']},
                    'test_ids': {'type': 'array', 'items': {'type': 'integer'}},
                },
                'required': ['name', 'test_ids'],
            }},
        }, 'required': ['contact_phone', 'date', 'slot', 'patients']},
    },
}
