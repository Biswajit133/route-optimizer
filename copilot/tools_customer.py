"""Returning-customer lookup for booking (phase 2b).

customer_history(phone) reads this lab's past orders for a phone number (only
the orders this user could open in the CRM - scope.visit_scope) and returns:

- saved addresses (latest 3 distinct), as refs A1, A2, A3
- family members with today's age (stored age + years since that order, or dob)
- each member's last tests, mapped to today's bookable test and price
- orders already booked for today or later (to avoid a double booking)

The widget shows the full result as the "Returning customer" card. The model
gets the same thing with every address cut down to "city pincode", and books a
saved address by its ref: prepare_order looks the ref up in
copilot_customer_lookups, so the full address and its address_id go from the
database straight into the order card without passing through the model.
"""
import json
import logging
from datetime import date, datetime

from . import clock, db
from .scope import visit_scope
from .tools_order import GENDERS, _test_rows, can_create_orders, currency_symbol, normalize_phone
from .tools_read import mask_phone, order_link

log = logging.getLogger('copilot.customer')

LOOKBACK_VISITS = 30
MAX_ADDRESSES = 3
MAX_MEMBERS = 6
CLOSED_STATUSES = ('Cancelled', 'Completed')


def _as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], '%Y-%m-%d').date()
    except (TypeError, ValueError):
        return None


def _clean(value):
    return ' '.join(str(value or '').split())


def _age_now(age, dob, seen_on, today):
    """(age today, age stored on the order)."""
    born = _as_date(dob)
    if born and born.year > 1900:
        years = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        return (years if 0 <= years <= 120 else None), None
    try:
        was = int(str(age).strip())
    except (TypeError, ValueError):
        return None, None
    if not 0 <= was <= 120:
        return None, None
    seen_on = _as_date(seen_on)
    grown = (today - seen_on).days // 365 if seen_on and seen_on < today else 0
    return min(was + grown, 120), was


def _gender(value):
    text = _clean(value).lower()
    return GENDERS.get(text) or (text.capitalize() if text else None)


def _label(day):
    return day.strftime('%d %b %Y') if day else None


def _visits(conn, p, phone):
    where, params = visit_scope(p, 'v')
    return db.fetch_all(conn, f"""
        SELECT v.id, v.order_id, v.order_number, v.slot_date, v.slot_time, v.status,
               v.patient_address_id, v.patient_address, v.patient_landmark, v.patient_pincode,
               v.main_patient_name, c.city_name
        FROM visits v
        LEFT JOIN cities c ON c.id = COALESCE(NULLIF(v.patient_city_id, 0), v.city_id)
        WHERE {where} AND v.patient_phone LIKE %s
        ORDER BY v.id DESC
        LIMIT {LOOKBACK_VISITS}
    """, (*params, '%' + phone[-10:]))


def _addresses(conn, p, visits):
    """Latest distinct addresses. A live enduser_addresses row is booked by its id
    (create-order reuses that contact and address, lat/lng included); a visit
    without one falls back to its address text as a new address."""
    ids = {int(v['patient_address_id']) for v in visits if str(v['patient_address_id'] or '').isdigit()
           and int(v['patient_address_id']) > 0}
    live = {}
    if ids:
        marks, params = db.in_list(ids)
        for r in db.fetch_all(conn, f"""
            SELECT a.id, a.address, a.landmark, a.city, a.pincode, e.first_name, e.last_name
            FROM enduser_addresses a
            LEFT JOIN endusers e ON e.id = a.enduser_id
            WHERE a.customer_id = %s AND a.deleted_at IS NULL AND a.id IN ({marks})
        """, (p.customer_id, *params)):
            live[int(r['id'])] = r

    found = {}
    for v in visits:
        row = live.get(int(v['patient_address_id'] or 0))
        if row:
            street, pincode, city = _clean(row['address']), _clean(row['pincode']), _clean(row['city'] or v['city_name'])
            landmark = _clean(row['landmark'])
            contact = _clean(f"{row['first_name'] or ''} {row['last_name'] or ''}")
            address_id = int(row['id'])
        else:
            street, pincode, city = _clean(v['patient_address']), _clean(v['patient_pincode']), _clean(v['city_name'])
            landmark, contact, address_id = _clean(v['patient_landmark']), '', None
        if not street:
            continue
        key = (street.lower(), pincode)
        entry = found.get(key)
        if entry:
            entry['times'] += 1
            if not entry['address_id'] and address_id:
                entry['address_id'] = address_id
            continue
        found[key] = {'address_id': address_id, 'street': street, 'landmark': landmark, 'city': city,
                      'pincode': pincode, 'contact_name': contact or _clean(v['main_patient_name']),
                      'times': 1, 'last_date': _as_date(v['slot_date'])}

    out = list(found.values())[:MAX_ADDRESSES]   # visits are newest first, so dict order is too
    for i, a in enumerate(out, 1):
        a['ref'] = f'A{i}'
    return out


def _members(conn, p, visits, today):
    by_visit = {int(v['id']): v for v in visits}
    marks, params = db.in_list(by_visit)
    rows = db.fetch_all(conn, f"""
        SELECT id, visit_id, first_name, last_name, gender, age, dob
        FROM visit_patients
        WHERE customer_id = %s AND deleted_at IS NULL AND visit_id IN ({marks})
        ORDER BY visit_id DESC, id
    """, (p.customer_id, *params))

    members = {}
    for r in rows:
        name = _clean(f"{r['first_name'] or ''} {r['last_name'] or ''}")
        if not name or name.lower() in members:
            continue
        seen_on = _as_date(by_visit[int(r['visit_id'])]['slot_date'])
        age, was = _age_now(r['age'], r['dob'], seen_on, today)
        members[name.lower()] = {'visit_patient_id': int(r['id']), 'name': name, 'gender': _gender(r['gender']),
                                 'age': age, 'age_was': was if was is not None and was != age else None,
                                 'last_date': seen_on}
        if len(members) >= MAX_MEMBERS:
            break
    members = list(members.values())
    if not members:
        return members

    marks, params = db.in_list([m['visit_patient_id'] for m in members])
    tests = db.fetch_all(conn, f"""
        SELECT visit_patient_id, test_id, test_name, final_price, test_price
        FROM visit_patients_tests
        WHERE customer_id = %s AND deleted_at IS NULL AND visit_patient_id IN ({marks})
        ORDER BY id
    """, (p.customer_id, *params))
    ids = {int(t['test_id']) for t in tests if str(t['test_id'] or '').isdigit()}
    bookable = {}
    if ids:
        marks, params = db.in_list(ids)
        bookable = {int(r['id']): r for r in _test_rows(conn, p.customer_id, f"id IN ({marks})", params, 200)}

    for m in members:
        m['tests'], m['unavailable'] = [], []
        seen = set()
        for t in tests:
            if int(t['visit_patient_id']) != m['visit_patient_id']:
                continue
            tid = int(t['test_id']) if str(t['test_id'] or '').isdigit() else None
            if tid in seen:
                continue
            seen.add(tid)
            row = bookable.get(tid)
            if not row:
                m['unavailable'].append(_clean(t['test_name']) or 'Unknown test')
                continue
            price = float(row['discount_price'] or 0)
            paid = t['final_price'] if t['final_price'] is not None else t['test_price']
            paid = float(paid) if paid is not None else None
            m['tests'].append({'test_id': tid, 'name': _clean(row['test_name']), 'price': price,
                               'old_price': paid if paid is not None and abs(paid - price) >= 0.01 else None})
    return members


def customer_history(p, args, ctx):
    ok, reason = can_create_orders(p)
    if not ok:
        return {'error': reason}
    if not ctx or not ctx.get('conversation_id'):
        return {'error': 'No conversation to attach the card to.'}
    phone = normalize_phone(p, args.get('phone'))
    if not 7 <= len(phone) <= 15:
        return {'error': 'Give the full phone number to look up.'}

    today = clock.today_local(p.tz_offset)
    with db.read() as conn:
        visits = _visits(conn, p, phone)
        if not visits:
            return {'found': False, 'note': 'No past orders for this number in this lab. Treat it as a new customer '
                                            'and ask for the details that are missing.'}
        addresses = _addresses(conn, p, visits)
        members = _members(conn, p, visits, today)

    open_orders = []
    for v in visits:
        day = _as_date(v['slot_date'])
        if day and day >= today and (v['status'] or '') not in CLOSED_STATUSES:
            open_orders.append({'order_number': v['order_number'], 'date': day.isoformat(), 'date_label': _label(day),
                                'slot': v['slot_time'], 'status': v['status'],
                                'link': order_link(v['order_id'], v['id'])})
    open_orders = open_orders[:3]
    last_date = _as_date(visits[0]['slot_date'])
    contact_name = (addresses[0]['contact_name'] if addresses else '') or _clean(visits[0]['main_patient_name'])
    many = len(visits) >= LOOKBACK_VISITS

    card = {
        'contact_name': contact_name,
        'phone': mask_phone(phone),
        'orders': len(visits),
        'orders_more': many,
        'last_date_label': _label(last_date),
        'currency': currency_symbol(p),
        'addresses': [{'ref': a['ref'], 'number': i,
                       'line': ', '.join(x for x in [a['street'], a['landmark']] if x),
                       'place': ' '.join(x for x in [a['city'], a['pincode']] if x),
                       'times': a['times'], 'last_date_label': _label(a['last_date'])}
                      for i, a in enumerate(addresses, 1)],
        'patients': [{'name': m['name'], 'gender': m['gender'], 'age': m['age'], 'age_was': m['age_was'],
                      'tests': [{'name': t['name'], 'price': t['price'], 'old_price': t['old_price']} for t in m['tests']],
                      'unavailable': m['unavailable']} for m in members],
        'open_orders': open_orders,
    }
    model = {
        'found': True,
        'orders_found': f"{len(visits)}+" if many else len(visits),
        'last_order_date': last_date.isoformat() if last_date else None,
        'contact_name': contact_name,
        'saved_addresses': [{'address_ref': a['ref'], 'number_on_card': i,
                             'area': ' '.join(x for x in [a['city'], a['pincode']] if x) or 'saved address',
                             'times_used': a['times']} for i, a in enumerate(addresses, 1)],
        'patients': [{'name': m['name'], 'gender': m['gender'], 'age_today': m['age'],
                      'last_tests': [{'test_id': t['test_id'], 'name': t['name'], 'price': t['price']} for t in m['tests']],
                      'no_longer_bookable': m['unavailable']} for m in members],
        'already_booked': [{'order_number': o['order_number'], 'date': o['date'], 'slot': o['slot'],
                            'link': o['link']} for o in open_orders],
    }
    data = {'card': card, 'model': model, 'phone': phone,
            'addresses': [{k: (v.isoformat() if isinstance(v, date) else v) for k, v in a.items()} for a in addresses]}

    with db.write() as conn:
        lookup_id = db.execute(conn, """
            INSERT INTO copilot_customer_lookups (customer_id, user_id, conversation_id, phone, data, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
        """, (p.customer_id, p.user_id, ctx['conversation_id'], phone, json.dumps(data, ensure_ascii=False)))
    ctx.setdefault('lookups', []).append(lookup_id)

    return dict(model, note=(
        "A 'Returning customer' card with these addresses, patients and last tests is now shown to the user. Do not "
        "list them all again. Book a saved address with address_ref (never retype it). For 'same tests' use the "
        "last_tests test_ids. Ask only for what is still missing; if nothing is, call prepare_order now."))


# ------------------------------------------------------------------ lookups by conversation / message

def saved_address(p, conversation_id, ref):
    """The saved address `ref` (A1...) from this conversation's latest lookup,
    re-read from enduser_addresses when it has an id. Returns (address, phone, problem)."""
    with db.write() as conn:
        row = db.fetch_one(conn, """
            SELECT phone, data FROM copilot_customer_lookups
            WHERE conversation_id = %s AND customer_id = %s AND user_id = %s
            ORDER BY id DESC LIMIT 1
        """, (conversation_id, p.customer_id, p.user_id))
    if not row:
        return None, None, "No saved addresses were looked up in this chat. Call customer_history with the phone first."
    data = json.loads(row['data'] or '{}')
    ref = str(ref or '').strip().upper()
    entry = next((a for a in data.get('addresses') or [] if a.get('ref') == ref), None)
    if not entry:
        return None, row['phone'], f"Saved address {ref} is not on the latest card. Use one of its address_refs."
    entry = dict(entry)
    if entry.get('address_id'):
        with db.read() as conn:
            live = db.fetch_one(conn, """
                SELECT id, address, landmark, city, pincode FROM enduser_addresses
                WHERE id = %s AND customer_id = %s AND deleted_at IS NULL
            """, (entry['address_id'], p.customer_id))
        if not live:
            return None, row['phone'], f"Saved address {ref} was removed. Ask the user for the address."
        entry.update(street=_clean(live['address']), landmark=_clean(live['landmark']),
                     city=_clean(live['city']) or entry.get('city'), pincode=_clean(live['pincode']))
    return entry, row['phone'], None


def attach_message(lookup_ids, message_id):
    if not lookup_ids or not message_id:
        return
    marks, params = db.in_list(lookup_ids)
    with db.write() as conn:
        db.execute(conn, f"UPDATE copilot_customer_lookups SET message_id = %s WHERE id IN ({marks})",
                   (message_id, *params))


def _by_message(message_ids):
    if not message_ids:
        return {}
    marks, params = db.in_list(message_ids)
    with db.write() as conn:
        rows = db.fetch_all(conn, f"""
            SELECT message_id, data FROM copilot_customer_lookups WHERE message_id IN ({marks}) ORDER BY id
        """, params)
    return {int(r['message_id']): json.loads(r['data'] or '{}') for r in rows}


def get_card(lookup_id):
    with db.write() as conn:
        row = db.fetch_one(conn, "SELECT data FROM copilot_customer_lookups WHERE id = %s", (lookup_id,))
    return json.loads(row['data']).get('card') if row else None


def cards_for_messages(message_ids):
    """{message_id: returning-customer card} for the widget."""
    return {mid: d.get('card') for mid, d in _by_message(message_ids).items()}


def notes_for_messages(message_ids):
    """{message_id: text} so the model still knows the refs and test_ids on later turns."""
    return {mid: '[Returning customer card shown: ' + json.dumps(d.get('model') or {}, ensure_ascii=False) + ']'
            for mid, d in _by_message(message_ids).items()}


TOOLS = {
    'customer_history': {
        'handler': customer_history,
        'description': "Look up this lab's past orders for a phone number while booking: saved addresses (address_ref), "
                       "family members with today's age, their last tests (test_ids, today's price) and orders already "
                       "booked. Call it as soon as the user gives the contact phone for a new order. It shows the user "
                       "a 'Returning customer' card.",
        'parameters': {'type': 'object', 'properties': {
            'phone': {'type': 'string', 'description': 'Phone number exactly as the user gave it.'},
        }, 'required': ['phone']},
    },
}
