"""Read-only tools the model may call.

Rules every tool follows:
- the visit scope (customer + user restrictions) comes from scope.visit_scope,
  never from the model's arguments;
- parameterized SQL only - the model never writes SQL;
- patient phone numbers are masked before they leave this service (the reply
  goes through a third-party LLM);
- results are small: counts, plus at most a handful of example rows.
"""
import re
from collections import OrderedDict

from . import clock, db, help_docs
from .scope import visit_scope

# Visit statuses seen in visits.status, grouped the way a lab desk talks.
STATUS_GROUPS = OrderedDict([
    ('pending', ['Accepted', 'Confirmed', 'Rescheduled']),
    ('assigned', ['Phlebo Assigned']),
    ('on_the_way', ['Phlebo Enroute', 'Phlebo Reached']),
    ('collected', ['Sample Collected', 'Collected', 'Enroute to Lab', 'Enroute To Runner',
                   'Sample handover to Runner', 'Dropped in Courier', 'Enroute to Ship', 'Dropped In Lab']),
    ('completed', ['Completed']),
    ('cancelled', ['Cancelled']),
])
_GROUP_OF = {s.lower(): g for g, statuses in STATUS_GROUPS.items() for s in statuses}
OPEN_GROUPS = ('pending', 'assigned', 'on_the_way')

MAX_ROWS = 20

PATIENT_NAME_SQL = """COALESCE(NULLIF(TRIM(v.main_patient_name), ''), (
    SELECT TRIM(CONCAT_WS(' ', vp.first_name, vp.last_name)) FROM visit_patients vp
    WHERE vp.visit_id = v.id AND vp.deleted_at IS NULL
    ORDER BY vp.is_primary = 'Yes' DESC, vp.id LIMIT 1))"""

ROW_SELECT = f"""
    SELECT v.id AS visit_id, v.order_id, v.order_number, v.status, v.slot_date, v.slot_time,
           {PATIENT_NAME_SQL} AS patient_name, v.patient_phone,
           v.phlebo_id, TRIM(CONCAT_WS(' ', ph.first_name, ph.last_name)) AS phlebo_name,
           v.payment_status, c.city_name AS city
    FROM visits v
    LEFT JOIN phlebos ph ON ph.id = v.phlebo_id
    LEFT JOIN cities c ON c.id = v.city_id
"""


def group_of(status):
    return _GROUP_OF.get(str(status or '').strip().lower(), 'other')


def mask_phone(phone):
    digits = re.sub(r'\D', '', str(phone or ''))
    if len(digits) < 4:
        return None
    return '•' * (len(digits) - 3) + digits[-3:]


def order_link(order_id, visit_id):
    return f"/orders/view-order/{order_id}/{visit_id}"


def _row(r):
    return {
        'order_number': r['order_number'],
        'status': r['status'],
        'status_group': group_of(r['status']),
        'slot_date': clock.iso(r['slot_date']),
        'slot_time': r['slot_time'],
        'patient_name': r['patient_name'],
        'patient_phone': mask_phone(r['patient_phone']),
        'phlebo': r['phlebo_name'] or None,
        'city': r['city'],
        'payment_status': r['payment_status'],
        'link': order_link(r['order_id'], r['visit_id']),
    }


def _date_filter(p, args, alias='v'):
    """(sql, params, info) for the date window the model asked for."""
    start, end = clock.date_range(args.get('date_from'), args.get('date_to'), p.tz_offset)
    basis = args.get('date_basis') if args.get('date_basis') in ('slot', 'created') else 'slot'
    if basis == 'created':
        lo, hi = clock.created_at_bounds(start, end, p.tz_offset)
        sql, params = f"{alias}.created_at >= %s AND {alias}.created_at < %s", [lo, hi]
    else:
        sql, params = f"{alias}.slot_date BETWEEN %s AND %s", [start.isoformat(), end.isoformat()]
    info = {'from': start.isoformat(), 'to': end.isoformat(),
            'basis': 'booked (created) date' if basis == 'created' else 'scheduled (slot) date'}
    return sql, params, info


def _status_filter(group):
    statuses = STATUS_GROUPS.get(group)
    if not statuses:
        return '', []
    marks, values = db.in_list(statuses)
    return f" AND v.status IN ({marks})", values


# --------------------------------------------------------------------------- tools

def order_summary(p, args):
    scope_sql, scope_params = visit_scope(p)
    date_sql, date_params, info = _date_filter(p, args)

    with db.read() as conn:
        rows = db.fetch_all(conn, f"""
            SELECT v.status, COUNT(*) AS n,
                   SUM(CASE WHEN v.phlebo_id IS NULL OR v.phlebo_id = 0 THEN 1 ELSE 0 END) AS unassigned,
                   SUM(CASE WHEN v.payment_status = 'pending' THEN 1 ELSE 0 END) AS payment_pending
            FROM visits v
            WHERE {scope_sql} AND {date_sql}
            GROUP BY v.status
        """, scope_params + date_params)

    groups = OrderedDict((g, 0) for g in list(STATUS_GROUPS) + ['other'])
    by_status, total, payment_pending, unassigned_open = {}, 0, 0, 0
    for r in rows:
        n = int(r['n'])
        g = group_of(r['status'])
        groups[g] += n
        by_status[r['status'] or 'Unknown'] = n
        total += n
        if g != 'cancelled':
            payment_pending += int(r['payment_pending'] or 0)
        if g in ('pending',):
            unassigned_open += int(r['unassigned'] or 0)

    return {
        'date_range': info,
        'total_orders': total,
        'by_group': groups,
        'by_status': by_status,
        'pending_without_phlebo': unassigned_open,
        'payment_pending_not_cancelled': payment_pending,
        'group_meaning': {
            'pending': 'booked, visit not started yet (Accepted / Rescheduled); may or may not have a phlebo - see pending_without_phlebo',
            'assigned': 'phlebo assigned, not started',
            'on_the_way': 'phlebo enroute or reached the patient',
            'collected': 'sample collected / on the way to or dropped in lab',
            'completed': 'Completed',
            'cancelled': 'Cancelled',
        },
        'orders_page': '/orders',
    }


def get_order(p, args):
    ref = str(args.get('order_ref') or '').strip()
    if not ref:
        return {'error': 'order_ref is required'}

    scope_sql, scope_params = visit_scope(p)
    match = ["v.order_number = %s", "v.order_ref_number = %s"]
    params = [ref, ref]
    if ref.isdigit():
        match += ["v.order_number LIKE %s", "v.id = %s", "v.order_id = %s"]
        params += ['%-' + ref, int(ref), int(ref)]

    with db.read() as conn:
        found = db.fetch_all(conn, ROW_SELECT + f"""
            WHERE {scope_sql} AND ({' OR '.join(match)})
            ORDER BY v.id DESC LIMIT 3
        """, scope_params + params)
        if not found:
            return {'found': False, 'message': f'No order matching "{ref}" that this user can see.'}

        results = []
        for r in found:
            vid = r['visit_id']
            visit = db.fetch_one(conn, """
                SELECT v.total_order_value, v.amount_to_collect, v.final_amount_to_collect, v.payment_mode,
                       v.cancellation_reason, v.reschedul_reason, v.is_delayed, v.delay_reason,
                       v.facility_name, v.special_instruction, v.created_at, v.patient_address,
                       l.lab_name
                FROM visits v LEFT JOIN labs l ON l.id = v.assign_lab_id
                WHERE v.id = %s
            """, (vid,))
            patients = db.fetch_all(conn, """
                SELECT vp.id, TRIM(CONCAT_WS(' ', vp.first_name, vp.last_name)) AS name, vp.age, vp.gender
                FROM visit_patients vp WHERE vp.visit_id = %s AND vp.deleted_at IS NULL ORDER BY vp.id
            """, (vid,))
            tests = db.fetch_all(conn, """
                SELECT visit_patient_id, test_name, test_code FROM visit_patients_tests
                WHERE visit_id = %s AND deleted_at IS NULL ORDER BY id
            """, (vid,))
            timeline = db.fetch_all(conn, """
                SELECT status, created_at FROM visit_status
                WHERE visit_id = %s AND deleted_at IS NULL ORDER BY id
            """, (vid,))

            item = _row(r)
            item.update({
                'booked_at': clock.iso(visit['created_at']),
                'order_value': visit['total_order_value'],
                'amount_to_collect': visit['final_amount_to_collect'] if visit['final_amount_to_collect'] is not None else visit['amount_to_collect'],
                'payment_mode': visit['payment_mode'],
                'lab': visit['lab_name'],
                'facility': visit['facility_name'],
                'area': (visit['patient_address'] or '')[:80] or None,
                'cancellation_reason': visit['cancellation_reason'] or None,
                'reschedule_reason': visit['reschedul_reason'] or None,
                'delayed': bool(visit['is_delayed']),
                'delay_reason': visit['delay_reason'] or None,
                'special_instruction': (visit['special_instruction'] or '')[:200] or None,
                'patients': [{
                    'name': pt['name'], 'age': pt['age'], 'gender': pt['gender'],
                    'tests': [t['test_name'] for t in tests if t['visit_patient_id'] == pt['id']][:15],
                } for pt in patients],
                'timeline': [{'status': t['status'], 'at': clock.iso(t['created_at'])} for t in timeline][-12:],
            })
            results.append(item)

    return {'found': True, 'matches': results}


def find_orders(p, args):
    query = str(args.get('query') or '').strip()
    group = args.get('status_group')
    phlebo = str(args.get('phlebo_name') or '').strip()
    limit = max(1, min(int(args.get('limit') or 10), MAX_ROWS))

    scope_sql, scope_params = visit_scope(p)
    where, params = [scope_sql], list(scope_params)
    info = None

    # A search by name/phone/number looks across all dates unless a date was given.
    if not query or args.get('date_from') or args.get('date_to'):
        date_sql, date_params, info = _date_filter(p, args)
        where.append(date_sql)
        params += date_params

    if query:
        like = '%' + query + '%'
        digits = re.sub(r'\D', '', query)
        parts = ["v.order_number LIKE %s", f"{PATIENT_NAME_SQL} LIKE %s",
                 "EXISTS (SELECT 1 FROM visit_patients vp2 WHERE vp2.visit_id = v.id AND vp2.deleted_at IS NULL "
                 "AND CONCAT_WS(' ', vp2.first_name, vp2.last_name) LIKE %s)"]
        params += [like, like, like]
        if len(digits) >= 4:
            dlike = '%' + digits + '%'
            parts += ["v.patient_phone LIKE %s", "v.altPhone LIKE %s",
                      "EXISTS (SELECT 1 FROM visit_patients vp3 WHERE vp3.visit_id = v.id AND vp3.deleted_at IS NULL "
                      "AND vp3.phone LIKE %s)"]
            params += [dlike, dlike, dlike]
        where.append('(' + ' OR '.join(parts) + ')')

    status_sql, status_params = _status_filter(group)
    if phlebo:
        where.append("CONCAT_WS(' ', ph.first_name, ph.last_name) LIKE %s")
        params.append('%' + phlebo + '%')

    base_where = ' AND '.join(where) + status_sql
    all_params = params + status_params

    with db.read() as conn:
        count = db.fetch_one(conn, f"""
            SELECT COUNT(*) AS n FROM visits v LEFT JOIN phlebos ph ON ph.id = v.phlebo_id
            WHERE {base_where}
        """, all_params)['n']
        rows = db.fetch_all(conn, ROW_SELECT + f"""
            WHERE {base_where}
            ORDER BY v.slot_date DESC, v.id DESC LIMIT {limit}
        """, all_params)

    return {
        'date_range': info,
        'filters': {'query': query or None, 'status_group': group if group in STATUS_GROUPS else None,
                    'phlebo_name': phlebo or None},
        'total_matching': int(count),
        'showing': len(rows),
        'orders': [_row(r) for r in rows],
        'orders_page': '/orders',
    }


def attention_needed(p, args):
    day = clock.resolve_date(args.get('date'), p.tz_offset, clock.today_local(p.tz_offset))
    scope_sql, scope_params = visit_scope(p)
    is_today = day == clock.today_local(p.tz_offset)
    now = clock.now_local(p.tz_offset)
    now_minutes = now.hour * 60 + now.minute

    with db.read() as conn:
        rows = db.fetch_all(conn, ROW_SELECT.replace(
            "v.payment_status, c.city_name AS city",
            "v.payment_status, c.city_name AS city, v.is_delayed, v.delay_reason"
        ) + f"""
            WHERE {scope_sql} AND v.slot_date = %s
            ORDER BY v.slot_time, v.id
        """, scope_params + [day.isoformat()])

    buckets = OrderedDict([
        ('unassigned', {'meaning': 'booked but no phlebo assigned yet', 'orders': []}),
        ('slot_passed_not_collected', {'meaning': 'slot time is over but sample not collected yet', 'orders': []}),
        ('delayed', {'meaning': 'flagged delayed by the delay alert', 'orders': []}),
        ('payment_pending', {'meaning': 'sample collected/completed but payment still pending', 'orders': []}),
    ])
    for r in rows:
        g = group_of(r['status'])
        if g == 'pending' and not r['phlebo_id']:
            buckets['unassigned']['orders'].append(r)
        if is_today and g in OPEN_GROUPS:
            end = clock.slot_end_minutes(r['slot_time'])
            if end is not None and end < now_minutes:
                buckets['slot_passed_not_collected']['orders'].append(r)
        if r.get('is_delayed') and g in OPEN_GROUPS:
            buckets['delayed']['orders'].append(r)
        if g in ('collected', 'completed') and str(r['payment_status'] or '').lower() == 'pending':
            buckets['payment_pending']['orders'].append(r)

    return {
        'date': day.isoformat(),
        'checked_at_local_time': now.strftime('%H:%M') if is_today else None,
        'total_orders_that_day': len(rows),
        'issues': {
            key: {'meaning': b['meaning'], 'count': len(b['orders']),
                  'examples': [_row(r) for r in b['orders'][:8]]}
            for key, b in buckets.items()
        },
    }


def phlebo_workload(p, args):
    day = clock.resolve_date(args.get('date'), p.tz_offset, clock.today_local(p.tz_offset))
    scope_sql, scope_params = visit_scope(p)
    with db.read() as conn:
        rows = db.fetch_all(conn, f"""
            SELECT v.phlebo_id, TRIM(CONCAT_WS(' ', ph.first_name, ph.last_name)) AS name, v.status, COUNT(*) AS n
            FROM visits v LEFT JOIN phlebos ph ON ph.id = v.phlebo_id
            WHERE {scope_sql} AND v.slot_date = %s
            GROUP BY v.phlebo_id, name, v.status
        """, scope_params + [day.isoformat()])

    people = OrderedDict()
    for r in rows:
        key = r['phlebo_id'] or 0
        person = people.setdefault(key, {
            'phlebo': r['name'] if key else 'Not assigned', 'total': 0,
            **{g: 0 for g in list(STATUS_GROUPS) + ['other']},
        })
        n = int(r['n'])
        person['total'] += n
        person[group_of(r['status'])] += n

    ranked = sorted(people.values(), key=lambda x: -x['total'])
    return {'date': day.isoformat(), 'phlebos': ranked[:40], 'phlebo_count': len([k for k in people if k])}


def help_search(p, args):
    return help_docs.search(str(args.get('query') or ''))


# --------------------------------------------------------------------------- registry

_DATE_PROPS = {
    'date_from': {'type': 'string', 'description': "Start date, YYYY-MM-DD or 'today'/'yesterday'/'tomorrow'. Defaults to today."},
    'date_to': {'type': 'string', 'description': 'End date (inclusive), YYYY-MM-DD. Defaults to date_from.'},
    'date_basis': {'type': 'string', 'enum': ['slot', 'created'],
                   'description': "'slot' = orders scheduled for those dates (default; what 'today's orders' usually means). "
                                  "'created' = orders booked/created on those dates."},
}

TOOLS = OrderedDict([
    ('order_summary', {
        'handler': order_summary,
        'description': 'Counts of orders (visits) in a date range: total, by status group and by exact status, '
                       'pending without phlebo, payment pending. Use for "how many orders today", "how many completed", '
                       '"cancelled this week", etc.',
        'parameters': {'type': 'object', 'properties': dict(_DATE_PROPS)},
    }),
    ('get_order', {
        'handler': get_order,
        'description': 'Full details of one order: status, slot, patients and tests, phlebo, payment, lab, '
                       'reasons and status timeline. Use when the user gives an order number/ID.',
        'parameters': {'type': 'object', 'properties': {
            'order_ref': {'type': 'string', 'description': 'Order number as the user wrote it, e.g. "NOV-3156623" or just "3156623".'},
        }, 'required': ['order_ref']},
    }),
    ('find_orders', {
        'handler': find_orders,
        'description': 'Search/list orders by patient name, phone or order number, and/or filter by date range, '
                       'status group or phlebo name. Returns total count plus up to 20 rows.',
        'parameters': {'type': 'object', 'properties': {
            'query': {'type': 'string', 'description': 'Patient name, phone digits or part of the order number.'},
            'status_group': {'type': 'string', 'enum': list(STATUS_GROUPS)},
            'phlebo_name': {'type': 'string'},
            'limit': {'type': 'integer', 'description': 'Rows to return, max 20. Default 10.'},
            **_DATE_PROPS,
        }},
    }),
    ('attention_needed', {
        'handler': attention_needed,
        'description': 'Problems for one day: orders without phlebo, slot passed but not collected, delayed, '
                       'payment pending after collection. Use for "what needs attention", "any issue today", "pending work".',
        'parameters': {'type': 'object', 'properties': {
            'date': {'type': 'string', 'description': "YYYY-MM-DD or 'today'. Default today."},
        }},
    }),
    ('phlebo_workload', {
        'handler': phlebo_workload,
        'description': 'Per-phlebo order counts for one day, by status group, including orders not yet assigned.',
        'parameters': {'type': 'object', 'properties': {
            'date': {'type': 'string', 'description': "YYYY-MM-DD or 'today'. Default today."},
        }},
    }),
    ('help_search', {
        'handler': help_search,
        'description': 'How-to help for the Pathoconnect CRM: where a feature is, which menu, the steps and the page link. '
                       'Use whenever the user asks how to do something or cannot find something.',
        'parameters': {'type': 'object', 'properties': {
            'query': {'type': 'string', 'description': 'What the user is trying to do, in English keywords.'},
        }, 'required': ['query']},
    }),
])


def openai_tools():
    return [{
        'type': 'function',
        'function': {'name': name, 'description': t['description'], 'parameters': t['parameters']},
    } for name, t in TOOLS.items()]


def run(p, name, args):
    tool = TOOLS.get(name)
    if not tool:
        return {'error': f'Unknown tool {name}'}
    if not isinstance(args, dict):
        args = {}
    return tool['handler'](p, args)
