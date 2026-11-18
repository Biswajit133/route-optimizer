"""Read-only insight tools: money (finance, payables, receivables), pending reports, test info, phlebo
performance, order trend, coupons and self-booking links.

Same rules as tools_read: visits are always limited by scope.visit_scope (the
CRM order-list rules for this user), SQL is parameterized, phones never leave,
results are small. Each tool also needs the CRM permission of the screen that
shows the same numbers (finance report, report view, test list, phlebos,
coupon settings).
"""
from collections import OrderedDict
from datetime import timedelta

from . import clock, db
from .scope import visit_scope
from .tools_actions import has_permission
from .tools_order import currency_symbol
from .tools_read import STATUS_GROUPS, _date_filter, order_link

MAX_ROWS = 15
COLLECTED_STATUSES = STATUS_GROUPS['collected'] + STATUS_GROUPS['completed']
DUE_SQL = "(CASE WHEN v.final_amount_to_collect > 0 THEN v.final_amount_to_collect ELSE COALESCE(v.amount_to_collect, 0) END)"
PAID_SQL = "LOWER(COALESCE(v.payment_status, '')) = 'success'"   # cash taken / marked paid / paid online


def _money(x):
    return round(float(x or 0), 2)


def _need(p, *perms):
    """None when the role has any of the permissions, else an error dict."""
    if any(has_permission(p, perm) for perm in perms):
        return None
    return {'error': "This user's role cannot see this. It needs the same permission as the matching CRM screen."}


def _status_in(statuses):
    marks, values = db.in_list(statuses)
    return f"v.status IN ({marks})", values


def _default_from(args, days_back, p):
    """date_from defaults to `days_back` days ago instead of today."""
    if args.get('date_from'):
        return args
    start = clock.today_local(p.tz_offset) - timedelta(days=days_back)
    return dict(args, date_from=start.isoformat(), date_to=args.get('date_to') or clock.today_local(p.tz_offset).isoformat())


# ------------------------------------------------------------------ money

def money_summary(p, args, ctx=None):
    denied = _need(p, 'finance-report')
    if denied:
        return denied
    if not p.customer.get('show_finance_report'):
        return {'error': 'The finance report is not turned on for this lab, so money totals are not shown.'}
    scope_sql, scope_params = visit_scope(p)
    date_sql, date_params, info = _date_filter(p, args)
    collected_sql, collected_params = _status_in(COLLECTED_STATUSES)
    where = f"{scope_sql} AND {date_sql} AND (v.status IS NULL OR v.status != 'Cancelled')"
    with db.read() as conn:
        t = db.fetch_one(conn, f"""
            SELECT COUNT(*) AS orders,
                   SUM(COALESCE(v.total_order_value, 0)) AS value,
                   SUM(CASE WHEN {PAID_SQL} THEN 1 ELSE 0 END) AS paid_orders,
                   SUM(CASE WHEN {PAID_SQL} THEN COALESCE(v.total_order_value, 0) ELSE 0 END) AS paid_value,
                   SUM(CASE WHEN {PAID_SQL} AND UPPER(COALESCE(v.payment_mode, '')) = 'CASH' THEN COALESCE(v.amount_collected, 0) ELSE 0 END) AS cash,
                   SUM(CASE WHEN {PAID_SQL} AND UPPER(COALESCE(v.payment_mode, '')) != 'CASH' THEN COALESCE(v.amount_collected, 0) ELSE 0 END) AS other_collected,
                   SUM(CASE WHEN NOT {PAID_SQL} AND {DUE_SQL} > 0 THEN 1 ELSE 0 END) AS due_orders,
                   SUM(CASE WHEN NOT {PAID_SQL} THEN {DUE_SQL} ELSE 0 END) AS due,
                   SUM(CASE WHEN NOT {PAID_SQL} AND {DUE_SQL} > 0 AND {collected_sql} THEN 1 ELSE 0 END) AS due_after_collection_orders,
                   SUM(CASE WHEN NOT {PAID_SQL} AND {collected_sql} THEN {DUE_SQL} ELSE 0 END) AS due_after_collection
            FROM visits v WHERE {where}
        """, collected_params + collected_params + scope_params + date_params)
        modes = db.fetch_all(conn, f"""
            SELECT COALESCE(NULLIF(UPPER(v.payment_mode), ''), 'NOT SET') AS mode, COUNT(*) AS orders,
                   SUM(COALESCE(v.amount_collected, 0)) AS collected
            FROM visits v WHERE {where} AND {PAID_SQL}
            GROUP BY mode ORDER BY orders DESC
        """, scope_params + date_params)
    cur = currency_symbol(p)
    return {
        'date_range': info,
        'currency': cur,
        'orders_not_cancelled': int(t['orders'] or 0),
        'order_value': _money(t['value']),
        'paid': {'orders': int(t['paid_orders'] or 0), 'order_value': _money(t['paid_value'])},
        'collected_by_phlebo_cash': _money(t['cash']),
        'collected_other_modes': _money(t['other_collected']),
        'paid_by_mode': [{'mode': m['mode'], 'orders': int(m['orders']), 'collected': _money(m['collected'])} for m in modes],
        'still_to_collect': {'orders': int(t['due_orders'] or 0), 'amount': _money(t['due'])},
        'unpaid_after_sample_collected': {'orders': int(t['due_after_collection_orders'] or 0),
                                          'amount': _money(t['due_after_collection'])},
        'meaning': {
            'paid': "payment_status success: cash taken by the phlebo, paid online, pay-to-lab, or marked paid "
                    "(prepaid / pay-to-lab orders have no 'collected' amount; see paid_by_mode)",
            'still_to_collect': 'amount to collect on orders not paid yet (cancelled excluded)',
            'unpaid_after_sample_collected': 'sample already collected or order completed but still unpaid - follow these up',
        },
        'page': '/finance-report',
    }


# ------------------------------------------------------------------ reports

def reports_pending(p, args, ctx=None):
    denied = _need(p, 'orders-view_report')
    if denied:
        return denied
    with db.read() as conn:
        tracked = db.fetch_one(conn, """
            SELECT EXISTS(SELECT 1 FROM order_patient_reports r JOIN visits rv ON rv.id = r.visit_id
                          WHERE rv.customer_id = %s AND r.deleted_at IS NULL)
                OR EXISTS(SELECT 1 FROM visit_patients vp JOIN visits pv ON pv.id = vp.visit_id
                          WHERE pv.customer_id = %s AND vp.report_available = 1) AS yes
        """, (p.customer_id, p.customer_id))
        if not int(tracked['yes'] or 0):
            return {'tracked': False,
                    'note': "This lab has never uploaded a report or marked one available in Pathoconnect, so report "
                            "status is not tracked here. Say so; reports are probably sent from the lab's own system."}
        args = _default_from(args, 7, p)
        scope_sql, scope_params = visit_scope(p)
        date_sql, date_params, info = _date_filter(p, args)
        status_sql, status_params = _status_in(COLLECTED_STATUSES)
        where = f"""{scope_sql} AND {date_sql} AND {status_sql}
            AND NOT EXISTS (SELECT 1 FROM order_patient_reports r WHERE r.visit_id = v.id AND r.deleted_at IS NULL)
            AND NOT EXISTS (SELECT 1 FROM visit_patients vp WHERE vp.visit_id = v.id AND vp.report_available = 1)"""
        params = scope_params + date_params + status_params
        count = db.fetch_one(conn, f"SELECT COUNT(*) AS n FROM visits v WHERE {where}", params)['n']
        rows = db.fetch_all(conn, f"""
            SELECT v.id AS visit_id, v.order_id, v.order_number, v.status, v.slot_date, v.main_patient_name
            FROM visits v WHERE {where} ORDER BY v.slot_date, v.id LIMIT {MAX_ROWS}
        """, params)
    today = clock.today_local(p.tz_offset)
    return {
        'tracked': True,
        'date_range': info,
        'orders_without_report': int(count),
        'oldest_first': [{
            'order_number': r['order_number'], 'status': r['status'], 'patient': (r['main_patient_name'] or '').strip() or None,
            'collection_date': clock.iso(r['slot_date']),
            'days_waiting': (today - (r['slot_date'].date() if hasattr(r['slot_date'], 'date') else r['slot_date'])).days
            if r['slot_date'] else None,
            'link': order_link(r['order_id'], r['visit_id']),
        } for r in rows],
        'meaning': 'sample collected or order completed, but no report uploaded and none marked available',
    }


# ------------------------------------------------------------------ tests

PREPARATION = {'fasting': 'Fasting required', 'random': 'No fasting needed', 'both': 'Fasting or non-fasting both fine'}


def test_info(p, args, ctx=None):
    denied = _need(p, 'customer-dos-view', 'orders-create', 'orders-view')
    if denied:
        return denied
    query = ' '.join(str(args.get('query') or '').split())
    if len(query) < 2:
        return {'error': 'Give at least 2 letters of the test name or code.'}
    words = [w for w in query.replace(',', ' ').split() if w.lower() not in ('test', 'tests', 'package', 'price', 'the')][:5]
    base = """SELECT id, test_name, test_code, test_price, discount_price, is_package, preparation, description,
                     test_includes, sample_or_qty, schedule_or_cut_off, special_instruction, is_booking,
                     speciment_1, speciment_1_vials, speciment_2, speciment_2_vials, speciment_3, speciment_3_vials
              FROM customer_dos WHERE customer_id = %s AND status = 'Active' AND deleted_at IS NULL AND ({where})
              ORDER BY (LOWER(TRIM(test_name)) = %s OR LOWER(TRIM(test_code)) = %s) DESC, is_popular DESC, test_name LIMIT 5"""
    exact = query.lower()
    with db.read() as conn:
        rows = db.fetch_all(conn, base.format(where="test_name LIKE %s OR test_code LIKE %s"),
                            (p.customer_id, f'%{query}%', f'%{query}%', exact, exact))
        if not rows and words:
            rows = db.fetch_all(conn, base.format(where=' AND '.join(['test_name LIKE %s'] * len(words))),
                                (p.customer_id, *[f'%{w}%' for w in words], exact, exact))
        children = {}
        pkg_ids = [r['id'] for r in rows if r['is_package']]
        if pkg_ids:
            marks, values = db.in_list(pkg_ids)
            for c in db.fetch_all(conn, f"""
                SELECT pt.test_package_id, t.test_name FROM package_tests pt JOIN customer_dos t ON t.id = pt.test_id
                WHERE pt.test_package_id IN ({marks}) AND pt.deleted_at IS NULL ORDER BY t.test_name
            """, values):
                children.setdefault(c['test_package_id'], []).append(str(c['test_name']).strip())
    if not rows:
        return {'found': False, 'note': f'No active test matched "{query}". Ask for another name or the code.'}
    cur = currency_symbol(p)
    out = []
    for r in rows:
        prep = str(r['preparation'] or '').strip()
        samples = [f"{r[f'speciment_{i}']}" + (f" x{r[f'speciment_{i}_vials']}" if r[f'speciment_{i}_vials'] else '')
                   for i in (1, 2, 3) if str(r[f'speciment_{i}'] or '').strip()]
        includes = children.get(r['id']) or [x.strip() for x in str(r['test_includes'] or '').replace('\n', ',').split(',') if x.strip()]
        out.append({
            'name': str(r['test_name']).strip(), 'code': r['test_code'] or None, 'package': bool(r['is_package']),
            'price': _money(r['discount_price']), 'mrp': _money(r['test_price']), 'currency': cur,
            'preparation': PREPARATION.get(prep.lower(), prep) or None,
            'sample': ', '.join(samples) or (r['sample_or_qty'] or None),
            'report_time': r['schedule_or_cut_off'] or None,
            'includes': includes[:40],
            'about': (str(r['description'] or '').strip()[:300]) or None,
            'instruction': (str(r['special_instruction'] or '').strip()[:200]) or None,
            'bookable_from_crm': bool(r['is_booking']),
        })
    return {'found': True, 'tests': out, 'page': '/customer-dos'}


# ------------------------------------------------------------------ phlebos

def phlebo_performance(p, args, ctx=None):
    denied = _need(p, 'phlebos-view', 'dashboards-phlebos')
    if denied:
        return denied
    args = _default_from(args, 29, p)
    scope_sql, scope_params = visit_scope(p)
    date_sql, date_params, info = _date_filter(p, args)
    collected_sql, collected_params = _status_in(COLLECTED_STATUSES)
    name = str(args.get('phlebo_name') or '').strip()
    extra, extra_params = '', []
    if name:
        extra, extra_params = "AND CONCAT_WS(' ', ph.first_name, ph.last_name) LIKE %s", ['%' + name + '%']
    with db.read() as conn:
        rows = db.fetch_all(conn, f"""
            SELECT v.phlebo_id, TRIM(CONCAT_WS(' ', ph.first_name, ph.last_name)) AS name, COUNT(*) AS assigned,
                   SUM(CASE WHEN {collected_sql} THEN 1 ELSE 0 END) AS collected,
                   SUM(CASE WHEN v.status = 'Completed' THEN 1 ELSE 0 END) AS completed,
                   SUM(CASE WHEN v.status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled,
                   SUM(CASE WHEN v.is_delayed = 1 THEN 1 ELSE 0 END) AS delayed_n,
                   SUM(CASE WHEN EXISTS (SELECT 1 FROM visit_delay_risk dr WHERE dr.visit_id = v.id AND dr.severity = 'high')
                            THEN 1 ELSE 0 END) AS high_delay_risk,
                   COUNT(DISTINCT DATE(v.slot_date)) AS working_days
            FROM visits v JOIN phlebos ph ON ph.id = v.phlebo_id
            WHERE {scope_sql} AND {date_sql} AND v.phlebo_id > 0 {extra}
            GROUP BY v.phlebo_id, name ORDER BY collected DESC, assigned DESC LIMIT 40
        """, collected_params + scope_params + date_params + extra_params)
    people = []
    for r in rows:
        assigned, collected = int(r['assigned']), int(r['collected'])
        open_ = assigned - collected - int(r['cancelled'])
        people.append({
            'phlebo': r['name'] or f"Phlebo {r['phlebo_id']}", 'assigned': assigned, 'collected': collected,
            'completed': int(r['completed']), 'cancelled': int(r['cancelled']), 'not_collected_yet': max(open_, 0),
            'delayed': int(r['delayed_n']), 'high_delay_risk': int(r['high_delay_risk']),
            'working_days': int(r['working_days']),
            'collected_per_day': round(collected / int(r['working_days']), 1) if int(r['working_days']) else 0,
        })
    return {
        'date_range': info, 'phlebos': people,
        'meaning': {'collected': 'sample collected or later (incl. completed)',
                    'delayed': 'flagged delayed by the delay alert',
                    'high_delay_risk': 'delay prediction marked high risk'},
        'page': '/phlebos',
    }


# ------------------------------------------------------------------ trend

def _counts(conn, p, args):
    scope_sql, scope_params = visit_scope(p)
    date_sql, date_params, info = _date_filter(p, args)
    r = db.fetch_one(conn, f"""
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN v.status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled,
               SUM(CASE WHEN v.status = 'Completed' THEN 1 ELSE 0 END) AS completed
        FROM visits v WHERE {scope_sql} AND {date_sql}
    """, scope_params + date_params)
    return info, {k: int(r[k] or 0) for k in ('total', 'cancelled', 'completed')}


def _pct(now, before):
    if not before:
        return None
    return round((now - before) * 100.0 / before, 1)


def order_trend(p, args, ctx=None):
    basis = args.get('date_basis') if args.get('date_basis') in ('slot', 'created') else 'slot'
    start, end = clock.date_range(args.get('date_from'), args.get('date_to'), p.tz_offset)
    if args.get('compare_from'):
        c_start, c_end = clock.date_range(args.get('compare_from'), args.get('compare_to'), p.tz_offset)
    else:  # the same number of days just before
        days = (end - start).days + 1
        c_end = start - timedelta(days=1)
        c_start = c_end - timedelta(days=days - 1)
    group = args.get('group_by') if args.get('group_by') in ('day', 'week', 'month') else None
    cur_args = {'date_from': start.isoformat(), 'date_to': end.isoformat(), 'date_basis': basis}
    with db.read() as conn:
        cur_info, cur = _counts(conn, p, cur_args)
        prev_info, prev = _counts(conn, p, {'date_from': c_start.isoformat(), 'date_to': c_end.isoformat(),
                                            'date_basis': basis})
        series = []
        if group:
            scope_sql, scope_params = visit_scope(p)
            date_sql, date_params, _ = _date_filter(p, cur_args)
            col = 'v.slot_date' if basis == 'slot' else 'v.created_at'
            key = {'day': f"DATE({col})", 'week': f"YEARWEEK({col}, 3)", 'month': f"DATE_FORMAT({col}, '%%Y-%%m')"}[group]
            first = f"MIN(DATE({col}))"
            series = [{'period': str(r['k']), 'starts': clock.iso(r['first']), 'orders': int(r['n'])}
                      for r in db.fetch_all(conn, f"""
                          SELECT {key} AS k, {first} AS first, COUNT(*) AS n FROM visits v
                          WHERE {scope_sql} AND {date_sql} GROUP BY k ORDER BY k LIMIT 40
                      """, scope_params + date_params)]
    return {
        'basis': cur_info['basis'],
        'period': {'from': cur_info['from'], 'to': cur_info['to'], **cur},
        'compared_with': {'from': prev_info['from'], 'to': prev_info['to'], **prev},
        'change_pct': {'total': _pct(cur['total'], prev['total']),
                       'completed': _pct(cur['completed'], prev['completed']),
                       'cancelled': _pct(cur['cancelled'], prev['cancelled'])},
        'series': series,
        'note': 'Counts include cancelled orders in total. A period that has not ended yet (e.g. this month) is '
                'compared with a full earlier period - say so when it matters.',
    }


# ------------------------------------------------------------------ coupons + self-booking links

def coupon_booking_summary(p, args, ctx=None):
    args = _default_from(args, 29, p)
    start, end = clock.date_range(args.get('date_from'), args.get('date_to'), p.tz_offset)
    lo, hi = clock.created_at_bounds(start, end, p.tz_offset)
    out = OrderedDict(date_range={'from': start.isoformat(), 'to': end.isoformat(), 'basis': 'booked (created) date'})
    cur = currency_symbol(p)
    out['currency'] = cur
    scope_sql, scope_params = visit_scope(p)
    with db.read() as conn:
        if p.customer.get('coupon_feature') and has_permission(p, 'setting-coupon'):
            coupons = db.fetch_all(conn, """
                SELECT c.coupon_code, c.coupon_type, c.coupon_value, c.status, c.coupon_active_from, c.coupon_expires_on,
                       c.max_num_usage,
                       (SELECT COUNT(*) FROM coupon_usages cu JOIN orders o ON o.id = cu.order_id
                         WHERE cu.coupon_code = c.coupon_code AND o.customer_id = c.customer_id) AS used_total,
                       (SELECT COUNT(*) FROM coupon_usages cu JOIN orders o ON o.id = cu.order_id
                         WHERE cu.coupon_code = c.coupon_code AND o.customer_id = c.customer_id
                           AND o.created_at >= %s AND o.created_at < %s) AS used_in_range
                FROM coupons c WHERE c.customer_id = %s AND c.deleted_at IS NULL
                ORDER BY used_in_range DESC, c.id DESC LIMIT 25
            """, (lo, hi, p.customer_id))
            today = clock.today_local(p.tz_offset).isoformat()
            out['coupons'] = [{
                'code': c['coupon_code'], 'type': c['coupon_type'], 'value': _money(c['coupon_value']),
                'status': c['status'],
                'expired': bool(c['coupon_expires_on']) and str(c['coupon_expires_on'])[:10] < today,
                'expires_on': clock.iso(c['coupon_expires_on']), 'max_uses': c['max_num_usage'],
                'used_in_range': int(c['used_in_range'] or 0), 'used_total': int(c['used_total'] or 0),
            } for c in coupons]
            out['coupons_page'] = '/setting/coupons'
        else:
            out['coupons'] = None
            out['coupons_note'] = 'Coupons are off for this lab or this role cannot see coupon settings.'

        links = db.fetch_one(conn, """
            SELECT COUNT(*) AS n, SUM(is_active = 1) AS active FROM public_booking_links WHERE customer_id = %s
        """, (p.customer_id,))
        b = db.fetch_one(conn, f"""
            SELECT COUNT(*) AS orders, SUM(COALESCE(bo.amount, 0)) AS amount, SUM(COALESCE(bo.discount, 0)) AS discount,
                   SUM(CASE WHEN LOWER(COALESCE(bo.payment_status, '')) IN ('paid', 'success', 'captured') THEN 1 ELSE 0 END) AS paid_online,
                   SUM(CASE WHEN bo.payment_method = 'online' THEN 1 ELSE 0 END) AS chose_online,
                   SUM(CASE WHEN bo.is_new_enduser = 1 THEN 1 ELSE 0 END) AS new_customers,
                   SUM(CASE WHEN COALESCE(bo.coupon_code, '') != '' THEN 1 ELSE 0 END) AS with_coupon,
                   SUM(CASE WHEN v.status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled
            FROM public_booking_orders bo JOIN visits v ON v.id = bo.visit_id
            WHERE bo.customer_id = %s AND bo.created_at >= %s AND bo.created_at < %s AND {scope_sql}
        """, [p.customer_id, lo, hi] + scope_params)
        by_source = db.fetch_all(conn, f"""
            SELECT COALESCE(bo.source, 'link') AS source, COUNT(*) AS orders FROM public_booking_orders bo
            JOIN visits v ON v.id = bo.visit_id
            WHERE bo.customer_id = %s AND bo.created_at >= %s AND bo.created_at < %s AND {scope_sql}
            GROUP BY source
        """, [p.customer_id, lo, hi] + scope_params)
    out['self_booking'] = {
        'links': int(links['n'] or 0), 'active_links': int(links['active'] or 0),
        'orders': int(b['orders'] or 0), 'amount': _money(b['amount']), 'discount_given': _money(b['discount']),
        'chose_online_payment': int(b['chose_online'] or 0), 'paid_online': int(b['paid_online'] or 0),
        'new_customers': int(b['new_customers'] or 0), 'with_coupon': int(b['with_coupon'] or 0),
        'cancelled': int(b['cancelled'] or 0),
        'by_source': [{'source': r['source'], 'orders': int(r['orders'])} for r in by_source],
    }
    return out


# ------------------------------------------------------------------ payables / receivables
# Same numbers as the CRM Payables and Receivables screens (orders service
# CustomerOrder::payableReportList + calculateTotals, AccountReceivableController
# ::accountReceivableList). Like those screens they cover the whole account, not
# the order-list scope, and they need the screen's own permission.

def _dropped_sql(statuses):
    marks, values = db.in_list(statuses)
    return (f"EXISTS (SELECT 1 FROM visit_status vs WHERE vs.visit_id = v.id AND vs.deleted_at IS NULL "
            f"AND vs.status IN ({marks}))"), values


def _slot_range(p, args):
    start, end = clock.date_range(args.get('date_from'), args.get('date_to'), p.tz_offset)
    return start, end, {'from': start.isoformat(), 'to': end.isoformat(), 'basis': 'collection (slot) date'}


def payable_summary(p, args, ctx=None):
    """What the lab owes its phlebos: visit charge + travel charge of visits that
    reached Dropped In Lab or Completed, and how much of it is batched / paid."""
    denied = _need(p, 'account-payable')
    if denied:
        return denied
    start, end, info = _slot_range(p, args)
    status_sql, status_params = _dropped_sql(['Dropped In Lab', 'Completed'])
    name = str(args.get('phlebo_name') or '').strip()
    extra, extra_params = '', []
    if name:
        extra, extra_params = "AND CONCAT_WS(' ', ph.first_name, ph.last_name) LIKE %s", ['%' + name + '%']
    with db.read() as conn:
        visits = db.fetch_all(conn, f"""
            SELECT v.id, v.phlebo_id, TRIM(CONCAT_WS(' ', ph.first_name, ph.last_name)) AS phlebo,
                   COALESCE(v.visit_charge, 0) AS visit_charge, COALESCE(v.phlebo_distance_charge, 0) AS per_km,
                   v.batch_number, v.batch_created, pb.status AS batch_status
            FROM visits v
            LEFT JOIN phlebos ph ON ph.id = v.phlebo_id
            LEFT JOIN payable_batches pb ON pb.batch_number = v.batch_number AND pb.customer_id = v.customer_id
                                         AND pb.deleted_at IS NULL
            WHERE v.customer_id = %s AND v.is_active = 1 AND v.deleted_at IS NULL
              AND DATE(v.slot_date) BETWEEN %s AND %s AND {status_sql} {extra}
        """, [p.customer_id, start.isoformat(), end.isoformat()] + status_params + extra_params)
        legs = {}
        ids = [v['id'] for v in visits]
        for i in range(0, len(ids), 1000):
            marks, values = db.in_list(ids[i:i + 1000])
            for d in db.fetch_all(conn, f"""
                SELECT visit_id, distance_value FROM phlebo_distances
                WHERE visit_id IN ({marks}) AND deleted_at IS NULL ORDER BY visit_id, id
            """, values):
                legs.setdefault(d['visit_id'], []).append(float(d['distance_value'] or 0))

    totals = {'visits': 0, 'collection': 0.0, 'travel': 0.0, 'km': 0.0}
    state = {'paid': [0, 0.0], 'batched_unpaid': [0, 0.0], 'not_batched': [0, 0.0]}
    people = {}
    for v in visits:
        rows = legs.get(v['id'], [])
        metres = sum(rows[i] for i in (1, 3, 5) if i < len(rows))      # 2nd, 4th, 6th leg, as the report
        travel = round(metres / 1000 * float(v['per_km']), 2)
        charge = float(v['visit_charge']) + travel
        batched = str(v['batch_created'] or '').strip() not in ('', '0') and bool(v['batch_number'])
        key = 'paid' if batched and str(v['batch_status']) == '1' else ('batched_unpaid' if batched else 'not_batched')
        state[key][0] += 1
        state[key][1] += charge
        totals['visits'] += 1
        totals['collection'] += float(v['visit_charge'])
        totals['travel'] += travel
        totals['km'] += metres / 1000
        person = people.setdefault(v['phlebo_id'] or 0, {'phlebo': v['phlebo'] or 'Not assigned', 'visits': 0,
                                                        'total': 0.0, 'unpaid': 0.0})
        person['visits'] += 1
        person['total'] += charge
        if key != 'paid':
            person['unpaid'] += charge
    total = totals['collection'] + totals['travel']
    return {
        'date_range': info,
        'currency': currency_symbol(p),
        'visits': totals['visits'],
        'collection_charge': _money(totals['collection']),
        'travel_charge': _money(totals['travel']),
        'travel_km': round(totals['km'], 1),
        'total_payable': _money(total),
        'paid': {'visits': state['paid'][0], 'amount': _money(state['paid'][1])},
        'in_batch_not_paid': {'visits': state['batched_unpaid'][0], 'amount': _money(state['batched_unpaid'][1])},
        'not_in_a_batch_yet': {'visits': state['not_batched'][0], 'amount': _money(state['not_batched'][1])},
        'by_phlebo': sorted(({**x, 'total': _money(x['total']), 'unpaid': _money(x['unpaid'])} for x in people.values()),
                            key=lambda x: -x['total'])[:12],
        'meaning': {
            'total_payable': 'what the lab owes phlebos: visit charge + travel charge (km x per-km rate), for visits '
                             'that reached Dropped In Lab or Completed',
            'paid': 'in a payable batch marked paid',
        },
        'page': '/payable-report',
    }


def receivable_summary(p, args, ctx=None):
    """What partner labs owe: lab price per visit dropped in that lab, paid or
    pending, and how much is not invoiced yet."""
    denied = _need(p, 'account-receivable')
    if denied:
        return denied
    start, end, info = _slot_range(p, args)
    status_sql, status_params = _dropped_sql(['Dropped In Lab'])
    lab = str(args.get('lab_name') or '').strip()
    extra, extra_params = '', []
    if lab:
        extra, extra_params = "AND l.lab_name LIKE %s", ['%' + lab + '%']
    with db.read() as conn:
        rows = db.fetch_all(conn, f"""
            SELECT v.assign_lab_id AS lab_id, COALESCE(l.lab_name, 'No lab') AS lab, COUNT(*) AS visits,
                   SUM(CASE WHEN l.price REGEXP '^[0-9]+(\\.[0-9]+)?$' THEN l.price ELSE 0 END) AS amount,
                   SUM(CASE WHEN v.lab_is_paid = 1 THEN 1 ELSE 0 END) AS paid_visits,
                   SUM(CASE WHEN v.lab_is_paid = 1 AND l.price REGEXP '^[0-9]+(\\.[0-9]+)?$' THEN l.price ELSE 0 END) AS paid,
                   SUM(CASE WHEN COALESCE(v.lab_invoice_created, 0) = 0 THEN 1 ELSE 0 END) AS uninvoiced_visits,
                   SUM(CASE WHEN COALESCE(v.lab_invoice_created, 0) = 0 AND l.price REGEXP '^[0-9]+(\\.[0-9]+)?$'
                            THEN l.price ELSE 0 END) AS uninvoiced
            FROM visits v LEFT JOIN labs l ON l.id = v.assign_lab_id AND l.deleted_at IS NULL
            WHERE v.customer_id = %s AND DATE(v.slot_date) BETWEEN %s AND %s AND {status_sql} {extra}
            GROUP BY v.assign_lab_id, lab ORDER BY amount DESC
        """, [p.customer_id, start.isoformat(), end.isoformat()] + status_params + extra_params)
    sums = {k: sum(float(r[k] or 0) for r in rows) for k in ('visits', 'amount', 'paid_visits', 'paid',
                                                              'uninvoiced_visits', 'uninvoiced')}
    return {
        'date_range': info,
        'currency': currency_symbol(p),
        'visits': int(sums['visits']),
        'total_receivable': _money(sums['amount']),
        'received': {'visits': int(sums['paid_visits']), 'amount': _money(sums['paid'])},
        'pending': {'visits': int(sums['visits'] - sums['paid_visits']), 'amount': _money(sums['amount'] - sums['paid'])},
        'not_invoiced_yet': {'visits': int(sums['uninvoiced_visits']), 'amount': _money(sums['uninvoiced'])},
        'by_lab': [{'lab': r['lab'], 'visits': int(r['visits']), 'amount': _money(r['amount']),
                    'pending': _money(float(r['amount'] or 0) - float(r['paid'] or 0)),
                    'not_invoiced': _money(r['uninvoiced'])} for r in rows[:12]],
        'meaning': {
            'total_receivable': "what partner labs owe: the lab's price per visit dropped in that lab",
            'received': 'visits on a lab invoice marked paid',
            'not_invoiced_yet': 'visits not on any lab invoice yet - create invoices on the Receivables page',
        },
        'page': '/receivable-report',
    }


# ------------------------------------------------------------------ registry

_DATES = {
    'date_from': {'type': 'string', 'description': "Start date YYYY-MM-DD (or 'today'/'yesterday')."},
    'date_to': {'type': 'string', 'description': 'End date YYYY-MM-DD, inclusive.'},
}
_BASIS = {'date_basis': {'type': 'string', 'enum': ['slot', 'created'],
                         'description': "'slot' = by collection date (default), 'created' = by booking date."}}

TOOLS = {
    'money_summary': {
        'handler': money_summary,
        'description': 'Patient payments for a date range (Finance report): order value, paid, cash collected by '
                       'phlebos, other modes, still to collect from patients, unpaid after collection. Default today. '
                       'Use for "cash collected from patients", "patient payment pending". Needs the finance report; '
                       'if refused, use receivable_summary / payable_summary for the lab\'s money instead.',
        'parameters': {'type': 'object', 'properties': {**_DATES, **_BASIS}},
    },
    'payable_summary': {
        'handler': payable_summary,
        'description': "What the lab owes its phlebos (Payables screen): visit + travel charges for visits dropped in "
                       "lab / completed, paid vs in a batch vs not batched, and per phlebo. Default today. "
                       "Use for 'how much to pay phlebos', 'phlebo payout this month', 'unpaid phlebo charges'.",
        'parameters': {'type': 'object', 'properties': {**_DATES, 'phlebo_name': {'type': 'string'}}},
    },
    'receivable_summary': {
        'handler': receivable_summary,
        'description': "What partner labs owe (Receivables screen): lab price per visit dropped in the lab, received "
                       "vs pending, not invoiced yet, per lab. Default today. Use for 'how much will we get from labs', "
                       "'pending from labs', 'earning this month', 'uninvoiced'.",
        'parameters': {'type': 'object', 'properties': {**_DATES, 'lab_name': {'type': 'string'}}},
    },
    'reports_pending': {
        'handler': reports_pending,
        'description': 'Orders whose sample is collected (or completed) but no report is uploaded / marked available. '
                       'Default last 7 days, oldest first. Tells you if this lab does not track reports here.',
        'parameters': {'type': 'object', 'properties': {**_DATES}},
    },
    'test_info': {
        'handler': test_info,
        'description': "A test or package of this lab: price, MRP, fasting/preparation, sample type, report time, what "
                       "a package includes. Use for 'price of thyroid profile', 'does lipid need fasting', 'what is in "
                       "the full body package'.",
        'parameters': {'type': 'object', 'properties': {
            'query': {'type': 'string', 'description': 'Test or package name or code.'}}, 'required': ['query']},
    },
    'phlebo_performance': {
        'handler': phlebo_performance,
        'description': 'Per-phlebo numbers for a date range: assigned, collected, completed, cancelled, delayed, high '
                       'delay risk, working days, collected per day. Default last 30 days. Optional phlebo_name.',
        'parameters': {'type': 'object', 'properties': {
            **_DATES, 'phlebo_name': {'type': 'string'}}},
    },
    'order_trend': {
        'handler': order_trend,
        'description': 'Order count of a period compared with an earlier one (default: the same number of days just '
                       'before), with % change, and optionally a day/week/month series. Use for "more or fewer orders '
                       'than last month", "orders per week". For calendar months pass both periods explicitly.',
        'parameters': {'type': 'object', 'properties': {
            **_DATES, **_BASIS,
            'compare_from': {'type': 'string', 'description': 'Start of the period to compare with, YYYY-MM-DD.'},
            'compare_to': {'type': 'string', 'description': 'End of the period to compare with, YYYY-MM-DD.'},
            'group_by': {'type': 'string', 'enum': ['day', 'week', 'month']},
        }},
    },
    'coupon_booking_summary': {
        'handler': coupon_booking_summary,
        'description': 'Coupons (code, value, status, expiry, times used) and self-booking link orders (count, amount, '
                       'discount, online paid, new customers) for a booking-date range. Default last 30 days.',
        'parameters': {'type': 'object', 'properties': {**_DATES}},
    },
}
