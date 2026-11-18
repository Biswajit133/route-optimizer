"""Dates in the customer's timezone.

slot_date is already the customer's local date. created_at is written by the
PHP services in APP_TIMEZONE (Asia/Kolkata, +05:30), so "booked today" for a
customer in another timezone has to be shifted into IST before comparing.
"""
import re
from datetime import date, datetime, timedelta, timezone

SERVER_OFFSET = '+05:30'  # orders/.env APP_TIMEZONE=Asia/Kolkata


def parse_offset(offset):
    m = re.match(r'^([+-])(\d{1,2}):?(\d{2})$', str(offset or '').strip())
    if not m:
        return timezone(timedelta(hours=5, minutes=30))
    sign = 1 if m.group(1) == '+' else -1
    return timezone(sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3))))


def now_local(offset):
    return datetime.now(timezone.utc).astimezone(parse_offset(offset))


def today_local(offset):
    return now_local(offset).date()


def resolve_date(value, offset, default=None):
    """'today' / 'yesterday' / 'tomorrow' / 'YYYY-MM-DD' -> date. Bad input -> default."""
    today = today_local(offset)
    if value in (None, ''):
        return default
    text = str(value).strip().lower()
    words = {'today': 0, 'yesterday': -1, 'tomorrow': 1}
    if text in words:
        return today + timedelta(days=words[text])
    for fmt in ('%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y'):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return default


def date_range(date_from, date_to, offset, max_days=93):
    """Inclusive (from, to) local dates, defaulting to today, capped in length."""
    today = today_local(offset)
    start = resolve_date(date_from, offset, today)
    end = resolve_date(date_to, offset, start)
    if end < start:
        start, end = end, start
    if (end - start).days > max_days:
        start = end - timedelta(days=max_days)
    return start, end


def created_at_bounds(start, end, offset):
    """Local [start 00:00, end+1 00:00) expressed as server (IST) wall-clock strings."""
    local = parse_offset(offset)
    server = parse_offset(SERVER_OFFSET)
    lo = datetime.combine(start, datetime.min.time(), local).astimezone(server)
    hi = datetime.combine(end + timedelta(days=1), datetime.min.time(), local).astimezone(server)
    fmt = '%Y-%m-%d %H:%M:%S'
    return lo.strftime(fmt), hi.strftime(fmt)


_TIME = re.compile(r'(\d{1,2}):(\d{2})\s*([AaPp][Mm])?')


def slot_end_minutes(slot_time):
    """'19:00 - 20:00' or '07:00 PM - 08:00 PM' -> minutes after midnight of the end. None if unknown."""
    found = _TIME.findall(str(slot_time or ''))
    if not found:
        return None
    hour, minute, ampm = found[-1]
    hour, minute = int(hour), int(minute)
    if ampm:
        ampm = ampm.lower()
        if ampm == 'pm' and hour < 12:
            hour += 12
        if ampm == 'am' and hour == 12:
            hour = 0
    return hour * 60 + minute


def iso(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat(sep=' ') if isinstance(value, datetime) else value.isoformat()
    return value
