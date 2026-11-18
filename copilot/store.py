"""Conversations, messages, tool-call audit and the daily limit (copilot_* tables).

Every query is scoped by customer_id + user_id from the token, so one user can
never read or continue another user's conversation.
"""
import json
import logging

from . import db

log = logging.getLogger('copilot.store')

HISTORY_MESSAGES = 10   # earlier turns sent back to the model for context


def user_messages_today(customer_id):
    with db.write() as conn:
        row = db.fetch_one(conn, """
            SELECT COUNT(*) AS n FROM copilot_messages
            WHERE customer_id = %s AND role = 'user' AND created_at >= CURDATE()
        """, (customer_id,))
    return int(row['n'])


def get_conversation(p, conversation_id):
    with db.write() as conn:
        return db.fetch_one(conn, """
            SELECT id, title, created_at, updated_at FROM copilot_conversations
            WHERE id = %s AND customer_id = %s AND user_id = %s AND deleted_at IS NULL
        """, (conversation_id, p.customer_id, p.user_id))


def create_conversation(p, first_message):
    title = ' '.join(first_message.split())[:80]
    with db.write() as conn:
        return db.execute(conn, """
            INSERT INTO copilot_conversations (customer_id, user_id, title, created_at, updated_at)
            VALUES (%s, %s, %s, NOW(), NOW())
        """, (p.customer_id, p.user_id, title))


DRAFT_NOTES = {
    'pending': 'card shown, waiting for the user to click Confirm (order NOT created yet)',
    'creating': 'user clicked Confirm, order is being created',
    'created': 'user confirmed, order CREATED: {order_number}',
    'failed': 'user clicked Confirm but creating the order FAILED: {error}',
    'cancelled': 'user cancelled the card, no order created',
    'replaced': 'replaced by a newer card',
}


def history(conversation_id):
    try:
        rows = _history_rows(conversation_id, with_drafts=True)
    except Exception as e:  # copilot_order_drafts not created yet (002 SQL not run)
        log.warning("history without order cards: %s", e)
        rows = _history_rows(conversation_id, with_drafts=False)
    # The model only sees text, so tell it what became of each order card.
    for r in rows:
        if r.get('draft'):
            status, number, error, expired = (r['draft'].split('|') + ['', '', '', ''])[:4]
            note = 'card expired, no order created' if expired else DRAFT_NOTES.get(status, status)
            r['content'] = f"{r['content'] or ''}\n[Order card: {note.format(order_number=number, error=error)}]"

    # ...and which saved addresses / last tests a returning-customer card offered.
    try:
        from .tools_customer import notes_for_messages
        notes = notes_for_messages([r['id'] for r in rows if r['role'] == 'assistant'])
    except Exception as e:  # copilot_customer_lookups not created yet (003 SQL not run)
        log.warning("history without customer cards: %s", e)
        notes = {}
    for r in rows:
        if r['id'] in notes:
            r['content'] = f"{r['content'] or ''}\n{notes[r['id']]}"

    # ...and what became of each reschedule / cancel / assign / payment-link card.
    try:
        from .tools_actions import notes_for_messages as action_notes
        notes = action_notes([r['id'] for r in rows if r['role'] == 'assistant'])
    except Exception as e:  # copilot_action_drafts not created yet (004 SQL not run)
        log.warning("history without action cards: %s", e)
        notes = {}
    for r in rows:
        if r['id'] in notes:
            r['content'] = f"{r['content'] or ''}\n{notes[r['id']]}"

    # The model expects the history to start with a user turn.
    while rows and rows[0]['role'] != 'user':
        rows.pop(0)
    return rows


def _history_rows(conversation_id, with_drafts):
    draft_sql = """(SELECT CONCAT_WS('|', d.status, IFNULL(d.order_number, ''), IFNULL(d.error, ''),
                                    IF(d.status = 'pending' AND d.expires_at < UTC_TIMESTAMP(), 'expired', ''))
                   FROM copilot_order_drafts d WHERE d.message_id = m.id ORDER BY d.id DESC LIMIT 1)""" \
        if with_drafts else 'NULL'
    with db.write() as conn:
        rows = db.fetch_all(conn, f"""
            SELECT m.id, m.role, m.content, {draft_sql} AS draft
            FROM copilot_messages m
            WHERE m.conversation_id = %s AND m.error IS NULL
            ORDER BY m.id DESC LIMIT %s
        """, (conversation_id, HISTORY_MESSAGES))
    rows.reverse()
    return rows


def save_turn(p, conversation_id, user_text, result=None, conf=None, error=None):
    """Writes the user message, the assistant reply (or error) and the tool audit.
    Returns the assistant message id."""
    conf = conf or {}
    usage = (result or {}).get('usage') or {}
    with db.write() as conn:
        db.execute(conn, """
            INSERT INTO copilot_messages (conversation_id, customer_id, user_id, role, content, created_at)
            VALUES (%s, %s, %s, 'user', %s, NOW())
        """, (conversation_id, p.customer_id, p.user_id, user_text))
        message_id = db.execute(conn, """
            INSERT INTO copilot_messages
                (conversation_id, customer_id, user_id, role, content, provider, model,
                 prompt_tokens, completion_tokens, latency_ms, error, created_at)
            VALUES (%s, %s, %s, 'assistant', %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (conversation_id, p.customer_id, p.user_id,
              (result or {}).get('reply') if not error else error,
              conf.get('provider'), conf.get('model'),
              usage.get('prompt'), usage.get('completion'),
              (result or {}).get('latency_ms'), error[:500] if error else None))
        for call in (result or {}).get('tool_calls') or []:
            db.execute(conn, """
                INSERT INTO copilot_tool_calls
                    (message_id, conversation_id, customer_id, user_id, tool, arguments, ok, error, duration_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (message_id, conversation_id, p.customer_id, p.user_id, call['name'],
                  json.dumps(call['arguments'], ensure_ascii=False, default=str)[:2000],
                  1 if call['ok'] else 0, (call['error'] or '')[:500] or None, call['duration_ms']))
        db.execute(conn, "UPDATE copilot_conversations SET updated_at = NOW() WHERE id = %s", (conversation_id,))
    return message_id


def list_conversations(p, limit=30):
    with db.write() as conn:
        return db.fetch_all(conn, """
            SELECT id, title, updated_at FROM copilot_conversations
            WHERE customer_id = %s AND user_id = %s AND deleted_at IS NULL
            ORDER BY updated_at DESC, id DESC LIMIT %s
        """, (p.customer_id, p.user_id, limit))


def conversation_messages(conversation_id):
    with db.write() as conn:
        return db.fetch_all(conn, """
            SELECT id, role, content, error IS NOT NULL AS failed, created_at FROM copilot_messages
            WHERE conversation_id = %s ORDER BY id
        """, (conversation_id,))


def delete_conversation(p, conversation_id):
    with db.write() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE copilot_conversations SET deleted_at = NOW()
                WHERE id = %s AND customer_id = %s AND user_id = %s AND deleted_at IS NULL
            """, (conversation_id, p.customer_id, p.user_id))
            return cur.rowcount
