"""HTTP API for the CRM chat widget. Every route except /health needs the CRM JWT.

GET    /copilot/health
GET    /copilot/me                      -> is it on for me, suggestions, limit left
POST   /copilot/chat                    {message, conversation_id?}
GET    /copilot/conversations
GET    /copilot/conversations/<id>
DELETE /copilot/conversations/<id>
POST   /copilot/drafts/<id>/confirm     -> creates the order (orders create-order API, user's JWT)
POST   /copilot/drafts/<id>/cancel
POST   /copilot/drafts/<id>/update      one edit from the card (draft_edit.update_draft), no LLM
GET    /copilot/drafts/<id>/slots?date= -> slots of a date for the card's slot editor
GET    /copilot/tests?q=                -> bookable tests for the card's Add test box
"""
import json
import logging
from functools import wraps

from flask import Blueprint, g, jsonify, request

from . import agent, auth, draft_edit, orders_api, providers, settings, store, tools_customer, tools_order

log = logging.getLogger('copilot.routes')

bp = Blueprint('copilot', __name__, url_prefix='/copilot')

MAX_MESSAGE_CHARS = 1500

SUGGESTIONS = [
    "How many orders today?",
    "How many orders were completed today?",
    "What needs attention today?",
    "Show today's orders without phlebo",
    "How do I add a new test?",
    "Book a home collection order",
]


def fail(message, status=400, code=None):
    body = {'success': False, 'message': message, 'error': message}
    if code:
        body['code'] = code
    return jsonify(body), status


def requires_user(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            g.principal = auth.authenticate()
        except auth.AuthError as e:
            return fail(e.message, e.status, e.code)
        return fn(*args, **kwargs)
    return wrapper


@bp.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'copilot'})


@bp.route('/me', methods=['GET'])
@requires_user
def me():
    p = g.principal
    conf = settings.for_customer(p.customer_id)
    used = None
    try:
        used = store.user_messages_today(p.customer_id)
    except Exception as e:
        log.warning("usage count failed: %s", e)
    limit = conf['daily_message_limit']
    return jsonify({'success': True, 'data': {
        'enabled': True,
        'user_name': p.user_name,
        'customer_name': p.customer_name,
        'llm': settings.public_view(conf),
        'daily_limit': limit or None,
        'used_today': used,
        'suggestions': SUGGESTIONS,
    }})


@bp.route('/chat', methods=['POST'])
@requires_user
def chat():
    p = g.principal
    body = request.get_json(silent=True) or {}
    message = str(body.get('message') or '').strip()
    if not message:
        return fail("Type a question first.")
    if len(message) > MAX_MESSAGE_CHARS:
        return fail(f"Please keep the question under {MAX_MESSAGE_CHARS} characters.")

    conf = settings.for_customer(p.customer_id)
    try:
        settings.require_ready(conf)
    except settings.SettingsError as e:
        log.error("Copilot not ready for customer %s: %s", p.customer_id, e)
        return fail("Patho Assist is not configured yet. Please contact support.", 503, 'COPILOT_NOT_CONFIGURED')

    limit = conf['daily_message_limit']
    if limit and store.user_messages_today(p.customer_id) >= limit:
        return fail("Today's Patho Assist limit for your account is used up. It resets tomorrow.", 429, 'COPILOT_LIMIT')

    conversation_id = body.get('conversation_id')
    if conversation_id:
        if not str(conversation_id).isdigit() or not store.get_conversation(p, int(conversation_id)):
            return fail("Conversation not found.", 404)
        conversation_id = int(conversation_id)
        history = store.history(conversation_id)
    else:
        conversation_id = store.create_conversation(p, message)
        history = []

    ctx = {'conversation_id': conversation_id}
    try:
        result = agent.run(p, conf, history, message, ctx)
    except providers.LLMError as e:
        store.save_turn(p, conversation_id, message, conf=conf, error=str(e))
        return fail(str(e), 502, 'COPILOT_LLM_ERROR')
    except Exception as e:
        log.exception("Copilot chat failed")
        store.save_turn(p, conversation_id, message, conf=conf, error=f"{e.__class__.__name__}: {e}")
        return fail("Something went wrong while answering. Please try again.", 500)

    message_id = store.save_turn(p, conversation_id, message, result=result, conf=conf)
    card = None
    if ctx.get('drafts'):
        tools_order.attach_message(ctx['drafts'], message_id)
        card = tools_order.card_from_row(tools_order.get_draft(p, ctx['drafts'][-1]))
    customer = None
    if ctx.get('lookups'):
        tools_customer.attach_message(ctx['lookups'], message_id)
        customer = tools_customer.get_card(ctx['lookups'][-1])
    return jsonify({'success': True, 'data': {
        'conversation_id': conversation_id,
        'message_id': message_id,
        'reply': result['reply'],
        'card': card,
        'customer': customer,
        'tools_used': [c['name'] for c in result['tool_calls']],
    }})


@bp.route('/conversations', methods=['GET'])
@requires_user
def conversations():
    rows = store.list_conversations(g.principal)
    return jsonify({'success': True, 'data': [
        {'id': r['id'], 'title': r['title'], 'updated_at': r['updated_at'].isoformat(sep=' ') if r['updated_at'] else None}
        for r in rows
    ]})


@bp.route('/conversations/<int:conversation_id>', methods=['GET'])
@requires_user
def conversation(conversation_id):
    conv = store.get_conversation(g.principal, conversation_id)
    if not conv:
        return fail("Conversation not found.", 404)
    rows = store.conversation_messages(conversation_id)
    try:
        cards = tools_order.cards_for_messages([r['id'] for r in rows if r['role'] == 'assistant'])
    except Exception as e:  # copilot_order_drafts not created yet
        log.warning("order cards unavailable: %s", e)
        cards = {}
    try:
        customers = tools_customer.cards_for_messages([r['id'] for r in rows if r['role'] == 'assistant'])
    except Exception as e:  # copilot_customer_lookups not created yet
        log.warning("customer cards unavailable: %s", e)
        customers = {}
    return jsonify({'success': True, 'data': {
        'id': conv['id'], 'title': conv['title'],
        'messages': [{'id': r['id'], 'role': r['role'], 'content': r['content'], 'failed': bool(r['failed']),
                      'card': cards.get(int(r['id'])), 'customer': customers.get(int(r['id']))}
                     for r in rows],
    }})


@bp.route('/conversations/<int:conversation_id>', methods=['DELETE'])
@requires_user
def delete_conversation(conversation_id):
    if not store.delete_conversation(g.principal, conversation_id):
        return fail("Conversation not found.", 404)
    return jsonify({'success': True})


@bp.route('/drafts/<int:draft_id>/confirm', methods=['POST'])
@requires_user
def confirm_draft(draft_id):
    """The only place an order is created: the user clicked Confirm on a card."""
    p = g.principal
    allowed, reason = tools_order.can_create_orders(p)
    if not allowed:
        return fail(reason, 403)
    row, problem = tools_order.claim(p, draft_id)
    if problem:
        return fail(problem[0], problem[1])

    try:
        order = orders_api.create_order(json.loads(row['payload']), request.headers.get('Authorization', ''))
    except orders_api.OrderApiError as e:
        log.warning("Copilot draft %s: create-order failed: %s", draft_id, e)
        tools_order.finish(draft_id, 'failed', error=str(e))
        return jsonify({'success': False, 'message': str(e), 'error': str(e), 'code': 'ORDER_CREATE_FAILED',
                        'card': tools_order.card_from_row(tools_order.get_draft(p, draft_id))}), 502
    except Exception as e:
        log.exception("Copilot draft %s: create-order crashed", draft_id)
        tools_order.finish(draft_id, 'failed', error=f"{e.__class__.__name__}: {e}")
        message = "Something went wrong while creating the order. Check the Orders list before trying again."
        return jsonify({'success': False, 'message': message, 'error': message, 'code': 'ORDER_CREATE_FAILED',
                        'card': tools_order.card_from_row(tools_order.get_draft(p, draft_id))}), 500

    tools_order.finish(draft_id, 'created', order=order)
    log.info("Copilot draft %s: created order %s for customer %s by user %s",
             draft_id, order.get('order_number'), p.customer_id, p.user_id)
    return jsonify({'success': True, 'data': {'card': tools_order.card_from_row(tools_order.get_draft(p, draft_id))}})


@bp.route('/drafts/<int:draft_id>/cancel', methods=['POST'])
@requires_user
def cancel_draft(draft_id):
    p = g.principal
    if not tools_order.cancel(p, draft_id) and not tools_order.get_draft(p, draft_id):
        return fail("Order card not found.", 404)
    return jsonify({'success': True, 'data': {'card': tools_order.card_from_row(tools_order.get_draft(p, draft_id))}})


@bp.route('/drafts/<int:draft_id>/update', methods=['POST'])
@requires_user
def update_draft(draft_id):
    """A click on the card: payment type, another saved address or another matching test."""
    p = g.principal
    allowed, reason = tools_order.can_create_orders(p)
    if not allowed:
        return fail(reason, 403)
    try:
        card = draft_edit.update_draft(p, draft_id, request.get_json(silent=True) or {})
    except draft_edit.EditProblem as e:
        return jsonify({'success': False, 'message': str(e), 'error': str(e),
                        'card': tools_order.card_from_row(tools_order.get_draft(p, draft_id))}), e.status
    return jsonify({'success': True, 'data': {'card': card}})


@bp.route('/drafts/<int:draft_id>/slots', methods=['GET'])
@requires_user
def draft_slots(draft_id):
    try:
        return jsonify({'success': True, 'data': draft_edit.slot_choices(g.principal, draft_id, request.args.get('date'))})
    except draft_edit.EditProblem as e:
        return fail(str(e), e.status)


@bp.route('/tests', methods=['GET'])
@requires_user
def find_tests():
    allowed, reason = tools_order.can_create_orders(g.principal)
    if not allowed:
        return fail(reason, 403)
    return jsonify({'success': True, 'data': draft_edit.find_tests(g.principal, request.args.get('q'))})
